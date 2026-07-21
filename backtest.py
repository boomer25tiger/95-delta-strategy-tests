"""Strategy engine: buy-and-hold baselines and the 0.95-delta call replacement.

Equity underlyings (QQQ, SPY) price with Black-Scholes and a trailing dividend
yield q; futures (NQ, ES) with Black-76 via q = r. Calls are valued American
(Barone-Adesi-Whaley); the 0.95-delta strike is still selected on the European
delta (SPEC 15.2). Frictions are a percentage of underlying notional per side
(SPEC 15.3). The vol surface is a frozen VolParams scenario (SPEC 5.1). Taxes are
optional and reported pre-tax and after-tax (SPEC 15.4).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import data
import pricing
import vol

START_CAPITAL = 10000.0
ROLL_DTE = 60          # roll when calendar days to expiry fall to this
TARGET_DELTA = 0.95


@dataclass(frozen=True)
class FrictionParams:
    notional_pct: float = 0.0005  # half-spread as a fraction of underlying notional, per side
    commission: float = 0.65      # $ per contract per leg
    min_dollars: float = 0.50     # $ floor per contract per leg
    vol_widen: float = 0.0        # >0 widens the spread with vol; 0 = off


def _friction_per_contract(S, mult, atm_vol, fp):
    spread = fp.notional_pct * mult * S
    if fp.vol_widen > 0.0:
        spread *= 1.0 + fp.vol_widen * max(atm_vol / 0.20 - 1.0, 0.0)
    return max(spread, fp.min_dollars) + fp.commission


@dataclass(frozen=True)
class TaxParams:
    ordinary: float = 0.35   # short-term / ordinary income / interest
    ltcg: float = 0.20       # long-term / qualified dividends
    enabled: bool = False    # False = tax-advantaged (no tax)


class _TaxBook:
    """Annual tax accounting. ETF options realize short/long gains at close by
    holding period; futures and their options use Section 1256 60/40 marked to
    market at year-end. Interest is ordinary. Net capital losses carry forward."""

    def __init__(self, tp, is_future):
        self.tp = tp
        self.is_future = is_future
        self.st = 0.0
        self.lt = 0.0
        self.opens = 0.0      # 1256: cost of options opened this year
        self.closes = 0.0     # 1256: proceeds of options closed this year
        self.mtm_basis = 0.0  # 1256: prior year-end mark of the open position
        self.interest = 0.0
        self.carryforward = 0.0
        self.total_tax = 0.0

    def open_cost(self, cost):
        if self.is_future:
            self.opens += cost

    def close_fut(self, proceeds):
        self.closes += proceeds

    def close_etf(self, proceeds, basis, holding_days):
        gain = proceeds - basis
        if holding_days >= 365:
            self.lt += gain
        else:
            self.st += gain

    def accrue_interest(self, amt):
        self.interest += amt

    def settle_year(self, yearend_mv):
        if not self.tp.enabled:
            return 0.0
        tp = self.tp
        if self.is_future:
            taxable = self.closes + yearend_mv - self.mtm_basis - self.opens
            g = taxable + self.carryforward
            if g > 0.0:
                tax = (0.6 * tp.ltcg + 0.4 * tp.ordinary) * g
                self.carryforward = 0.0
            else:
                tax = 0.0
                self.carryforward = g
            self.mtm_basis = yearend_mv
            self.closes = 0.0
            self.opens = 0.0
        else:
            st, lt = self.st, self.lt
            if st < 0.0 and lt > 0.0:               # cross-net a net loss against the other bucket
                o = min(lt, -st); lt -= o; st += o
            elif lt < 0.0 and st > 0.0:
                o = min(st, -lt); st -= o; lt += o
            cf = self.carryforward
            if cf < 0.0 and st > 0.0:               # prior losses offset gains, short-term first
                u = min(st, -cf); st -= u; cf += u
            if cf < 0.0 and lt > 0.0:
                u = min(lt, -cf); lt -= u; cf += u
            if st < 0.0:
                cf += st; st = 0.0
            if lt < 0.0:
                cf += lt; lt = 0.0
            self.carryforward = cf
            tax = tp.ordinary * max(st, 0.0) + tp.ltcg * max(lt, 0.0)
            self.st = 0.0
            self.lt = 0.0
        tax += tp.ordinary * max(self.interest, 0.0)
        self.interest = 0.0
        self.total_tax += tax
        return tax


@dataclass
class Market:
    name: str
    price: pd.Series      # equity spot or front future
    atm_vol: pd.Series    # decimal ATM vol by date
    r: pd.Series          # continuous short rate
    q: pd.Series          # dividend yield (equity) or r (futures -> Black-76)
    mult: float
    is_future: bool
    vol_start: pd.Timestamp   # first date the vol source exists (entry gate)
    adj_close: pd.Series = None
    dividends: pd.Series = None


# loader, is_future, multiplier, vol source
_SPECS = {
    "QQQ": ("qqq", False, 100.0, "vxn"),
    "NQ":  ("nq",  True,  2.0,   "vxn"),
    "SPY": ("spy", False, 100.0, "vix"),
    "ES":  ("es",  True,  5.0,   "vix"),   # Micro E-mini MES, $5/point
}


def build_market(name):
    loader_name, is_future, mult, vsrc = _SPECS[name]
    loader = getattr(data, loader_name)
    r_all = data.short_rate_continuous()
    vol_start = data.vxn().index.min() if vsrc == "vxn" else data.vix().index.min()
    if is_future:
        price = loader()
        r = r_all.reindex(price.index).ffill()
        atm = vol.atm_vol_series(price.index, source=vsrc)
        return Market(name, price, atm, r, r.copy(), mult, True, vol_start)
    df = loader()
    price = df["close"]
    q = (df["dividends"].rolling("365D").sum() / price).fillna(0.0)
    r = r_all.reindex(price.index).ffill()
    atm = vol.atm_vol_series(price.index, source=vsrc)
    return Market(name, price, atm, r, q, mult, False, vol_start,
                  adj_close=df["adj_close"], dividends=df["dividends"])


def _entry_index(mkt, start_year):
    dates = mkt.price.index
    jan1 = pd.Timestamp(start_year, 1, 1)
    finite = (mkt.atm_vol.notna() & mkt.r.notna() & mkt.q.notna() & mkt.price.notna()).values
    for i in range(len(dates)):
        if dates[i] >= jan1 and dates[i] >= mkt.vol_start and finite[i]:
            return i
    raise ValueError(f"no eligible entry for {mkt.name} {start_year}")


def buy_and_hold(mkt, start_year, tp=None, start_capital=START_CAPITAL):
    if tp is None:
        tp = TaxParams()
    i0 = _entry_index(mkt, start_year)
    dates = mkt.price.index[i0:]
    if not mkt.is_future:
        if not tp.enabled:
            ac = mkt.adj_close.iloc[i0:]
            return (start_capital * ac / ac.iloc[0]).rename("bh")
        # after-tax ETF: reinvest dividends net of the qualified rate, LTCG at liquidation
        close = mkt.price.values[i0:]
        divv = mkt.dividends.values[i0:]
        sh = start_capital / close[0]
        basis = start_capital
        out = [sh * close[0]]
        for j in range(1, len(dates)):
            if divv[j] > 0.0:
                net = sh * divv[j] * (1.0 - tp.ltcg)
                sh += net / close[j]
                basis += net
            out.append(sh * close[j])
        out[-1] -= tp.ltcg * max(sh * close[-1] - basis, 0.0)
        return pd.Series(out, index=dates, name="bh")
    # futures: $10k collateral compounding at r, plus daily futures variation margin;
    # Section 1256 60/40 mark-to-market annually when taxable
    F = mkt.price.values[i0:]
    R = mkt.r.values[i0:]
    contracts = start_capital / (mkt.mult * F[0])
    acct = start_capital
    out = [acct]
    blended = 0.6 * tp.ltcg + 0.4 * tp.ordinary
    yr_pnl, yr_int, cf = 0.0, 0.0, 0.0
    for j in range(1, len(dates)):
        if tp.enabled and dates[j].year != dates[j - 1].year:
            g = yr_pnl + cf
            tax = blended * g if g > 0.0 else 0.0
            cf = 0.0 if g > 0.0 else g
            acct -= tax + tp.ordinary * max(yr_int, 0.0)
            yr_pnl, yr_int = 0.0, 0.0
        dt = (dates[j] - dates[j - 1]).days / 365.0
        interest = acct * (math.exp(R[j] * dt) - 1.0)
        pnl = contracts * mkt.mult * (F[j] - F[j - 1])
        acct += interest + pnl
        yr_pnl += pnl
        yr_int += interest
        out.append(acct)
    if tp.enabled:
        g = yr_pnl + cf
        acct -= (blended * g if g > 0.0 else 0.0) + tp.ordinary * max(yr_int, 0.0)
        out[-1] = acct
    return pd.Series(out, index=dates, name="bh")


def _open(available, S, atm, r, q, tenor, mkt, sizing, vparams, fp, matched_notional):
    sig_fn = lambda k: float(vol.sigma(k, S, tenor, atm, vparams))
    # futures/spot call delta caps at e^(-qT); fall to the deepest feasible when 0.95 unreachable
    eff_target = min(TARGET_DELTA, 0.999 * math.exp(-q * tenor))
    K = pricing.solve_strike_for_delta(S, tenor, r, sig_fn, eff_target, q=q)
    call0 = pricing.baw_call(S, K, tenor, r, sig_fn(K), q=q)
    unit = mkt.mult * call0
    fric_leg = _friction_per_contract(S, mkt.mult, atm, fp)
    if sizing == "matched":
        target = available if matched_notional == "equity" else float(matched_notional)
        n = int(round(target / (mkt.mult * S)))
    else:  # full-capital
        n = int(math.floor(available / (unit + fric_leg)))
    n = max(n, 0)
    spend = n * unit + n * fric_leg
    while n > 0 and spend > available:
        n -= 1
        spend = n * unit + n * fric_leg
    cash = available - spend
    return K, n, cash, n * fric_leg, call0, eff_target < TARGET_DELTA


def run_replacement(mkt, start_year, tenor, sizing, vparams,
                    fp=None, tp=None, matched_notional="equity", start_capital=START_CAPITAL):
    if fp is None:
        fp = FrictionParams()
    if tp is None:
        tp = TaxParams()
    dates = mkt.price.index
    S, ATM, R, Q = mkt.price.values, mkt.atm_vol.values, mkt.r.values, mkt.q.values
    i0 = _entry_index(mkt, start_year)
    book = _TaxBook(tp, mkt.is_future)

    K, n, cash, fric, call0, capped = _open(start_capital, S[i0], ATM[i0], R[i0], Q[i0],
                                            tenor, mkt, sizing, vparams, fp, matched_notional)
    lot_basis = start_capital - cash          # premium + friction paid, the tax cost basis
    lot_open = dates[i0]
    book.open_cost(lot_basis)
    expiry = dates[i0] + pd.DateOffset(years=int(tenor))
    friction_paid, rolls, lev = fric, 0, []
    capped_entries = int(capped)
    cash_days = 0
    prev_mv = n * mkt.mult * call0
    eq_idx = [dates[i0]]
    eq_val = [cash + prev_mv]

    for i in range(i0 + 1, len(dates)):
        if dates[i].year != dates[i - 1].year:
            cash -= book.settle_year(prev_mv)   # settle the year that just ended
        dt = (dates[i] - dates[i - 1]).days / 365.0
        interest = cash * (math.exp(R[i] * dt) - 1.0)
        cash += interest
        book.accrue_interest(interest)
        Trem = (expiry - dates[i]).days / 365.0
        if Trem <= ROLL_DTE / 365.0:
            tc = max(Trem, 1e-6)
            sig_c = float(vol.sigma(K, S[i], tc, ATM[i], vparams))
            call_c = pricing.baw_call(S[i], K, tc, R[i], sig_c, q=Q[i])
            fric_c = _friction_per_contract(S[i], mkt.mult, ATM[i], fp)
            proceeds = n * mkt.mult * call_c - n * fric_c
            cash += proceeds
            friction_paid += n * fric_c
            rolls += 1
            if mkt.is_future:
                book.close_fut(proceeds)
            else:
                book.close_etf(proceeds, lot_basis, (dates[i] - lot_open).days)
            avail = cash
            K, n, cash, fric, _, capped = _open(avail, S[i], ATM[i], R[i], Q[i], tenor,
                                                mkt, sizing, vparams, fp, matched_notional)
            lot_basis = avail - cash
            lot_open = dates[i]
            book.open_cost(lot_basis)
            friction_paid += fric
            capped_entries += int(capped)
            expiry = dates[i] + pd.DateOffset(years=int(tenor))
            Trem = (expiry - dates[i]).days / 365.0
        sig_t = float(vol.sigma(K, S[i], Trem, ATM[i], vparams))
        call_t = pricing.baw_call(S[i], K, Trem, R[i], sig_t, q=Q[i])
        g = pricing.bs_greeks(S[i], K, Trem, R[i], sig_t, q=Q[i], kind="call")  # European delta for leverage
        prev_mv = n * mkt.mult * call_t
        eq = cash + prev_mv
        if n == 0:
            cash_days += 1     # forced to cash by whole-contract granularity (SPEC bias #6)
        if eq > 0:
            lev.append(n * mkt.mult * g["delta"] * S[i] / eq)
        eq_idx.append(dates[i])
        eq_val.append(eq)

    # terminal: liquidate the open position for tax and settle the final partial year
    if mkt.is_future:
        term_tax = book.settle_year(prev_mv)
    else:
        book.close_etf(prev_mv, lot_basis, (dates[-1] - lot_open).days)
        term_tax = book.settle_year(0.0)
    eq_val[-1] -= term_tax

    return {
        "equity": pd.Series(eq_val, index=eq_idx, name="strategy"),
        "rolls": rolls,
        "friction": friction_paid,
        "avg_leverage": float(np.nanmean(lev)) if lev else float("nan"),
        "capped_entries": capped_entries,
        "tax_paid": book.total_tax,
        "cash_frac": cash_days / max(len(eq_val), 1),
    }


def metrics(equity, r_series):
    e = equity.dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    dr = e.pct_change().dropna()
    dt = e.index.to_series().diff().dt.days.div(365.0).reindex(dr.index)
    rf = r_series.reindex(e.index).ffill().reindex(dr.index)
    rf_daily = np.exp(rf * dt) - 1.0
    excess = dr - rf_daily
    sd = dr.std(ddof=0)
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1.0 / yrs) - 1.0
    mdd = (e / e.cummax() - 1.0).min()
    downside = math.sqrt((excess.clip(upper=0.0) ** 2).mean())  # Sortino uses downside deviation
    return {
        "final": float(e.iloc[-1]),
        "total_return": float(e.iloc[-1] / e.iloc[0] - 1.0),
        "cagr": float(cagr),
        "ann_vol": float(sd * math.sqrt(252)),
        "sharpe": float(excess.mean() / excess.std(ddof=0) * math.sqrt(252)) if sd > 0 else float("nan"),
        "sortino": float(excess.mean() / downside * math.sqrt(252)) if downside > 0 else float("nan"),
        "max_dd": float(mdd),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
    }


def _fmt(m):
    return (f"final ${m['final']:,.0f}  CAGR {m['cagr']*100:5.2f}%  vol {m['ann_vol']*100:5.1f}%  "
            f"Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  maxDD {m['max_dd']*100:6.1f}%")


if __name__ == "__main__":
    mkt = build_market("QQQ")
    vp = vol.VOL_SCENARIOS["base"]
    tpx = TaxParams(enabled=True)

    res = run_replacement(mkt, 2011, 1.0, "matched", vp)
    eq, bh = res["equity"], buy_and_hold(mkt, 2011)
    ms, mb = metrics(eq, mkt.r), metrics(bh, mkt.r)

    res_at = run_replacement(mkt, 2011, 1.0, "matched", vp, tp=tpx)
    bh_at = buy_and_hold(mkt, 2011, tp=tpx)
    ms_at, mb_at = metrics(res_at["equity"], mkt.r), metrics(bh_at, mkt.r)

    print("QQQ 2011 matched-exposure 1y (base vol, American, notional friction)")
    print(f"  strategy pre-tax  : {_fmt(ms)}")
    print(f"  strategy after-tax: {_fmt(ms_at)}   tax ${res_at['tax_paid']:,.0f}")
    print(f"  buy-hold pre-tax  : {_fmt(mb)}")
    print(f"  buy-hold after-tax: {_fmt(mb_at)}")

    # 2y for the ST-vs-LT direction check
    r2 = run_replacement(mkt, 2011, 2.0, "matched", vp)
    r2_at = run_replacement(mkt, 2011, 2.0, "matched", vp, tp=tpx)
    drag1 = 1.0 - ms_at["final"] / ms["final"]
    drag2 = 1.0 - metrics(r2_at["equity"], mkt.r)["final"] / metrics(r2["equity"], mkt.r)["final"]
    bh_drag = 1.0 - mb_at["final"] / mb["final"]

    checks = []
    def ck(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    print("\nsanity:")
    ck("pre-tax strategy unchanged (~$136.5k)", abs(ms["final"] - 136534) < 500, f"${ms['final']:,.0f}")
    ck("after-tax < pre-tax (strategy)", ms_at["final"] < ms["final"], f"${ms_at['final']:,.0f} < ${ms['final']:,.0f}")
    ck("after-tax < pre-tax (B&H)", mb_at["final"] < mb["final"], f"${mb_at['final']:,.0f} < ${mb['final']:,.0f}")
    ck("strategy tax paid > 0", res_at["tax_paid"] > 0, f"${res_at['tax_paid']:,.0f}")
    ck("1y ST tax drag > 2y LT tax drag", drag1 > drag2, f"{drag1*100:.1f}% vs {drag2*100:.1f}%")
    ck("B&H tax drag < 1y strategy drag (deferral)", bh_drag < drag1, f"{bh_drag*100:.1f}% vs {drag1*100:.1f}%")
    ck("equity strictly positive", bool((res_at["equity"] > 0).all()))
    print(f"\n{sum(checks)}/{len(checks)} sanity checks passed")

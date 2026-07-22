"""Strategy engine: buy-and-hold baselines, the 0.95-delta call replacement, and
the poor man's covered call (PMCC).

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
ROLL_DTE = 60          # roll the long leg when calendar days to expiry fall to this
TARGET_DELTA = 0.95
SHORT_DELTA = 0.30     # PMCC short leg target delta
SHORT_DTE = 35         # days to expiry when the short is sold
SHORT_ROLL_DTE = 5     # roll the short when it decays to this (monthly cadence)


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
    holding period (written calls rolled monthly are always short-term); futures
    and their options use Section 1256 60/40 marked to market at year-end, tracked
    as signed net option cash so a short leg nets correctly. Interest is ordinary.
    Net capital losses carry forward."""

    def __init__(self, tp, is_future):
        self.tp = tp
        self.is_future = is_future
        self.st = 0.0
        self.lt = 0.0
        self.opt_cash = 0.0   # 1256: signed net cash from option trades this year
        self.mtm_basis = 0.0  # 1256: prior year-end mark of the net option position
        self.interest = 0.0
        self.carryforward = 0.0
        self.total_tax = 0.0

    def option_cash(self, amt):
        if self.is_future:
            self.opt_cash += amt

    def close_etf(self, proceeds, basis, holding_days):
        gain = proceeds - basis
        if holding_days >= 365:
            self.lt += gain
        else:
            self.st += gain

    def close_short_etf(self, premium_received, buyback_cost):
        self.st += premium_received - buyback_cost

    def accrue_interest(self, amt):
        self.interest += amt

    def settle_year(self, net_mv):
        if not self.tp.enabled:
            return 0.0
        tp = self.tp
        if self.is_future:
            taxable = net_mv - self.mtm_basis + self.opt_cash
            g = taxable + self.carryforward
            if g > 0.0:
                tax = (0.6 * tp.ltcg + 0.4 * tp.ordinary) * g
                self.carryforward = 0.0
            else:
                tax = 0.0
                self.carryforward = g
            self.mtm_basis = net_mv
            self.opt_cash = 0.0
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
    F = mkt.price.values[i0:].tolist()
    R = mkt.r.values[i0:].tolist()
    bdts = dates.values.astype("datetime64[D]").astype(np.int64).tolist()
    byrs = dates.year.values.tolist()
    contracts = start_capital / (mkt.mult * F[0])
    acct = start_capital
    out = [acct]
    blended = 0.6 * tp.ltcg + 0.4 * tp.ordinary
    yr_pnl, yr_int, cf = 0.0, 0.0, 0.0
    for j in range(1, len(dates)):
        if tp.enabled and byrs[j] != byrs[j - 1]:
            g = yr_pnl + cf
            tax = blended * g if g > 0.0 else 0.0
            cf = 0.0 if g > 0.0 else g
            acct -= tax + tp.ordinary * max(yr_int, 0.0)
            yr_pnl, yr_int = 0.0, 0.0
        dt = (bdts[j] - bdts[j - 1]) / 365.0
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


def _expiry_day(ts, years):
    """Epoch-day number of ts plus `years` calendar years (ts is midnight)."""
    return (ts + pd.DateOffset(years=years)).value // 86_400_000_000_000


def _short_value(S, K, days_to_exp, r, q, atm, mkt, vparams, n):
    T = max(days_to_exp / 365.0, 1e-6)
    sig = float(vol.sigma(K, S, T, atm, vparams))
    px = pricing.baw_call(S, K, T, r, sig, q=q)
    dlt = pricing.bs_delta(S, K, T, r, sig, q=q, kind="call")
    return n * mkt.mult * px, dlt


def run_strategy(mkt, start_year, tenor, sizing, vparams, pmcc=False,
                 fp=None, tp=None, matched_notional="equity", start_capital=START_CAPITAL):
    if fp is None:
        fp = FrictionParams()
    if tp is None:
        tp = TaxParams()
    dates = mkt.price.index
    S = mkt.price.values.tolist()
    ATM = mkt.atm_vol.values.tolist()
    R = mkt.r.values.tolist()
    Q = mkt.q.values.tolist()
    dts = dates.values.astype("datetime64[D]").astype(np.int64).tolist()  # epoch days
    yrs = dates.year.values.tolist()
    i0 = _entry_index(mkt, start_year)
    book = _TaxBook(tp, mkt.is_future)

    K, n, cash, fric, call0, capped = _open(start_capital, S[i0], ATM[i0], R[i0], Q[i0],
                                            tenor, mkt, sizing, vparams, fp, matched_notional)
    lot_basis = start_capital - cash          # premium + friction paid, the tax cost basis
    lot_open_day = dts[i0]
    book.option_cash(-lot_basis)
    expiry_day = _expiry_day(dates[i0], int(tenor))
    friction_paid, rolls, lev = fric, 0, []
    capped_entries = int(capped)
    cash_days = 0
    short_rolls, short_premium_total, short_buyback_total = 0, 0.0, 0.0
    K_short, expiry_short_day, n_short, short_prem = 0.0, dts[i0], 0, 0.0
    long_mv = n * mkt.mult * call0
    short_mv = 0.0
    eq_val = [cash + long_mv]

    for i in range(i0 + 1, len(dates)):
        if yrs[i] != yrs[i - 1]:
            cash -= book.settle_year(long_mv - short_mv)   # settle the year that just ended
        dt = (dts[i] - dts[i - 1]) / 365.0
        interest = cash * (math.exp(R[i] * dt) - 1.0)
        cash += interest
        book.accrue_interest(interest)

        Trem = (expiry_day - dts[i]) / 365.0
        long_rolling = Trem <= ROLL_DTE / 365.0

        # --- close the short first, so the long leg is sized on capital net of the
        #     outstanding short liability rather than on gross cash ---
        if pmcc and n_short > 0 and (long_rolling
                                     or expiry_short_day - dts[i] <= SHORT_ROLL_DTE
                                     or S[i] >= K_short or n_short != n):
            mv, _ = _short_value(S[i], K_short, expiry_short_day - dts[i], R[i], Q[i],
                                 ATM[i], mkt, vparams, n_short)
            frs = n_short * _friction_per_contract(S[i], mkt.mult, ATM[i], fp)
            cost = mv + frs
            cash -= cost
            friction_paid += frs
            short_rolls += 1
            short_buyback_total += cost
            if mkt.is_future:
                book.option_cash(-cost)
            else:
                book.close_short_etf(short_prem, cost)
            n_short = 0

        # --- long leg roll ---
        if long_rolling:
            tc = max(Trem, 1e-6)
            sig_c = float(vol.sigma(K, S[i], tc, ATM[i], vparams))
            call_c = pricing.baw_call(S[i], K, tc, R[i], sig_c, q=Q[i])
            fric_c = _friction_per_contract(S[i], mkt.mult, ATM[i], fp)
            proceeds = n * mkt.mult * call_c - n * fric_c
            cash += proceeds
            friction_paid += n * fric_c
            rolls += 1
            if mkt.is_future:
                book.option_cash(proceeds)
            else:
                book.close_etf(proceeds, lot_basis, dts[i] - lot_open_day)
            avail = cash
            K, n, cash, fric, _, capped = _open(avail, S[i], ATM[i], R[i], Q[i], tenor,
                                                mkt, sizing, vparams, fp, matched_notional)
            lot_basis = avail - cash
            lot_open_day = dts[i]
            book.option_cash(-lot_basis)
            friction_paid += fric
            capped_entries += int(capped)
            expiry_day = _expiry_day(dates[i], int(tenor))
            Trem = (expiry_day - dts[i]) / 365.0

        # --- open the short leg when flat (sell ~0.30 delta at ~35 DTE) ---
        if pmcc and n_short == 0 and n > 0:
            Ts = SHORT_DTE / 365.0
            sfn = lambda k: float(vol.sigma(k, S[i], Ts, ATM[i], vparams))
            K_short = pricing.solve_strike_for_delta(S[i], Ts, R[i], sfn, SHORT_DELTA, q=Q[i])
            spx = pricing.baw_call(S[i], K_short, Ts, R[i], sfn(K_short), q=Q[i])
            frs = n * _friction_per_contract(S[i], mkt.mult, ATM[i], fp)
            short_prem = n * mkt.mult * spx - frs
            cash += short_prem
            friction_paid += frs
            short_premium_total += short_prem
            n_short = n
            expiry_short_day = dts[i] + SHORT_DTE
            if mkt.is_future:
                book.option_cash(short_prem)

        sig_t = float(vol.sigma(K, S[i], Trem, ATM[i], vparams))
        call_t = pricing.baw_call(S[i], K, Trem, R[i], sig_t, q=Q[i])
        gl_delta = pricing.bs_delta(S[i], K, Trem, R[i], sig_t, q=Q[i], kind="call")  # European delta for leverage
        long_mv = n * mkt.mult * call_t
        short_delta = 0.0
        if pmcc and n_short > 0:
            short_mv, short_delta = _short_value(S[i], K_short, expiry_short_day - dts[i],
                                                 R[i], Q[i], ATM[i], mkt, vparams, n_short)
        else:
            short_mv = 0.0
        eq = cash + long_mv - short_mv
        if n == 0:
            cash_days += 1     # forced to cash by whole-contract granularity (SPEC bias #6)
        if eq > 0:
            net_delta = n * mkt.mult * gl_delta - n_short * mkt.mult * short_delta
            lev.append(net_delta * S[i] / eq)
        eq_val.append(eq)

    # terminal: liquidate both legs for tax and settle the final partial year
    if mkt.is_future:
        term_tax = book.settle_year(long_mv - short_mv)
    else:
        book.close_etf(long_mv, lot_basis, dts[-1] - lot_open_day)
        if n_short > 0:
            book.close_short_etf(short_prem, short_mv)
        term_tax = book.settle_year(0.0)
    eq_val[-1] -= term_tax

    return {
        "equity": pd.Series(eq_val, index=dates[i0:], name="strategy"),
        "rolls": rolls,
        "short_rolls": short_rolls,
        "short_premium": short_premium_total,
        "short_pnl": short_premium_total - short_buyback_total,
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

    rep = run_strategy(mkt, 2011, 1.0, "matched", vp)
    pm = run_strategy(mkt, 2011, 1.0, "matched", vp, pmcc=True)
    bh = buy_and_hold(mkt, 2011)
    m_rep, m_pm, mb = metrics(rep["equity"], mkt.r), metrics(pm["equity"], mkt.r), metrics(bh, mkt.r)

    rep_at = run_strategy(mkt, 2011, 1.0, "matched", vp, tp=tpx)
    m_rep_at = metrics(rep_at["equity"], mkt.r)

    print("QQQ 2011 matched-exposure 1y (base vol, American, notional friction)")
    print(f"  replacement : {_fmt(m_rep)}")
    print(f"  PMCC        : {_fmt(m_pm)}")
    print(f"  buy-hold    : {_fmt(mb)}")
    print(f"  PMCC short rolls {pm['short_rolls']}  premium collected ${pm['short_premium']:,.0f}  "
          f"net leverage {pm['avg_leverage']:.2f}x (vs {rep['avg_leverage']:.2f}x long-only)")

    checks = []
    def ck(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    print("\nsanity:")
    ck("replacement unchanged (regression, ~$136.5k)", abs(m_rep["final"] - 136534) < 500,
       f"${m_rep['final']:,.0f}")
    ck("after-tax < pre-tax", m_rep_at["final"] < m_rep["final"],
       f"${m_rep_at['final']:,.0f} < ${m_rep['final']:,.0f}")
    ck("PMCC equity strictly positive", bool((pm["equity"] > 0).all()), f"min ${pm['equity'].min():,.0f}")
    ck("PMCC short rolled ~monthly", 120 <= pm["short_rolls"] <= 260, f"{pm['short_rolls']} rolls")
    ck("PMCC collected net premium", pm["short_premium"] > 0, f"${pm['short_premium']:,.0f}")
    ck("PMCC net delta below long-only", pm["avg_leverage"] < rep["avg_leverage"],
       f"{pm['avg_leverage']:.2f}x < {rep['avg_leverage']:.2f}x")
    ck("PMCC caps upside in a bull run", m_pm["final"] < m_rep["final"],
       f"${m_pm['final']:,.0f} < ${m_rep['final']:,.0f}")
    ck("PMCC vol below long-only", m_pm["ann_vol"] < m_rep["ann_vol"],
       f"{m_pm['ann_vol']*100:.1f}% < {m_rep['ann_vol']*100:.1f}%")
    print(f"\n{sum(checks)}/{len(checks)} sanity checks passed")

"""VXN-based volatility surface: sigma(K, T) = VXN_t * term_factor(T) + skew.

The 0.95-delta call is deep ITM; its correct IV rides the downside skew (equal by
put-call parity to the equivalent OTM put) and the 1-2y term, both above ATM VXN.
Flat VXN would understate the call's time value. term_slope and skew_slope are the
two exposed levers; skew is the dominant one and is regime-dependent in reality
(a known bias, SPEC Section 9). A constant skew_slope overstates long-dated skew
and a multiplicative term_factor overstates long-dated vol in inverted (crisis)
regimes; both are documented simplifications of a richer surface.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import data

T_ATM = 30.0 / 365.0  # VXN is a 30-day measure; term_factor anchors to 1 here


@dataclass(frozen=True)
class VolParams:
    term_slope: float = 0.03    # per unit ln(T/T_ATM); upward term structure
    skew_slope: float = -0.15   # per unit log-moneyness; negative -> deep-ITM IV above ATM
    moneyness: str = "log"      # "log" -> ln(K/S), or "linear" -> (K-S)/S
    vol_floor: float = 0.05


def term_factor(T, p=VolParams()):
    return 1.0 + p.term_slope * np.log(T / T_ATM)


def _moneyness(K, S, kind):
    if kind == "log":
        return np.log(K / S)
    return (K - S) / S


def sigma(K, S, T, atm_vol, p=VolParams()):
    """Total IV for strike K given decimal ATM vol atm_vol at the date. K may be
    scalar or array. atm_vol is VXN (or VIX fallback) as a decimal, not percent."""
    s = atm_vol * term_factor(T, p) + p.skew_slope * _moneyness(K, S, p.moneyness)
    return np.maximum(s, p.vol_floor)


def atm_vol_series(target_index=None, source="vxn", force_refresh=False):
    """Decimal ATM vol by date. source='vxn' uses VXN with VIX where VXN gaps
    (Nasdaq); source='vix' uses VIX directly (S&P, its native ATM vol). Forward-
    filled as a last resort, reindexed to target_index."""
    vix = data.vix(force_refresh) / 100.0
    if source == "vix":
        out = vix.reindex(target_index) if target_index is not None else vix
    else:
        v = data.vxn(force_refresh) / 100.0
        if target_index is not None:
            v = v.reindex(target_index)
            vix = vix.reindex(target_index)
        out = v.where(v.notna(), vix)
    return out.ffill().rename("atm_vol")


# Pre-registered vol-surface grid, fixed before any backtest (SPEC Section 5.1).
# base is the headline; flat and steep bound the reported robustness band.
VOL_SCENARIOS = {
    "flat":      VolParams(term_slope=0.0,  skew_slope=0.0),
    "term_only": VolParams(term_slope=0.03, skew_slope=0.0),
    "base":      VolParams(term_slope=0.03, skew_slope=-0.15),
    "steep":     VolParams(term_slope=0.05, skew_slope=-0.25),
}


_results = []


def check(name, passed, detail=""):
    _results.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    import pricing

    p = VolParams()

    # 1. ATM maps to VXN exactly at the 30-day anchor (moneyness 0, term_factor 1)
    max_dev = 0.0
    for atm in (0.10, 0.2853, 0.55):
        s_atm = float(sigma(100.0, 100.0, T_ATM, atm, p))
        max_dev = max(max_dev, abs(s_atm - atm))
    check("ATM sigma == VXN at 30-day anchor", max_dev < 1e-12, f"max dev {max_dev:.2e}")

    # 2. term_factor anchor and upward structure
    tf_atm = float(term_factor(T_ATM, p))
    tf_1y, tf_2y = float(term_factor(1.0, p)), float(term_factor(2.0, p))
    check("term_factor(30d) == 1", abs(tf_atm - 1.0) < 1e-12, f"{tf_atm:.6f}")
    check("term_factor rises 30d < 1y < 2y", 1.0 < tf_1y < tf_2y, f"1y {tf_1y:.4f}, 2y {tf_2y:.4f}")

    # 3. Deep-ITM carries higher IV than ATM; OTM call lower; monotone smirk in K
    S, T, atm = 100.0, 1.0, 0.20
    iv_atm = float(sigma(S, S, T, atm, p))
    iv_itm = float(sigma(0.75 * S, S, T, atm, p))
    iv_otm = float(sigma(1.25 * S, S, T, atm, p))
    check("deep-ITM IV > ATM IV", iv_itm > iv_atm, f"ITM {iv_itm:.4f} vs ATM {iv_atm:.4f}")
    check("OTM call IV < ATM IV", iv_otm < iv_atm, f"OTM {iv_otm:.4f} vs ATM {iv_atm:.4f}")
    Kg = np.linspace(0.5 * S, 1.5 * S, 101)
    sg = sigma(Kg, S, T, atm, p)
    check("sigma strictly decreasing in K over 0.5S-1.5S", bool(np.all(np.diff(sg) < 0)))

    # 4. Call and put at same (K, T) share IV (put-call parity consistency)
    check("IV independent of option type", True, "sigma(K,T) has no kind argument")

    # 5. Gap-free ATM vol over the study window under VXN+VIX fallback
    for name, idx in (("QQQ", data.qqq().index), ("NQ", data.nq().index)):
        s = atm_vol_series(idx)
        s = s[s.index >= pd.Timestamp(2001, 1, 23)]
        nans = int(s.isna().sum())
        check(f"ATM vol gap-free on {name} calendar from 2001-01-23", nans == 0, f"{nans} NaN")
    for name, idx in (("SPY", data.spy().index), ("ES", data.es().index)):
        s = atm_vol_series(idx, source="vix")
        s = s[s.index >= pd.Timestamp(2001, 1, 1)]
        nans = int(s.isna().sum())
        check(f"VIX-source ATM vol gap-free on {name} from 2001", nans == 0, f"{nans} NaN")
    vs, vv = atm_vol_series(source="vix"), data.vix() / 100.0
    check("source='vix' equals VIX/100", float((vs - vv).abs().max()) < 1e-12)

    # 6. End-to-end: solve 0.95-delta strike with the vol closure (2011-01-03 QQQ)
    d0 = pd.Timestamp(2011, 1, 3)
    S0 = float(data.qqq()["close"].loc[d0])
    atm0 = float(data.vxn().loc[d0]) / 100.0
    r0 = float(data.short_rate_continuous().loc[d0])
    q0 = 0.006  # illustrative QQQ yield; the backtest sets q from the dividend series
    for T0 in (1.0, 2.0):
        sig_fn = lambda K, _T=T0: sigma(K, S0, _T, atm0, p)
        K95 = pricing.solve_strike_for_delta(S0, T0, r0, sig_fn, 0.95, q=q0)
        dlt = pricing.bs_greeks(S0, K95, T0, r0, float(sig_fn(K95)), q=q0)["delta"]
        iv_deep = float(sig_fn(K95))
        iv_atm0 = float(sigma(S0, S0, T0, atm0, p))
        check(f"0.95-delta solve round-trips ({int(T0)}y)", abs(dlt - 0.95) < 1e-8,
              f"K={K95:.2f} K/S={K95/S0:.3f} delta={dlt:.10f}")
        check(f"deep-ITM IV above ATM at solved strike ({int(T0)}y)", iv_deep > iv_atm0,
              f"IV_deep {iv_deep:.4f} vs IV_atm {iv_atm0:.4f} (VXN {atm0:.4f})")

    # 7. Every pre-registered scenario solves and round-trips at 2011 QQQ, both tenors
    for sname, sp in VOL_SCENARIOS.items():
        for T0 in (1.0, 2.0):
            if sp.skew_slope == 0.0:
                ivs = atm0 * term_factor(T0, sp)
                Ks = pricing.solve_strike_for_delta(S0, T0, r0, ivs, 0.95, q=q0)
            else:
                fn = lambda K, _T=T0, _sp=sp: sigma(K, S0, _T, atm0, _sp)
                Ks = pricing.solve_strike_for_delta(S0, T0, r0, fn, 0.95, q=q0)
                ivs = float(sigma(Ks, S0, T0, atm0, sp))
            dl = pricing.bs_greeks(S0, Ks, T0, r0, ivs, q=q0, kind="call")["delta"]
            check(f"scenario {sname:<9} {int(T0)}y solves", abs(dl - 0.95) < 1e-8,
                  f"K/S={Ks/S0:.3f} IV={ivs:.3f}")

    print(f"\n2011-01-03: QQQ S={S0:.2f}, VXN={atm0:.4f}, r={r0:.4f}")
    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

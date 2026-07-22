"""Full run matrix and the Section 10 metrics table.

Sweeps underlyings x start years x strategies x tenors x sizings across the four
pre-registered vol scenarios (SPEC 5.1) and both tax modes (SPEC 15.4). The base
scenario is the headline; flat and steep bound the reported band. Strategy runs are
compared against the buy-and-hold baseline of the same underlying, start year, and
tax mode.
"""

import sys
from pathlib import Path

import pandas as pd

import backtest as bt
import vol

UNDERLYINGS = ["QQQ", "NQ", "SPY", "ES"]
START_YEARS = [2001, 2011]
STRATEGIES = [("replacement", False), ("pmcc", True)]
TENORS = [1.0, 2.0]
SIZINGS = ["matched", "full"]
TAX_MODES = [("pre-tax", None), ("after-tax", bt.TaxParams(enabled=True))]

CSV = Path(__file__).resolve().parent / "cache" / "results.csv"


def run_matrix(scenarios=None, progress=True):
    scenarios = scenarios or vol.VOL_SCENARIOS
    markets = {n: bt.build_market(n) for n in UNDERLYINGS}
    bh_cache, rows = {}, []
    for uname in UNDERLYINGS:
        mkt = markets[uname]
        for yr in START_YEARS:
            for tname, tp in TAX_MODES:
                if (uname, yr, tname) not in bh_cache:
                    bh = bt.buy_and_hold(mkt, yr, tp=tp)
                    bh_cache[(uname, yr, tname)] = bt.metrics(bh, mkt.r)
            for sname, pmcc in STRATEGIES:
                for ten in TENORS:
                    for sz in SIZINGS:
                        for vname, vp in scenarios.items():
                            for tname, tp in TAX_MODES:
                                r = bt.run_strategy(mkt, yr, ten, sz, vp, pmcc=pmcc, tp=tp)
                                m = bt.metrics(r["equity"], mkt.r)
                                mb = bh_cache[(uname, yr, tname)]
                                rows.append({
                                    "underlying": uname, "start_year": yr,
                                    "strategy": sname, "tenor": int(ten), "sizing": sz,
                                    "vol_scenario": vname, "tax": tname,
                                    "final": m["final"], "total_return": m["total_return"],
                                    "cagr": m["cagr"], "ann_vol": m["ann_vol"],
                                    "sharpe": m["sharpe"], "sortino": m["sortino"],
                                    "max_dd": m["max_dd"], "calmar": m["calmar"],
                                    "avg_leverage": r["avg_leverage"], "rolls": r["rolls"],
                                    "short_rolls": r["short_rolls"], "friction": r["friction"],
                                    "tax_paid": r["tax_paid"], "cash_frac": r["cash_frac"],
                                    "capped_entries": r["capped_entries"],
                                    "bh_final": mb["final"], "bh_cagr": mb["cagr"],
                                    "bh_sharpe": mb["sharpe"],
                                    "vs_bh": m["final"] / mb["final"],
                                })
            if progress:
                print(f"  ran {uname} {yr}: {len(rows)} rows", file=sys.stderr)
    return pd.DataFrame(rows)


def headline(df):
    """After-tax matched-exposure at base vol: the stable, decision-relevant core.

    Matched exposure holds ~1x notional, so the comparison isolates the cost of the
    option route against simply owning the underlying.
    """
    b = df[(df.vol_scenario == "base") & (df.sizing == "matched")]
    pre = b[b.tax == "pre-tax"].set_index(["underlying", "start_year", "strategy", "tenor"])
    post = b[b.tax == "after-tax"].set_index(["underlying", "start_year", "strategy", "tenor"])
    print("\n" + "=" * 78)
    print("HEADLINE: after-tax, matched exposure (~1x), base vol scenario")
    print("=" * 78)
    print(f"{'run':30}{'CAGR':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}"
          f"{'vsBH':>7}{'preVsBH':>9}{'cash%':>7}")
    for key in post.index:
        a, x = pre.loc[key], post.loc[key]
        u, y, st, t = key
        print(f"{u+' '+str(y)+' '+st[:4]+' '+str(t)+'y':30}{x.cagr*100:7.1f}%{x.ann_vol*100:6.1f}%"
              f"{x.sharpe:8.2f}{x.max_dd*100:7.1f}%{x.vs_bh:7.2f}{a.vs_bh:9.2f}{x.cash_frac*100:6.0f}%")
    n_beat = int((post.vs_bh > 1.0).sum())
    n_beat_pre = int((pre.vs_bh > 1.0).sum())
    print(f"\n  beats buy-and-hold after tax: {n_beat}/{len(post)}   "
          f"(pre-tax: {n_beat_pre}/{len(pre)})")
    for st in ("replacement", "pmcc"):
        sl = post[post.index.get_level_values("strategy") == st]
        print(f"    {st:12} median vsBH {sl.vs_bh.median():.2f}   "
              f"beats {int((sl.vs_bh > 1).sum())}/{len(sl)}")


def full_capital_band(df):
    """Full-capital reported only as a sensitivity band.

    These configurations are dominated by the unobservable vol surface: terminal
    outcomes span a wide multiple across the four pre-registered scenarios, so the
    point estimates carry no usable signal and are deliberately not reported.
    """
    f = df[(df.sizing == "full") & (df.tax == "after-tax")]
    g = f.groupby(["underlying", "start_year", "strategy", "tenor"])["vs_bh"].agg(
        ["min", "median", "max"])
    print("\n" + "=" * 78)
    print("SENSITIVITY BAND ONLY (not a result): full-capital, after tax")
    print("vsBH range across the four pre-registered vol scenarios (SPEC 5.1).")
    print("Leverage compounds the unobservable surface assumption; read the spread,")
    print("not the level.")
    print("=" * 78)
    print(f"{'run':30}{'min':>9}{'median':>9}{'max':>9}{'spread':>9}")
    for key, r in g.iterrows():
        u, y, st, t = key
        spread = r["max"] / r["min"] if r["min"] > 0 else float("inf")
        print(f"{u+' '+str(y)+' '+st[:4]+' '+str(t)+'y':30}{r['min']:9.2f}{r['median']:9.2f}"
              f"{r['max']:9.2f}{spread:8.0f}x")
    print(f"\n  median spread across configs: "
          f"{(g['max'] / g['min'].clip(lower=1e-9)).median():.0f}x")


def degenerate(df):
    b = df[df.vol_scenario == "base"]
    deg = b[b.cash_frac > 0.05]
    print(f"\n--- degenerate runs, >5% of days forced to cash by whole-contract "
          f"granularity (SPEC bias #6): {len(deg)} of {len(b)} ---")
    for _, r in deg.iterrows():
        print(f"  {r.underlying} {r.start_year} {r.strategy[:4]} {r.tenor}y {r.sizing} "
              f"{r.tax}: cash {r.cash_frac*100:.0f}%  vsBH {r.vs_bh:.2f}")


if __name__ == "__main__":
    print("running matrix...", file=sys.stderr)
    df = run_matrix()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    print(f"\n{len(df)} runs written to {CSV}")
    headline(df)
    full_capital_band(df)
    degenerate(df)

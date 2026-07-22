"""Forward Monte Carlo: GBM paths through the same verified strategy engine.

Two overlapping historical paths cannot answer a distributional question, so the
leverage question is answered here: the fan of terminal outcomes, the probability
the strategy beats buy-and-hold, and the ruin probability of full-capital sizing.

Each GBM path becomes a synthetic Market handed to backtest.run_strategy, so the
strategy logic is identical to the historical study. The implied vol used to price
options is a separate input from the realized vol that generates the path, so the
variance risk premium can be set explicitly (SPEC bias #7) rather than assumed.

Capital defaults to $100k here so whole-contract granularity (SPEC bias #6) does
not contaminate the leverage question; that constraint is studied separately in the
historical layer at $10k.
"""

import numpy as np
import pandas as pd

import backtest as bt
import vol

TRADING_DAYS = 252


def gbm_paths(S0, mu, sigma, years, n_paths, seed=0):
    """Daily GBM: S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)."""
    rng = np.random.default_rng(seed)
    n = int(round(years * TRADING_DAYS))
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_paths, n))
    incr = (mu - 0.5 * sigma * sigma) * dt + sigma * math_sqrt(dt) * z
    logs = np.log(S0) + np.cumsum(incr, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(logs)])


def math_sqrt(x):
    return float(np.sqrt(x))


def _market(path, dates, implied_vol, r, q, mult, is_future):
    px = pd.Series(path, index=dates)
    iv = pd.Series(np.full(len(dates), implied_vol), index=dates)
    rr = pd.Series(np.full(len(dates), r), index=dates)
    qq = pd.Series(np.full(len(dates), r if is_future else q), index=dates)
    t = np.arange(len(dates)) / TRADING_DAYS
    adj = pd.Series(path * np.exp((0.0 if is_future else q) * t), index=dates)
    return bt.Market("MC", px, iv, rr, qq, mult, is_future, dates[0], adj_close=adj)


def run(n_paths=300, years=15.0, S0=100.0, mu=0.08, sigma=0.20, implied_vol=None,
        r=0.04, q=0.006, mult=100.0, tenor=1.0, pmcc=False, is_future=False,
        start_capital=100_000.0, seed=0, vparams=None, sizings=("matched", "full")):
    vparams = vparams or vol.VOL_SCENARIOS["base"]
    iv = sigma if implied_vol is None else implied_vol
    paths = gbm_paths(S0, mu, sigma, years, n_paths, seed)
    dates = pd.bdate_range("2000-01-03", periods=paths.shape[1])
    rows = {sz: [] for sz in sizings}
    for k in range(n_paths):
        mkt = _market(paths[k], dates, iv, r, q, mult, is_future)
        bh = float(bt.buy_and_hold(mkt, 2000, start_capital=start_capital).iloc[-1])
        for sz in sizings:
            res = bt.run_strategy(mkt, 2000, tenor, sz, vparams, pmcc=pmcc,
                                  start_capital=start_capital)
            eq = res["equity"]
            rows[sz].append({"final": float(eq.iloc[-1]), "bh": bh,
                             "min_eq": float(eq.min()), "lev": res["avg_leverage"],
                             "cash_frac": res["cash_frac"]})
    return {sz: pd.DataFrame(v) for sz, v in rows.items()}


def summarize(df, start_capital=100_000.0):
    f = df["final"].values
    ratio = df["final"].values / df["bh"].values
    return {
        "p5": float(np.percentile(f, 5)), "p25": float(np.percentile(f, 25)),
        "median": float(np.median(f)), "p75": float(np.percentile(f, 75)),
        "p95": float(np.percentile(f, 95)),
        "median_bh": float(np.median(df["bh"].values)),
        "beat_bh": float((ratio > 1.0).mean()),
        "median_vs_bh": float(np.median(ratio)),
        "loss_50": float((f < 0.5 * start_capital).mean()),
        "ruin_90": float((f < 0.1 * start_capital).mean()),
        "avg_lev": float(df["lev"].mean()),
        "cash_frac": float(df["cash_frac"].mean()),
        # cash among surviving paths isolates granularity from post-ruin inability
        # to afford a contract
        "cash_frac_alive": float(df.loc[df["final"] >= 0.5 * start_capital, "cash_frac"].mean()),
    }


def _fmt(name, s, cap):
    return (f"  {name:9} median ${s['median']:>11,.0f} ({s['median']/cap:5.2f}x)  "
            f"p5 ${s['p5']:>10,.0f}  p95 ${s['p95']:>12,.0f}  "
            f"beatBH {s['beat_bh']*100:5.1f}%  medVsBH {s['median_vs_bh']:5.2f}  "
            f"P(<50%) {s['loss_50']*100:5.1f}%  P(ruin) {s['ruin_90']*100:5.1f}%  "
            f"lev {s['avg_lev']:.2f}")


if __name__ == "__main__":
    CAP, N, YRS = 100_000.0, 300, 15.0
    print(f"Monte Carlo: {N} GBM paths, {YRS:.0f}y, mu 8%, realized vol 20%, r 4%, "
          f"q 0.6%, ${CAP:,.0f} start, 1y tenor, base vol surface")

    print("\n[A] implied vol = realized vol (no variance risk premium)")
    res = run(n_paths=N, years=YRS, start_capital=CAP)
    sm, sf = summarize(res["matched"], CAP), summarize(res["full"], CAP)
    print(f"  buy-hold  median ${sm['median_bh']:>11,.0f} ({sm['median_bh']/CAP:5.2f}x)")
    print(_fmt("matched", sm, CAP))
    print(_fmt("full", sf, CAP))

    print("\n[B] implied vol 22% vs realized 20% (a 2-point variance risk premium)")
    res2 = run(n_paths=N, years=YRS, implied_vol=0.22, start_capital=CAP)
    sm2, sf2 = summarize(res2["matched"], CAP), summarize(res2["full"], CAP)
    print(_fmt("matched", sm2, CAP))
    print(_fmt("full", sf2, CAP))

    print("\n[C] PMCC, testing whether its historical full-capital edge survives")
    res3 = run(n_paths=N, years=YRS, pmcc=True, start_capital=CAP, sizings=("full",))
    sp = summarize(res3["full"], CAP)
    print(_fmt("pmcc full", sp, CAP))
    print(f"  replacement full for comparison: median {sf['median']/CAP:.2f}x  "
          f"beatBH {sf['beat_bh']*100:.1f}%  P(ruin) {sf['ruin_90']*100:.1f}%")

    print("\nsanity:")
    ck = []
    def c(name, ok, detail=""):
        ck.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    exp_bh = CAP * np.exp((0.08 - 0.5 * 0.04 + 0.006) * YRS)
    c("B&H median matches GBM expectation", abs(sm["median_bh"] / exp_bh - 1) < 0.10,
      f"${sm['median_bh']:,.0f} vs ${exp_bh:,.0f}")
    c("matched tracks near 1x leverage", 0.8 < sm["avg_lev"] < 1.25, f"{sm['avg_lev']:.2f}x")
    c("full-capital levers above matched", sf["avg_lev"] > sm["avg_lev"] + 0.5,
      f"{sf['avg_lev']:.2f}x vs {sm['avg_lev']:.2f}x")
    c("full-capital dispersion exceeds matched",
      (sf["p95"] / sf["p5"]) > (sm["p95"] / sm["p5"]),
      f"{sf['p95']/sf['p5']:.0f}x vs {sm['p95']/sm['p5']:.0f}x")
    c("full-capital carries ruin risk, matched does not",
      sf["ruin_90"] > sm["ruin_90"], f"{sf['ruin_90']*100:.1f}% vs {sm['ruin_90']*100:.1f}%")
    c("granularity not contaminating surviving paths",
      sm["cash_frac_alive"] < 0.01 and sf["cash_frac_alive"] < 0.01,
      f"alive-path cash {sm['cash_frac_alive']*100:.2f}% / {sf['cash_frac_alive']*100:.2f}% "
      f"(all-path {sf['cash_frac']*100:.2f}% is post-ruin)")
    print(f"\n{sum(ck)}/{len(ck)} sanity checks passed")

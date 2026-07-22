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

import math

import numpy as np
import pandas as pd

import backtest as bt
import vol

TRADING_DAYS = 252


def simulate_paths(S0, mu, sigma, years, n_paths, seed=0,
                   jump_lambda=0.0, jump_mean=-0.01, jump_vol=0.08,
                   match_total_vol=True):
    """Daily paths under Merton jump-diffusion.

    Leaving jump_lambda at zero reduces the process to pure geometric Brownian
    motion and draws no extra randomness, so seeded paths stay bit-identical to
    the diffusion-only case and earlier results remain valid.

    Jump counts per step come from Poisson(lambda*dt). Because the sum of n
    independent normal jump sizes is itself normal with mean n*jump_mean and
    variance n*jump_vol^2, the whole jump contribution draws in one vectorized
    step. The -lambda*k*dt term compensates the drift so expected total return
    still equals mu.

    match_total_vol shrinks the diffusion component so total annualized variance
    still equals sigma^2, which separates the effect of jump shape from the
    effect of simply adding variance.
    """
    rng = np.random.default_rng(seed)
    n = int(round(years * TRADING_DAYS))
    dt = 1.0 / TRADING_DAYS
    sd = sigma
    if jump_lambda > 0.0 and match_total_vol:
        jump_var = jump_lambda * (jump_mean * jump_mean + jump_vol * jump_vol)
        if jump_var >= sigma * sigma:
            raise ValueError("jump variance exceeds the total volatility target")
        sd = math.sqrt(sigma * sigma - jump_var)
    z = rng.standard_normal((n_paths, n))
    incr = (mu - 0.5 * sd * sd) * dt + sd * math.sqrt(dt) * z
    if jump_lambda > 0.0:
        k = math.exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0
        nj = rng.poisson(jump_lambda * dt, size=(n_paths, n))
        z2 = rng.standard_normal((n_paths, n))
        incr += nj * jump_mean + np.sqrt(nj) * jump_vol * z2 - jump_lambda * k * dt
    logs = np.log(S0) + np.cumsum(incr, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(logs)])


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
        start_capital=100_000.0, seed=0, vparams=None, sizings=("matched", "full"),
        jump_lambda=0.0, jump_mean=-0.01, jump_vol=0.08):
    vparams = vparams or vol.VOL_SCENARIOS["base"]
    iv = sigma if implied_vol is None else implied_vol
    paths = simulate_paths(S0, mu, sigma, years, n_paths, seed,
                           jump_lambda, jump_mean, jump_vol)
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


def run_paths(n_paths=200, years=15.0, S0=100.0, mu=0.08, sigma=0.20, implied_vol=None,
              r=0.04, q=0.006, mult=100.0, tenor=1.0, pmcc=False, is_future=False,
              sizing="full", start_capital=100_000.0, seed=0, vparams=None,
              jump_lambda=0.0, jump_mean=-0.01, jump_vol=0.08):
    """Single sizing, also returning the equity-curve matrix for a fan chart."""
    vparams = vparams or vol.VOL_SCENARIOS["base"]
    iv = sigma if implied_vol is None else implied_vol
    paths = simulate_paths(S0, mu, sigma, years, n_paths, seed,
                           jump_lambda, jump_mean, jump_vol)
    dates = pd.bdate_range("2000-01-03", periods=paths.shape[1])
    curves, bh_curves, rows = [], [], []
    for k in range(n_paths):
        mkt = _market(paths[k], dates, iv, r, q, mult, is_future)
        bh = bt.buy_and_hold(mkt, 2000, start_capital=start_capital)
        res = bt.run_strategy(mkt, 2000, tenor, sizing, vparams, pmcc=pmcc,
                              start_capital=start_capital)
        eq = res["equity"]
        curves.append(eq.values)
        bh_curves.append(bh.values)
        rows.append({"final": float(eq.iloc[-1]), "bh": float(bh.iloc[-1]),
                     "min_eq": float(eq.min()), "lev": res["avg_leverage"],
                     "cash_frac": res["cash_frac"]})
    return {"df": pd.DataFrame(rows), "curves": np.array(curves),
            "bh_curves": np.array(bh_curves), "dates": dates}


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

    print("\n[D] single-stock parameters, diffusion only against jumps at equal total vol")
    print("    vol 35%, no dividend, 4 jumps/yr averaging -1% with 8% jump vol")
    for label, jl in (("no jumps ", 0.0), ("jumps    ", 4.0)):
        o = run(n_paths=N, years=YRS, sigma=0.35, q=0.0, start_capital=CAP,
                jump_lambda=jl)
        for sz in ("matched", "full"):
            st = summarize(o[sz], CAP)
            print(f"  {label} {sz:8} medVsBH {st['median_vs_bh']:5.2f}  "
                  f"beatBH {st['beat_bh']*100:5.1f}%  P(<50%) {st['loss_50']*100:5.1f}%  "
                  f"P(ruin) {st['ruin_90']*100:5.1f}%")

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
    base_p = simulate_paths(100.0, 0.08, 0.20, 5.0, 400, 7)
    jump_p = simulate_paths(100.0, 0.08, 0.20, 5.0, 400, 7, jump_lambda=4.0)
    c("jump_lambda=0 leaves paths bit-identical to diffusion only",
      float(np.abs(base_p - simulate_paths(100.0, 0.08, 0.20, 5.0, 400, 7,
                                           jump_lambda=0.0)).max()) == 0.0)
    lr_b = np.diff(np.log(base_p), axis=1).ravel()
    lr_j = np.diff(np.log(jump_p), axis=1).ravel()
    v_b, v_j = lr_b.std() * np.sqrt(TRADING_DAYS), lr_j.std() * np.sqrt(TRADING_DAYS)
    c("jumps preserve total volatility", abs(v_j - v_b) < 0.02, f"{v_b:.3f} vs {v_j:.3f}")
    k_b = float(((lr_b - lr_b.mean())**4).mean() / lr_b.var()**2 - 3.0)
    k_j = float(((lr_j - lr_j.mean())**4).mean() / lr_j.var()**2 - 3.0)
    c("jumps produce fat tails", k_j > k_b + 1.0, f"excess kurtosis {k_b:.2f} vs {k_j:.2f}")
    c("granularity not contaminating surviving paths",
      sm["cash_frac_alive"] < 0.01 and sf["cash_frac_alive"] < 0.01,
      f"alive-path cash {sm['cash_frac_alive']*100:.2f}% / {sf['cash_frac_alive']*100:.2f}% "
      f"(all-path {sf['cash_frac']*100:.2f}% is post-ruin)")
    print(f"\n{sum(ck)}/{len(ck)} sanity checks passed")

"""Static figures for the README.

Regenerates every figure from the engine and the run matrix. Writes PNGs to
figures/. Requires cache/results.csv (produced by results.py).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import montecarlo as mc
import vol

FIG = Path(__file__).resolve().parent / "figures"
CSV = Path(__file__).resolve().parent / "cache" / "results.csv"
C_STRAT, C_BH, C_ALT = "#1f4e79", "#c0504d", "#7f7f7f"


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_equity():
    """Tax is what decides the matched-exposure comparison."""
    mkt = bt.build_market("QQQ")
    vp = vol.VOL_SCENARIOS["base"]
    tpx = bt.TaxParams(enabled=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tp, label in ((axes[0], None, "Tax-advantaged (pre-tax)"),
                          (axes[1], tpx, "Taxable (after-tax, 35/20)")):
        res = bt.run_strategy(mkt, 2011, 1.0, "matched", vp, tp=tp)
        bh = bt.buy_and_hold(mkt, 2011, tp=tp)
        eq = res["equity"]
        ax.plot(eq.index, eq.values, color=C_STRAT, lw=1.6, label="0.95-delta replacement")
        ax.plot(bh.index, bh.values, color=C_BH, lw=1.6, ls="--", label="buy-and-hold")
        ax.set_yscale("log")
        ax.set_title(f"{label}\nterminal {eq.iloc[-1]/bh.iloc[-1]:.2f}x buy-and-hold",
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("equity ($, log scale)")
    fig.suptitle("QQQ from 2011, matched exposure (~1x), 1-year tenor", fontsize=11)
    _save(fig, "fig1_tax_effect.png")


def fig_montecarlo(n_paths=300, years=15.0, cap=100_000.0):
    """The leverage question, answered as a distribution."""
    out = mc.run(n_paths=n_paths, years=years, start_capital=cap)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(2.5, 7.5, 60)
    for sz, color, name in (("matched", C_STRAT, "matched (~1x)"),
                            ("full", C_BH, "full-capital (~2.6x)")):
        df = out[sz]
        s = mc.summarize(df, cap)
        ax.hist(np.log10(np.clip(df["final"], 10, None)), bins=bins, alpha=0.55,
                color=color, label=f"{name}: median {s['median']/cap:.2f}x start, "
                                   f"ruin {s['ruin_90']*100:.0f}%")
        ax.axvline(np.log10(s["median"]), color=color, lw=2, ls="-")
    ax.axvline(np.log10(cap), color="k", lw=1.4, ls=":", label="starting capital")
    ax.axvline(np.log10(np.median(out["matched"]["bh"])), color=C_ALT, lw=1.8, ls="--",
               label="median buy-and-hold")
    ticks = [3, 4, 5, 6, 7]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${10**t:,.0f}" for t in ticks])
    ax.set_xlabel(f"terminal equity after {years:.0f} years (log scale)")
    ax.set_ylabel("paths")
    ax.set_title(f"{n_paths} forward GBM paths: leverage widens the distribution "
                 f"and adds ruin", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig2_montecarlo.png")


def fig_beat_rates(df):
    """How often each configuration beats its own buy-and-hold."""
    base = df[df.vol_scenario == "base"]
    groups = [("replacement", "matched"), ("pmcc", "matched"),
              ("replacement", "full"), ("pmcc", "full")]
    labels = ["replacement\nmatched", "PMCC\nmatched", "replacement\nfull", "PMCC\nfull"]
    pre, post = [], []
    for st, sz in groups:
        for tax, acc in (("pre-tax", pre), ("after-tax", post)):
            sl = base[(base.strategy == st) & (base.sizing == sz) & (base.tax == tax)]
            acc.append(100.0 * (sl.vs_bh > 1.0).mean())
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.19, pre, 0.38, color=C_ALT, label="pre-tax (tax-advantaged)")
    ax.bar(x + 0.19, post, 0.38, color=C_STRAT, label="after-tax (taxable, 35/20)")
    for i, (a, b) in enumerate(zip(pre, post)):
        ax.text(i - 0.19, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.19, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8)
    ax.axhline(50, color="k", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of configurations beating buy-and-hold")
    ax.set_ylim(0, 108)
    ax.set_title("Taxes decide the matched-exposure comparison", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig3_beat_rates.png")


def fig_sensitivity(df):
    """Why full-capital results are reported only as a band."""
    groups = [("replacement", "matched"), ("pmcc", "matched"),
              ("replacement", "full"), ("pmcc", "full")]
    labels = ["replacement\nmatched", "PMCC\nmatched", "replacement\nfull", "PMCC\nfull"]
    data = [df[(df.strategy == st) & (df.sizing == sz) &
               (df.tax == "pre-tax")].vs_bh.values for st, sz in groups]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, widths=0.55, patch_artist=True,
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    for patch, sz in zip(bp["boxes"], [g[1] for g in groups]):
        patch.set_facecolor(C_STRAT if sz == "matched" else C_BH)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("k")
    ax.axhline(1.0, color="k", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("terminal equity / buy-and-hold (log scale)")
    ax.set_title("Spread across the four pre-registered vol scenarios\n"
                 "Matched exposure is stable; leverage compounds the assumption",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y", which="both")
    _save(fig, "fig4_sensitivity.png")


if __name__ == "__main__":
    if not CSV.exists():
        raise SystemExit("cache/results.csv missing; run `python results.py` first")
    df = pd.read_csv(CSV)
    print("generating figures...")
    fig_equity()
    fig_beat_rates(df)
    fig_sensitivity(df)
    fig_montecarlo()
    print("done")

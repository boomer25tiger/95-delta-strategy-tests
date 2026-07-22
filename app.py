"""Streamlit app: historical and Monte Carlo simulators for the 0.95-delta study.

Every option price here is model-derived, never a traded price. The vol surface is
the largest modeling lever and is exposed in the sidebar; the four pre-registered
scenarios (SPEC 5.1) were fixed before any result existed.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtest as bt
import montecarlo as mc
import vol

st.set_page_config(page_title="0.95-delta strategy tests", layout="wide")

UNDERLYINGS = ["QQQ", "NQ", "SPY", "ES"]


@st.cache_resource(show_spinner=False)
def market(name):
    return bt.build_market(name)


@st.cache_data(show_spinner=False)
def historical(name, year, tenor, sizing, pmcc, term_slope, skew_slope,
               taxable, ordinary, ltcg, notional_pct, commission):
    mkt = market(name)
    vp = vol.VolParams(term_slope=term_slope, skew_slope=skew_slope)
    fp = bt.FrictionParams(notional_pct=notional_pct, commission=commission)
    tp = bt.TaxParams(ordinary=ordinary, ltcg=ltcg, enabled=taxable)
    res = bt.run_strategy(mkt, year, tenor, sizing, vp, pmcc=pmcc, fp=fp, tp=tp)
    bh = bt.buy_and_hold(mkt, year, tp=tp)
    return res, bh, mkt.r


@st.cache_data(show_spinner=False)
def montecarlo(n_paths, years, mu, sigma, implied_vol, r, q, tenor, sizing, pmcc,
               start_capital, term_slope, skew_slope, seed):
    vp = vol.VolParams(term_slope=term_slope, skew_slope=skew_slope)
    return mc.run_paths(n_paths=n_paths, years=years, mu=mu, sigma=sigma,
                        implied_vol=implied_vol, r=r, q=q, tenor=tenor,
                        sizing=sizing, pmcc=pmcc, start_capital=start_capital,
                        seed=seed, vparams=vp)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Modeling assumptions")
st.sidebar.caption(
    "Free historical option prices do not exist this far back, so every option is "
    "priced synthetically. The vol surface below is the single largest lever."
)

scen = st.sidebar.selectbox("Vol scenario (pre-registered, SPEC 5.1)",
                            list(vol.VOL_SCENARIOS) + ["custom"], index=2)
if scen == "custom":
    term_slope = st.sidebar.slider("term_slope", 0.0, 0.10, 0.03, 0.005)
    skew_slope = st.sidebar.slider("skew_slope", -0.40, 0.0, -0.15, 0.01)
else:
    p = vol.VOL_SCENARIOS[scen]
    term_slope, skew_slope = p.term_slope, p.skew_slope
    st.sidebar.caption(f"term_slope {term_slope:.3f}, skew_slope {skew_slope:.3f}")

st.sidebar.subheader("Taxes")
taxable = st.sidebar.radio("Account", ["Tax-advantaged (pre-tax)", "Taxable (after-tax)"],
                           index=1).startswith("Taxable")
ordinary = st.sidebar.number_input("Short-term / ordinary rate", 0.0, 0.60, 0.35, 0.01)
ltcg = st.sidebar.number_input("Long-term / qualified rate", 0.0, 0.40, 0.20, 0.01)

st.sidebar.subheader("Frictions")
notional_pct = st.sidebar.number_input("Half-spread, % of notional per side",
                                       0.0, 0.01, 0.0005, 0.0001, format="%.4f")
commission = st.sidebar.number_input("Commission $/contract/leg", 0.0, 5.0, 0.65, 0.05)

st.title("0.95-delta call strategies versus buy-and-hold")

tab_hist, tab_mc, tab_notes = st.tabs(
    ["Historical simulator", "Monte Carlo simulator", "Assumptions and biases"])

# ---------------------------------------------------------------- historical
with tab_hist:
    c = st.columns(5)
    u = c[0].selectbox("Underlying", UNDERLYINGS)
    yr = c[1].selectbox("Start year", [2001, 2011])
    strat = c[2].selectbox("Strategy", ["95-delta replacement", "PMCC"])
    ten = c[3].selectbox("Tenor", [1.0, 2.0], format_func=lambda t: f"{int(t)}y")
    sz = c[4].selectbox("Sizing", ["matched", "full"],
                        format_func=lambda s: "matched (~1x)" if s == "matched" else "full-capital")

    res, bh, rser = historical(u, yr, ten, sz, strat == "PMCC", term_slope, skew_slope,
                               taxable, ordinary, ltcg, notional_pct, commission)
    eq = res["equity"]

    idx = eq.index
    stop = st.slider("Walk the divergence forward", 0, len(idx) - 1, len(idx) - 1,
                     format="")
    upto = idx[stop]
    st.caption(f"showing through {upto.date()}")

    e, b = eq.loc[:upto], bh.loc[:upto]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e.index, y=e.values, name=strat, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=b.index, y=b.values, name="buy-and-hold",
                             line=dict(width=2, dash="dash")))
    fig.update_layout(height=430, yaxis_type="log", yaxis_title="equity ($, log)",
                      margin=dict(t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    ms, mb = bt.metrics(e, rser), bt.metrics(b, rser)
    tbl = pd.DataFrame({
        strat: [ms["final"], ms["cagr"], ms["ann_vol"], ms["sharpe"], ms["sortino"],
                ms["max_dd"], ms["calmar"]],
        "buy-and-hold": [mb["final"], mb["cagr"], mb["ann_vol"], mb["sharpe"],
                         mb["sortino"], mb["max_dd"], mb["calmar"]],
    }, index=["final $", "CAGR", "ann vol", "Sharpe", "Sortino", "max drawdown", "Calmar"])
    fmt = {"final $": "${:,.0f}", "CAGR": "{:.2%}", "ann vol": "{:.2%}",
           "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "max drawdown": "{:.1%}",
           "Calmar": "{:.2f}"}
    st.dataframe(tbl.apply(lambda row: row.map(fmt[row.name].format), axis=1),
                 use_container_width=True)

    k = st.columns(5)
    k[0].metric("terminal vs B&H", f"{ms['final']/mb['final']:.2f}x")
    k[1].metric("avg leverage", f"{res['avg_leverage']:.2f}x")
    k[2].metric("rolls", f"{res['rolls']}" + (f" + {res['short_rolls']} short"
                                              if res["short_rolls"] else ""))
    k[3].metric("friction paid", f"${res['friction']:,.0f}")
    k[4].metric("tax paid", f"${res['tax_paid']:,.0f}")

    if res["cash_frac"] > 0.05:
        st.warning(
            f"{res['cash_frac']*100:.0f}% of days were forced to cash because the account "
            f"could not afford one whole contract (SPEC bias #6). This run's comparison "
            f"reflects that constraint as much as the strategy.")
    if sz == "full":
        st.info(
            "Full-capital results are dominated by the vol-surface assumption once leverage "
            "compounds: across the four pre-registered scenarios the same configuration spans "
            "a wide multiple. Read the Monte Carlo tab for the distributional answer.")

# ---------------------------------------------------------------- monte carlo
with tab_mc:
    st.caption(
        "Two overlapping historical start years cannot answer a distributional question. "
        "These are forward GBM paths through the same strategy engine. Implied vol is "
        "separate from realized vol, so the variance risk premium is an explicit input.")
    c = st.columns(4)
    n_paths = c[0].slider("paths", 50, 600, 200, 50)
    years = c[1].slider("horizon (years)", 3, 25, 15)
    mu = c[2].slider("drift", 0.0, 0.15, 0.08, 0.01)
    sigma = c[3].slider("realized vol", 0.10, 0.45, 0.20, 0.01)
    c2 = st.columns(4)
    implied_vol = c2[0].slider("implied vol (pricing)", 0.10, 0.45, 0.20, 0.01)
    rr = c2[1].slider("risk-free rate", 0.0, 0.08, 0.04, 0.005)
    mc_sz = c2[2].selectbox("Sizing ", ["full", "matched"],
                            format_func=lambda s: "full-capital" if s == "full" else "matched (~1x)")
    mc_strat = c2[3].selectbox("Strategy ", ["95-delta replacement", "PMCC"])
    cap = st.number_input("starting capital", 10_000, 1_000_000, 100_000, 10_000,
                          help="Defaults to $100k so whole-contract granularity does not "
                               "contaminate the leverage question.")

    if st.button("Run simulation", type="primary"):
        with st.spinner(f"running {n_paths} paths..."):
            out = montecarlo(n_paths, float(years), mu, sigma, implied_vol, rr, 0.006,
                             1.0, mc_sz, mc_strat == "PMCC", float(cap),
                             term_slope, skew_slope, 0)
        df, curves, dates = out["df"], out["curves"], out["dates"]
        s = mc.summarize(df, cap)

        k = st.columns(5)
        # delta_color off: a multiple of starting capital is not a signed change, and
        # Streamlit would tint "0.94x start" green while it is a 6% loss
        k[0].metric("median terminal", f"${s['median']:,.0f}",
                    f"{s['median']/cap:.2f}x start", delta_color="off")
        k[1].metric("beats buy-and-hold", f"{s['beat_bh']*100:.1f}%")
        k[2].metric("median vs B&H", f"{s['median_vs_bh']:.2f}x")
        k[3].metric("P(lose >50%)", f"{s['loss_50']*100:.1f}%")
        k[4].metric("P(ruin, -90%)", f"{s['ruin_90']*100:.1f}%")

        pct = np.percentile(curves, [5, 25, 50, 75, 95], axis=0)
        fig = go.Figure()
        for lo, hi, name in ((0, 4, "5-95%"), (1, 3, "25-75%")):
            fig.add_trace(go.Scatter(x=dates, y=pct[hi], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dates, y=pct[lo], line=dict(width=0), fill="tonexty",
                                     name=name, fillcolor="rgba(70,130,180,0.25)"))
        fig.add_trace(go.Scatter(x=dates, y=pct[2], name="median strategy",
                                 line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=dates, y=np.median(out["bh_curves"], axis=0),
                                 name="median buy-and-hold", line=dict(width=2, dash="dash")))
        fig.update_layout(height=430, yaxis_type="log", yaxis_title="equity ($, log)",
                          margin=dict(t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        h = go.Figure()
        h.add_trace(go.Histogram(x=np.log10(np.clip(df["final"], 1, None)), nbinsx=45,
                                 name="strategy"))
        h.add_vline(x=np.log10(cap), line_dash="dot",
                    annotation_text="starting capital")
        h.update_layout(height=300, xaxis_title="terminal equity (log10 $)",
                        margin=dict(t=30, b=10), showlegend=False)
        st.plotly_chart(h, use_container_width=True)
        st.caption(
            f"Mean terminal ${df['final'].mean():,.0f} against a median of ${s['median']:,.0f}. "
            "A mean far above the median means the average is carried by a thin right tail "
            "while the typical path does much worse.")
    else:
        st.info("Set the parameters and press Run simulation.")

# ---------------------------------------------------------------- notes
with tab_notes:
    st.subheader("What this study found")
    st.markdown(
        "- At **matched (~1x) exposure**, the option route underperforms owning the "
        "underlying by its carry: dividends forgone, time value paid, and frictions. "
        "Across 300 Monte Carlo paths it beat buy-and-hold **2.3%** of the time.\n"
        "- **Taxes decide it.** Rolling realizes gains annually, largely at short-term "
        "rates, while buy-and-hold defers to a single liquidation. After tax the "
        "strategies beat buy-and-hold in 10 of 32 historical configurations.\n"
        "- **Full-capital leverage is a right-skewed lottery**: median 0.31x buy-and-hold "
        "with 18.7% ruin, alongside a thin tail reaching many multiples.\n"
        "- The PMCC's apparent full-capital edge in the historical paths **did not survive** "
        "the distribution (median 0.14x buy-and-hold, 19% ruin). It was path luck."
    )
    st.subheader("Known modeling biases (SPEC Section 9)")
    st.markdown(
        "1. Options are priced synthetically, never traded prices.\n"
        "2. A VXN/VIX-based parametric term-and-skew surface approximates a surface that "
        "in reality moves richly across strike and maturity. This is the largest lever, "
        "which is why four scenarios were pre-registered before any result existed.\n"
        "3. American exercise is modeled via Barone-Adesi-Whaley.\n"
        "4. A single short rate is used rather than a term-matched curve.\n"
        "5. The continuous futures series carries quarterly roll seams.\n"
        "6. Whole-contract granularity at a $10k account can force a run entirely to cash "
        "for high-priced underlyings after tax erosion. Such runs are flagged.\n"
        "7. The model carries almost no variance risk premium (mean VXN 21.32% against "
        "20.79% realized), so the PMCC's income case is understated.\n"
        "8. Two overlapping historical start years cannot support inference, which is why "
        "the leverage question is answered by Monte Carlo."
    )
    st.subheader("Pre-registration")
    st.markdown(
        "The vol scenarios, friction model, tax treatment, sizing rule, and exercise style "
        "were fixed in SPEC before the run matrix existed, so none was tuned to results.")

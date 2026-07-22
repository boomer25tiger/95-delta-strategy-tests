# 95-delta-strategy-tests

A local Python study comparing deep in-the-money (0.95 delta) call strategies against
buy-and-hold on the Nasdaq-100 (QQQ, NQ) and the S&P 500 (SPY, ES). A Streamlit app
provides a historical simulator and a forward Monte Carlo.

**This is a research and study tool. It is not investment advice.**

## Why I built this

I came across the 0.95-delta call as a proxy for owning the underlying. One deep
in-the-money call moves about 95 cents for every dollar the underlying moves, so it
reproduces almost all of the return exposure while committing a fraction of the
capital that buying the shares outright would require. The rest of the capital stays
free. That is the appeal, the same directional exposure for less money down.

I wanted to know whether that capital efficiency survives a long holding period. If a
deep in-the-money call really is a cheaper way to own an index, it should hold up as a
standing substitute for owning one, over decades of real market history and after
every cost I could model. So I tested it that way, over two indices, two start years,
two tenors, both tax treatments, and two ways of deploying the freed capital, against
simply buying and holding the same underlying.

**The 0.95-delta call works best as a bullish position on a particular asset over a
defined horizon. It works poorly as a permanent replacement for owning one.** Two
findings drive that.

- The costs accumulate quietly. At matched (~1x) exposure the freed capital earns
  roughly what the option's embedded financing charges, so those two cancel. What
  remains is dividends forgone, time value paid, frictions, and taxes. Taxes matter
  most, because rolling realizes gains every cycle while owning shares defers them, so
  the gap widens with every year held. The longer I intended to hold, the worse the
  substitution looked.
- Leverage is what would make the capital efficiency pay, and it brings ruin risk with
  it. Deploying the freed capital into additional contracts lifts exposure to roughly
  2.6x. Across 300 forward paths that version ends below its starting capital in the
  median case after 15 years, and loses 90% or more in 18.7% of paths.

The structure does one thing well. It expresses a directional view on a specific asset
with a known maximum loss over a window I choose. Costs that barely register across a
few months decide the outcome across years.

## Why the options are synthetic

Free historical option prices for these underlyings do not go back to 2001. No
provider offers them, so the backtest cannot replay real fills. The model instead
prices every option at each date from a volatility input, which makes every result
model-derived. The volatility surface is the largest single lever in the study, so I
fixed four surface scenarios in [SPEC.md](SPEC.md) Section 5.1 before running
anything, and every headline number carries its spread across them.

## What I found

### Taxes decide the comparison

At matched (~1x) exposure the option route tracks the underlying closely before tax
and falls steadily behind after tax. Rolling realizes gains every cycle, short-term at
the 1-year tenor, while buy-and-hold defers everything to a single liquidation.

![Tax effect](figures/fig1_tax_effect.png)

Across the 32 matched-exposure historical configurations, the strategies beat their
own buy-and-hold baseline in 20 of 32 before tax and **10 of 32 after tax**. The
0.95-delta replacement wins 3 of 16 after tax with a median of 0.70x. The covered-call
variant wins 7 of 16 with a median of 0.89x.

![Beat rates](figures/fig3_beat_rates.png)

### At 1x the answer sits close to arithmetic

A 0.95-delta call is 95% of the underlying by construction. Its return equals the
underlying's return, minus dividends forgone, minus time value paid, minus frictions,
minus taxes, plus interest on the freed cash. On QQQ from 2011 the entire
volatility-surface choice moves the long leg's carry by roughly $17 to $55 a year per
$10,000, against about $60 a year of dividend drag. The elaborate surface barely
registers at 1x. It dominates once leverage compounds it.

### Leverage turns it into a lottery, and two historical paths could not show that

The two start years, 2001 and 2011, both run to 2026, which makes them nested slices
of a single price history. They cannot support a distributional claim. Running the
same verified engine over 300 forward GBM paths answers the question properly.

![Monte Carlo](figures/fig2_montecarlo.png)

| configuration | median terminal | vs buy-and-hold | beats B&H | P(lose 90%) |
|---|---|---|---|---|
| matched (~1x) | 2.37x start | 0.90x | 2.3% | 0.0% |
| full-capital (~2.6x) | 0.91x start | 0.31x | 23.0% | 18.7% |
| covered call, full-capital | 0.40x start | 0.14x | 5.3% | 19.0% |

The full-capital median path ends below its starting capital after 15 years while its
95th percentile reaches many multiples. Its mean sits far above its median, so a thin
right tail carries the average while the typical path disappoints.

The covered-call variant at full capital looked like 22x to 38x buy-and-hold in the
historical runs. Across the distribution its median comes in at 0.14x buy-and-hold
with 19% ruin. Two lucky nested windows explain that historical result.

### Why full-capital appears only as a band

A single full-capital configuration spans orders of magnitude across the four
pre-registered volatility scenarios. Those point estimates carry no usable signal, so
the report shows their spread and keeps them out of the headline.

![Sensitivity](figures/fig4_sensitivity.png)

### What this means in practice

The capital efficiency is real, and it is also why the substitution fails over long
horizons. Freeing capital only pays when that capital earns more than the option's
carry. The option's embedded financing sits at about the same short rate the freed
cash earns, so at 1x the two offset and the drag items decide the outcome. Putting the
freed capital to work at higher leverage would make it pay, and that is exactly what
brings in the ruin risk the distribution exposes.

That leaves the 0.95-delta call as an instrument for a directional view, held over a
window I choose and sized deliberately. Held indefinitely as a stand-in for ownership,
its costs accumulate in the way this study measures.

## What I tested

Underlyings {QQQ, NQ, SPY, ES} × start years {2001, 2011} × strategies {0.95-delta
replacement, poor man's covered call} × tenors {1y, 2y} × sizing {matched-exposure,
full-capital} × four pre-registered vol scenarios × two tax modes, giving 512 runs
plus buy-and-hold baselines. Every run starts with $10,000 and uses daily data.

Each run reports final equity, total return, CAGR, annualized volatility, Sharpe and
Sortino over the T-bill, max drawdown, Calmar, realized average leverage, roll count,
friction paid, tax paid, and terminal multiple against its own baseline.

## Method

- **Pricing.** Black-Scholes with a continuous dividend yield handles the equity
  underlyings and Black-76 handles the futures. Barone-Adesi-Whaley extends both to
  American exercise. The verification script checks the pricer against Hull's
  reference values, confirms put-call parity to machine precision, matches Black-76 to
  Black-Scholes with q equal to r, and agrees with a 1000-step binomial tree.
- **Volatility.** σ(K,T) = VXN × term_factor(T) + skew_slope × log-moneyness, anchored
  so at-the-money 30-day maps to VXN exactly. The Nasdaq side uses VXN and falls back
  to VIX. The S&P side uses VIX natively.
- **Rates.** The 13-week T-bill (^IRX) discount quote converts to a continuous rate,
  which drives discounting and interest on idle cash alike.
- **Frictions.** A half-spread set as a percentage of underlying notional per side,
  plus a per-contract commission. The engine charges both every time it opens or
  closes a leg.
- **Taxes.** Annual accounting comes out of the account so it compounds. ETF options
  realize short or long term by holding period. Futures and their options follow IRC
  Section 1256 with 60/40 treatment marked to market. Buy-and-hold defers share gains
  to a terminal liquidation. Net capital losses carry forward.
- **Strategies.** The long leg solves numerically for the strike at 0.95 delta and
  rolls at 60 days to expiry. The covered-call variant sells one 0.30-delta call per
  long call at about 35 days and rolls it monthly, moving up and out whenever the
  underlying breaches the short strike.

## Pre-registration

I wrote the volatility scenarios, the friction model, the tax treatment, the sizing
rule, the exercise style, and the choice of headline metric into [SPEC.md](SPEC.md)
before the run matrix existed. I did not select any of them after seeing results.
SPEC.md records the refinements I considered later as extensions and keeps them out of
the headline.

## Known biases

1. The model prices every option synthetically. No price here came from a real trade.
2. A parametric term-and-skew surface stands in for a real surface that moves richly
   across strike and maturity. This is the largest lever.
3. A single short rate stands in for a term-matched curve.
4. The continuous futures series carries quarterly roll seams.
5. Whole-contract granularity at a $10,000 account can push a run entirely into cash
   for high-priced underlyings after tax erosion. The reported cash fraction flags
   those runs.
6. The model carries almost no variance risk premium, with mean VXN at 21.32% against
   20.79% realized over 2011-2026, so it understates the covered call's income case.
7. Two overlapping historical start years cannot support inference, which is why the
   Monte Carlo answers the leverage question.
8. The model ignores wash-sale rules.

## Setup

```
pip install -r requirements.txt
```

## Usage

Every module verifies itself when run directly.

```
python verify_pricing.py    # pricer against Hull, parity, Black-76, binomial tree
python data.py              # fetch and cache series, print ranges, sanity checks
python vol.py               # volatility surface checks
python backtest.py          # single-run engine checks
python results.py           # the full 512-run matrix, writes cache/results.csv
python montecarlo.py        # the forward distribution
python plots.py             # regenerate figures/
streamlit run app.py        # interactive historical and Monte Carlo simulators
```

## Layout

```
pricing.py         Black-Scholes, Black-76, Greeks, American via BAW, solvers
vol.py             VXN/VIX term-and-skew surface, pre-registered scenarios
data.py            fetch and cache QQQ, NQ, SPY, ES, NDX, VXN, VIX, IRX
backtest.py        strategy engine, sizing, roll logic, frictions, taxes, metrics
results.py         the full run matrix and the metrics tables
montecarlo.py      forward GBM paths through the same engine
plots.py           static figures
app.py             Streamlit historical and Monte Carlo simulators
verify_pricing.py  pricer verification
figures/           generated figures
cache/             data and results cache (git-ignored)
```

[SPEC.md](SPEC.md) carries the full methodology and every decision behind it.

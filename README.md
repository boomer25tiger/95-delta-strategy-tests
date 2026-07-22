# 95-delta-strategy-tests

A local Python study comparing deep in-the-money (0.95 delta) call strategies against
buy-and-hold on the Nasdaq-100 (QQQ, NQ) and the S&P 500 (SPY, ES), with a Streamlit
app that provides a historical simulator and a forward Monte Carlo.

**Treat this as a research and study tool rather than investment advice.**

## Why I built this

While studying options greeks, I came across the 0.95-delta call as a proxy for owning a stock. A deep
in-the-money call moves about 95 cents for every dollar the underlying moves, so a
single contract reproduces almost all of the return exposure while tying up a fraction
of what buying the shares outright would cost, leaving the remainder of the capital
free for investment elsewhere. Given such an "efficient use of capital", I wanted to know whether that capital efficiency survives a long holding period. A deep in-the-money call that offers a cheaper route to owning an
index, for example, should hold up as a standing substitute for owning one, across decades of real
market history and after every cost I could model, so I tested it that way over two
indices, two start years, two tenors, both tax treatments, and two ways of deploying
the freed capital, always against simply buying and holding the same underlying.

**Two findings pushed me toward the same conclusion, that the 0.95-delta call works
best as a bullish position on a particular asset over a defined horizon and poorly as
a permanent replacement for owning one.**

- Costs accumulate quietly at matched (~1x) exposure, where the freed capital earns
  roughly what the option's embedded financing charges, so the two cancel and leave
  dividends forgone, time value paid, frictions, and taxes to erode the position.
  Taxes do the most damage, since rolling realizes gains every cycle while owning
  shares defers them, which widens the gap with each additional year held and makes
  the substitution look worse the longer I planned to keep it.
- Leverage is what would finally make the capital efficiency pay, and it carries ruin
  risk in the same motion. Deploying the freed capital into additional contracts lifts
  exposure to roughly 2.6x, and across 300 forward paths that version ends below its
  starting capital in the median case after 15 years while losing 90% or more in 18.7%
  of them.

Where the structure genuinely earns a place is in expressing a directional view on a
specific asset with a known maximum loss over a window I choose, because the costs
that barely register across a few months are the same ones that decide the outcome
across years.

## Why the options are synthetic

Free historical option prices for these underlyings do not go back to 2001, and no
provider sells them, so the backtest cannot replay real fills. The model instead
prices every option at each date from a volatility input, which makes every number
here model-derived rather than observed. Since the volatility surface turned out to be
the largest single lever in the study, I fixed four surface scenarios in
[SPEC.md](SPEC.md) Section 5.1 before running anything, and every headline figure
carries its spread across those four.

## What I found

### Taxes decide the comparison

At matched (~1x) exposure the option route tracks the underlying closely before tax
and falls steadily behind after it, because rolling realizes gains every cycle,
short-term at the 1-year tenor, while buy-and-hold defers everything to a single
liquidation.

![Tax effect](figures/fig1_tax_effect.png)

Across the 32 matched-exposure historical configurations the strategies beat their own
buy-and-hold baseline in 20 of 32 before tax and **10 of 32 after tax**, with the
0.95-delta replacement winning 3 of 16 at a median of 0.70x and the covered-call
variant winning 7 of 16 at a median of 0.89x.

![Beat rates](figures/fig3_beat_rates.png)

### At 1x the answer sits close to arithmetic

A 0.95-delta call amounts to 95% of the underlying by construction, so its return
comes out as the underlying's return, minus dividends forgone, minus time value paid,
minus frictions, minus taxes, plus interest on the freed cash. Measured on QQQ from
2011, the entire volatility-surface choice moves the long leg's carry by roughly $17
to $55 a year per $10,000, against about $60 a year of dividend drag alone. The
elaborate surface barely registers at 1x, yet leverage compounds that same assumption
until it swamps everything else.

### Leverage turns it into a lottery, and two historical paths could not show that

Both start years run to 2026, which makes 2011 a nested slice of 2001 rather than an
independent sample, so no amount of care with those two paths can support a
distributional claim. Running the same verified engine over 300 forward GBM paths
answers the question properly.

![Monte Carlo](figures/fig2_montecarlo.png)

| configuration | median terminal | vs buy-and-hold | beats B&H | P(lose 90%) |
|---|---|---|---|---|
| matched (~1x) | 2.37x start | 0.90x | 2.3% | 0.0% |
| full-capital (~2.6x) | 0.91x start | 0.31x | 23.0% | 18.7% |
| covered call, full-capital | 0.40x start | 0.14x | 5.3% | 19.0% |

The full-capital median path ends below its starting capital after 15 years while the
95th percentile of the same distribution reaches many multiples. Because the mean sits
far above the median, a thin right tail carries the average while the typical path
disappoints.

The covered-call variant at full capital looked like 22x to 38x buy-and-hold in the
historical runs, yet across the forward distribution its median comes in at 0.14x
buy-and-hold with 19% ruin, which means two lucky nested windows produced that
historical result.

### Why full-capital appears only as a band

A single full-capital configuration spans orders of magnitude across the four
pre-registered volatility scenarios, so those point estimates carry no usable signal
and the report shows their spread while keeping them out of the headline.

![Sensitivity](figures/fig4_sensitivity.png)

### What this means in practice

The capital efficiency is real, and the same efficiency explains why the substitution
fails over long horizons. Freeing capital only pays when the freed capital earns more
than the option's carry, and since the option's embedded financing sits at about the
same short rate that idle cash earns, the two offset at 1x and the drag items decide
the outcome. Higher leverage would finally make the efficiency pay, which is precisely
how the ruin risk the distribution exposes enters the picture.

What survives all of that is a case for holding the 0.95-delta call as a directional
instrument, over a window I choose and at a size I set deliberately, because holding
it indefinitely as a stand-in for ownership lets the costs accumulate in exactly the
way this study measures.

## What I tested

Underlyings {QQQ, NQ, SPY, ES} × start years {2001, 2011} × strategies {0.95-delta
replacement, poor man's covered call} × tenors {1y, 2y} × sizing {matched-exposure,
full-capital} × four pre-registered vol scenarios × two tax modes, giving 512 runs
alongside the buy-and-hold baselines, with every run starting from $10,000 on daily
data.

Each run reports final equity, total return, CAGR, annualized volatility, Sharpe and
Sortino over the T-bill, max drawdown, Calmar, realized average leverage, roll count,
friction paid, tax paid, and terminal multiple against its own baseline.

## Method

- **Pricing.** Black-Scholes with a continuous dividend yield handles the equity
  underlyings while Black-76 handles the futures, and Barone-Adesi-Whaley extends both
  to American exercise. The verification script checks the pricer against Hull's
  reference values, confirms put-call parity to machine precision, matches Black-76 to
  Black-Scholes with q equal to r, and agrees with a 1000-step binomial tree.
- **Volatility.** σ(K,T) = VXN × term_factor(T) + skew_slope × log-moneyness, anchored
  so that at-the-money 30-day vol maps to VXN exactly, with the Nasdaq side falling
  back to VIX wherever VXN gaps and the S&P side using VIX natively.
- **Rates.** The 13-week T-bill (^IRX) discount quote converts to a continuous rate,
  which drives discounting and interest on idle cash alike.
- **Frictions.** A half-spread set as a percentage of underlying notional per side,
  plus a per-contract commission, both of which the engine charges every time it opens
  or closes a leg.
- **Taxes.** Annual accounting comes out of the account so that the tax compounds
  against later sizing. ETF options realize short or long term by holding period,
  futures and their options follow IRC Section 1256 with 60/40 treatment marked to
  market, and buy-and-hold defers share gains to a terminal liquidation, while net
  capital losses carry forward.
- **Strategies.** The long leg solves numerically for the strike at 0.95 delta and
  rolls at 60 days to expiry, while the covered-call variant sells one 0.30-delta call
  per long call at about 35 days and rolls it monthly, moving up and out whenever the
  underlying breaches the short strike.

## Pre-registration

I wrote the volatility scenarios, the friction model, the tax treatment, the sizing
rule, the exercise style, and the choice of headline metric into [SPEC.md](SPEC.md)
before the run matrix existed, and I did not select any of them after seeing results.
SPEC.md records the refinements I considered later as extensions, which keeps them out
of the headline.

## Known biases

1. The model prices every option synthetically, so no price in this study came from a
   real trade.
2. A parametric term-and-skew surface stands in for a real surface that moves richly
   across strike and maturity, which makes this the largest lever in the study.
3. A single short rate stands in for a term-matched curve.
4. The continuous futures series carries quarterly roll seams.
5. Whole-contract granularity at a $10,000 account can push a run entirely into cash
   for high-priced underlyings once taxes erode the balance, and the reported cash
   fraction flags any run affected that way.
6. The model carries almost no variance risk premium, with mean VXN at 21.32% against
   20.79% realized over 2011-2026, so the covered call's income case comes out
   understated.
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

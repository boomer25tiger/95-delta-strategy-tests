# 95-Delta Strategy Tests — Specification

A backtest study comparing deep-in-the-money (0.95 delta) call strategies against
buy-and-hold on QQQ and NQ, over two start years (2001 and 2011), with an
interactive Streamlit layer. Everything runs locally in Python. $10,000 starting
capital, daily data.

This document is the locked spec. Read it fully before writing code.

---

## 1. The core data constraint (read first)

Free historical option prices for QQQ or NQ back to 2001 do not exist. yfinance
returns only today's chain; Alpaca gates options history behind a paid tier. So the
backtest cannot replay real fills. Every option is priced **synthetically** at each
date with a pricing model driven by a historical volatility input:

- QQQ options: Black-Scholes-Merton with continuous dividend yield q.
- NQ options: Black-76 on the futures price (no dividend term).

The outputs are model-derived, not traded prices. This is standard in options
research and must be stated plainly in the app and README.

---

## 2. Data sources and series

All via yfinance, cached locally to `cache/` (git-ignored). Ranges confirmed
2026-07-20 from this machine:

| Series   | yfinance ticker | Range start  | Use |
|----------|-----------------|--------------|-----|
| QQQ      | `QQQ`           | 1999-03-10   | QQQ spot path; dividends for drag/total return |
| NQ front | `NQ=F`          | 2000-09-18   | NQ futures price path (Black-76 underlying) |
| NDX      | `^NDX`          | 1985-10-01   | Reference / sanity check / fallback proxy |
| VXN      | `^VXN`          | 2001-01-23   | At-the-money implied-vol level input |
| VIX      | `^VIX`          | 1990-01-02   | Fallback vol if VXN gaps |
| 13w bill | `^IRX`          | 1960-01-04   | Risk-free rate and cash interest (discount yield, %) |

VXN starts 2001-01-23, which covers both start years. Fetch QQQ with
`auto_adjust=False` so raw Close and the Dividends column are both available; use
Adj Close for the QQQ total-return buy-and-hold.

---

## 3. Strategies

### 3.1 Buy-and-hold baselines (per underlying, per start year)

- **QQQ:** buy shares with $10,000 at the start close, reinvest dividends. Equity_t
  = 10000 × AdjClose_t / AdjClose_0.
- **NQ:** fully cash-collateralized long future scaled to $10,000 notional at entry.
  Post $10,000 cash earning the T-bill rate; hold futures whose P&L accrues on top;
  roll the continuous series quarterly. Equity_t = 10000 + cumulative futures P&L +
  cash interest. No dividend (carry is embedded in F).

### 3.2 95-delta call as stock replacement

Long a single deep-ITM call, rolled forward. No short leg.

- **Strike:** solve numerically for K such that call delta = 0.95 at entry, using
  the vol model σ(K,T) from Section 5 and the term rate r(T) from Section 6. For NQ
  (Black-76) the futures call delta caps at e^(−rT); when 0.95 is infeasible (short
  rates above roughly 5% for 1y or 2.5% for 2y) the strike falls to the deepest
  feasible, N(d1)=0.999. QQQ (cap e^(−qT) ≈ 0.99) is never bound.
- **Tenor:** swept variable, 1-year and 2-year LEAPS.
- **Roll:** when time-to-expiry falls to 60 calendar days, close the position at
  model value net of the bid/ask cost and open a fresh 0.95-delta call of the same
  tenor. Apply the sizing mode (Section 4) at every entry and roll.
- **Pricing:** European (BS for QQQ, Black-76 for NQ). American early exercise of
  deep-ITM calls around ex-dividend is ignored; documented as a bias (Section 9).
- **Dividends:** the call holder receives none. QQQ dividends enter only through q
  in the price. The gap versus QQQ buy-and-hold (which reinvests dividends) is the
  dividend drag, a genuine cost of the option route.

### 3.3 Poor man's covered call (PMCC)

Long leg identical to Section 3.2 (the 0.95-delta call, tenor swept). Add a short
near-dated call.

- **Short leg:** sell 1 call per long call at ~0.30 delta, ~35 days to expiry,
  rolled monthly. If the underlying rallies through the short strike, roll up-and-out
  (buy back, sell a higher strike further out) rather than allow assignment. The
  short caps upside near its strike and collects premium that lowers cost basis.
- **Net position:** priced by summing the two legs each day.

---

## 4. Sizing modes (apply to the long-call leg of both strategies)

The user framed exposure by notional (dollars of underlying controlled), matching
the worked example "25 cents on the dollar, so $2,500 of options controls $10,000 of
underlying."

### 4.1 Matched-exposure

Buy enough calls to control $10,000 of underlying notional at entry, hold the rest
of the $10,000 as cash at the T-bill rate.

- contracts ≈ round( 10000 / (multiplier × spot) ).
- Option outlay ≈ 10000 × (call_price / spot); remainder is cash.
- A 0.95-delta call gives delta-dollar exposure of about $9,500 against the $10,000
  of controlled notional. Report both framings.
- The notional target is the current account equity at each entry and roll (literally
  $10,000 at the first entry), so matched-exposure holds ~1x exposure as the account
  compounds rather than a fixed $10,000 that decays in relative terms.

### 4.2 Full-capital (most realistic implementation)

Deploy the whole account into 0.95-delta calls each cycle.

- At entry: contracts = floor( available_equity / (contract_cost + friction) ). Any
  remainder sits in cash at the T-bill rate.
- At each roll: close all calls at model value net of friction, then rebuy the max
  whole contracts the new equity affords. Leverage floats near its natural 3-4x and
  cannot go below zero (long calls need no margin).
- Whole-contract granularity is respected. At both start years the account affords
  several contracts; coarseness only appears late in the sample, by which point a
  compounding account has grown enough to absorb it. Report realized average leverage
  so the effect is visible.

---

## 5. Volatility model

VXN is a 30-day at-the-money implied vol for NDX. A 0.95-delta call is deep ITM,
whose correct IV rides the downside skew (by put-call parity it equals the IV of the
equivalent OTM put) and a 1-2y term, both above ATM VXN. Pricing at flat VXN would
understate the call's time value and flatter the option strategies.

Model: σ(K, T) = VXN_t × term_factor(T) + skew_slope × moneyness(K), where
moneyness is measured as log(K/S) or (K−S)/S. `term_factor` and `skew_slope` are
configurable with documented defaults; VXN falls back to VIX where it gaps. Expose
the parameters in the app and default `skew_slope` so deep-ITM strikes carry higher
IV than ATM. State the assumption in the UI. This is the single largest modeling
lever; keep it swappable.

### 5.1 Pre-registered vol sweep (fixed before any backtest)

The surface is unobservable back to 2001, so the two levers are not tuned to
results. Four settings are fixed in advance and frozen; the backtest reports every
metric across all four as a robustness band. Selecting or altering these after seeing
P&L would be specification search and is disallowed. `base` is the headline point
estimate; `flat` and `steep` are the bounds.

| name       | term_slope | skew_slope | role                              |
|------------|-----------:|-----------:|-----------------------------------|
| flat       | 0.00       | 0.00       | sigma = VXN; optimistic bound     |
| term_only  | 0.03       | 0.00       | term premium, no skew             |
| base       | 0.03       | -0.15      | headline point estimate (default) |
| steep      | 0.05       | -0.25      | pessimistic bound (richer options)|

`moneyness` (log) and `vol_floor` (0.05) are held constant across all four. More
realistic surface variants (mean-reverting term_factor, maturity-decaying skew) may
be added only as separately labeled robustness extensions decided on a-priori
grounds; they never replace the headline.

---

## 6. Risk-free rate and cash interest

Use `^IRX` (13-week T-bill discount yield, in percent) as the short rate for both
pricing (r) and interest paid on idle cash. Documented simplification: a single
short rate rather than a term-matched curve. Rho on 1-2y LEAPS has some sensitivity
to this; a term-matched curve is a possible later refinement. Convert the discount
quote to a decimal continuous rate for the engine.

---

## 7. Contract specifications

- **QQQ:** equity option, multiplier 100 shares per contract, Black-Scholes with q.
- **NQ:** modeled as the **Micro** E-mini Nasdaq-100 (MNQ), multiplier **$2 per index
  point**, Black-76. The full E-mini ($20/point) makes one option control ~$30k+ of
  notional, too coarse for a $10,000 account, so the micro is the realistic unit at
  this size. The multiplier is only the position unit; pricing is synthetic either
  way. Continuous future rolled quarterly.

---

## 8. Frictions

- **Bid/ask:** fixed half-spread in dollars per share per leg, applied on entry and
  exit. Default $0.25/share ($25 per QQQ contract; per-point-equivalent for NQ).
  Configurable; deep-ITM LEAPS spreads run wide.
- **Cash interest:** idle cash earns the `^IRX` short rate.
- No exchange commissions modeled by default (can be added as a flat per-contract fee).

---

## 9. Known modeling biases (surface these in the app and README)

1. Options are synthetically priced, not real fills.
2. VXN-based vol with a parametric term and skew adjustment approximates a surface
   that in reality moves richly across strike and maturity.
3. European pricing ignores American early exercise near QQQ ex-dividend dates.
4. Single short rate (`^IRX`), not a term-matched curve.
5. NQ continuous-future series has quarterly roll seams.
6. Whole-contract granularity at a $10k account. The micro futures and compounding
   mitigate it for lower-priced entries, but for high-priced underlyings (SPY and QQQ
   near recent levels) combined with after-tax erosion the account can fall below one
   contract's cost and be forced entirely to cash. Such degenerate runs are flagged by
   the reported cash fraction, not silently scored as strategy performance.
7. The PMCC short leg is sold at the model's VXN/VIX-implied vol and marked at that
   same vol, so it captures only the realized implied-minus-realized spread, which is
   small in sample (mean VXN 21.32% vs 20.79% realized QQQ vol over 2011-2026, a
   0.53-point spread). Measured net short-leg P&L is near zero. Real covered-call
   income leans heavily on the variance risk premium, which this model barely
   represents, so the PMCC's income case is understated and what remains is
   essentially the delta reduction and the volatility-drag effect.

---

## 10. Test matrix

Underlyings {QQQ, NQ} × start years {2001, 2011} × strategies {95-delta replacement,
PMCC} × tenors {1y, 2y} × sizing {full-capital, matched-exposure} = 32 strategy runs,
plus 4 buy-and-hold baselines (one per underlying per start year).

Each of the 32 strategy runs is evaluated under all four pre-registered vol settings
(Section 5.1). The base case is the headline and the four settings are reported as a
band. Buy-and-hold baselines are vol-independent.

Per run, report: final equity, total return, CAGR, annualized volatility, Sharpe
(excess over the T-bill), max drawdown, realized average leverage (full-capital),
number of rolls, total friction paid, and the terminal multiple versus its
buy-and-hold baseline.

---

## 11. Interactive layer (Streamlit, two separate simulators)

### 11.1 Historical simulator

Runs over the real QQQ and NQ price paths. Controls for underlying, start year,
strategy, tenor, and sizing mode. Shows the strategy equity curve against its
buy-and-hold baseline, the metrics table, and a time control that walks the
divergence forward. Both underlyings viewable.

### 11.2 Monte Carlo simulator

Forward GBM paths of the underlying. Controls for drift, volatility, horizon, number
of paths, and sizing/leverage. Runs the strategies across every path and shows the
fan of terminal outcomes, the probability the strategy beats buy-and-hold, and the
ruin probability of the full-capital mode. Demonstrates why leverage diverges hard or
wipes out.

---

## 12. Project layout

```
95-delta-strategy-tests/
  SPEC.md            this file
  README.md          public-facing overview and run instructions
  requirements.txt
  .gitignore
  pricing.py         BS + Black-76, Greeks, 0.95-delta strike solver, implied vol
  vol.py             VXN-based sigma(K, T) with term and skew parameters
  data.py            fetch + cache QQQ, NQ=F, ^NDX, ^VXN, ^VIX, ^IRX
  backtest.py        strategy engine, sizing modes, roll logic, metrics
  montecarlo.py      GBM path generation + strategy application
  verify_pricing.py  pricer checks vs known values and parity
  app.py             Streamlit, the two simulators
  cache/             git-ignored data cache
```

---

## 13. Build order (piecemeal, verify each step before the next)

1. `pricing.py` + `verify_pricing.py`. Check Black-Scholes against Hull's reference
   (call 4.759, put 0.808), put-call parity to machine precision, Black-76 equals BS
   with q = r, and a 0.95-delta strike solve that round-trips its delta. Run, all
   pass, before moving on.
2. `data.py`. Fetch and cache all series, print ranges, sanity-check values.
3. `vol.py`. The vol model; confirm ATM maps to VXN and deep-ITM strikes carry
   higher IV than ATM under the default skew.
4. `backtest.py`. One run end to end first (QQQ, 2011, matched-exposure, 1y),
   sanity-check against buy-and-hold, then scale to the full 32-run matrix.
5. Metrics table and static plots.
6. `app.py` historical simulator.
7. `app.py` Monte Carlo simulator.
8. README, then publish to GitHub.

---

## 14. Pricing math source

The closed-form Black-Scholes price, the five Greeks, and the bisection implied-vol
solver are adapted from a prior verified engine (verified against Hull, put-call
parity, and published Greek values). Black-76 for the futures side is BS applied to
the forward with the dividend term set so delta_future = e^(−rT) N(d1). Keep one
source of truth in `pricing.py`; do not scatter the math.

---

## 15. Extensions (pre-registered 2026-07-21)

Decided before the run matrix exists, so none is tuned to results. Each is swept or
toggled, not silently baked in.

### 15.1 S&P 500 underlying (added alongside Nasdaq)

- SPY spot (from 1993-01-29), Black-Scholes with trailing dividend yield q, American
  (Section 15.2).
- ES=F continuous (from 2000-09-18), modeled as the Micro E-mini MES at $5 per index
  point, Black-76.
- Vol input is `^VIX`, the native 30-day ATM implied vol for the S&P; the Nasdaq side
  keeps VXN with VIX fallback. Same term+skew surface and pre-registered sweep
  (Section 5.1) apply.

The test matrix (Section 10) now spans two underlying families, {QQQ, NQ} and
{SPY, ES}.

### 15.2 American exercise (Barone-Adesi-Whaley)

Calls are priced American via the BAW quadratic approximation, which collapses to the
European price when r ≥ q (a call carries no early-exercise value then). The
0.95-delta strike is still selected on the European / Black-76 delta of Section 3.2;
only the valuation is American. The early-exercise premium on a deep-ITM 0.95-delta
call is 0 when r ≥ q and roughly 0.25–1% of spot when q > r (larger for higher-yield
SPY, the 2y tenor, and the low-rate 2009–2021 window). This removes bias #3.

### 15.3 Frictions (replaces the Section 8 half-spread default)

Half-spread is a percentage of underlying notional (multiplier × spot) per side,
default 0.05%, the natural scale for a deep-ITM option whose premium is mostly
recoverable intrinsic. Plus a per-contract commission (default $0.65) and a dollar
floor, charged on entry, exit, and each roll. An optional regime multiplier widens
the spread with VIX/VXN. Spreads are unobservable historically, so this is a
pre-registered assumption and may be swept.

### 15.4 Taxes (both modes reported)

Every run is reported pre-tax (tax-advantaged account) and after-tax (taxable), side
by side. Annual accounting; tax is deducted from the account at year-end so it
compounds correctly and reduces later sizing.

- ETF options (QQQ, SPY): each roll realizes a gain or loss, short-term if the closed
  leg was held under 12 months. The 1y tenor realizes short-term, the 2y long-term.
- Futures and their options (NQ, ES): IRC Section 1256, 60/40 (60% long-term, 40%
  short-term), marked to market at year-end. Applies to the strategy and the futures
  buy-and-hold baseline.
- ETF buy-and-hold (QQQ, SPY): only dividends taxed annually (qualified / long-term);
  share gains defer to a terminal liquidation.
- Cash interest taxed annually at the ordinary rate.
- Default rates 35% short-term / ordinary, 20% long-term / qualified; a moderate
  24 / 15 bracket is configurable. Net capital losses carry forward. Wash sales are
  ignored, documented as a simplification.

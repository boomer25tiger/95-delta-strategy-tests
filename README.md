# 95-delta-strategy-tests

Backtests of deep-in-the-money (0.95 delta) call strategies against buy-and-hold on
QQQ and the Nasdaq-100 future (NQ), with an interactive Streamlit layer. Runs locally
in Python.

## What it compares

- **95-delta call as stock replacement** versus holding the underlying.
- **Poor man's covered call** (long a 0.95-delta LEAPS, short a near-dated call)
  versus holding the underlying.

Across two start years (2001 and 2011), 1-year and 2-year LEAPS tenors, and two
position-sizing modes (full-capital and matched-exposure), on a $10,000 account.

## Method note

Historical option prices are not available for free this far back, so every option is
priced **synthetically** at each date with Black-Scholes (QQQ) or Black-76 (NQ),
driven by a VXN-based volatility input. Results are model-derived, not real fills.
See [SPEC.md](SPEC.md) for the full methodology, assumptions, and known biases.

This is a research and study tool, not investment advice.

## Setup

```
pip install -r requirements.txt
```

## Usage

Run the backtests, then launch the interactive app:

```
python backtest.py
streamlit run app.py
```

(Implementation in progress; see [SPEC.md](SPEC.md) for the build plan.)

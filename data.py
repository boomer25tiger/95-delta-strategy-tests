"""Fetch and cache the daily series for the study.

One yfinance Ticker.history(period='max', auto_adjust=False) per series, cached
as CSV under cache/. QQQ keeps raw Close, Adj Close (dividend-reinvested total
return), and Dividends. ^IRX is a 13-week T-bill discount yield in percent and is
converted to a continuous rate before use.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"

SERIES = {
    "QQQ": "QQQ",
    "NQ": "NQ=F",
    "SPY": "SPY",
    "ES": "ES=F",
    "NDX": "^NDX",
    "VXN": "^VXN",
    "VIX": "^VIX",
    "IRX": "^IRX",
}

EXPECTED_START = {
    "QQQ": "1999-03-10",
    "NQ": "2000-09-18",
    "SPY": "1993-01-29",
    "ES": "2000-09-18",
    "NDX": "1985-10-01",
    "VXN": "2001-01-23",
    "VIX": "1990-01-02",
    "IRX": "1960-01-04",
}


def _download(ticker):
    df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df


def get(name, force_refresh=False):
    path = CACHE_DIR / f"{name}.csv"
    if path.exists() and not force_refresh:
        df = pd.read_csv(path, index_col="Date")
        df.index = pd.to_datetime(df.index)
        return df
    df = _download(SERIES[name])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def qqq(force_refresh=False):
    df = get("QQQ", force_refresh)
    out = df[["Close", "Adj Close", "Dividends"]].copy()
    out.columns = ["close", "adj_close", "dividends"]
    return out.astype(float)


def spy(force_refresh=False):
    df = get("SPY", force_refresh)
    out = df[["Close", "Adj Close", "Dividends"]].copy()
    out.columns = ["close", "adj_close", "dividends"]
    return out.astype(float)


def _close(name, force_refresh=False):
    return get(name, force_refresh)["Close"].astype(float).rename(name.lower())


def nq(force_refresh=False):
    return _close("NQ", force_refresh)


def es(force_refresh=False):
    return _close("ES", force_refresh)


def ndx(force_refresh=False):
    return _close("NDX", force_refresh)


def vxn(force_refresh=False):
    return _close("VXN", force_refresh)


def vix(force_refresh=False):
    return _close("VIX", force_refresh)


def irx(force_refresh=False):
    return _close("IRX", force_refresh)


def discount_to_continuous(discount_pct, days=91):
    # 13-week T-bill: price = 1 - d*(days/360) on the ACT/360 discount basis;
    # continuous rate to maturity on ACT/365: r = -ln(price)/(days/365).
    d = np.asarray(discount_pct, dtype=float) / 100.0
    price = 1.0 - d * days / 360.0
    return -np.log(price) / (days / 365.0)


def short_rate_continuous(force_refresh=False):
    s = irx(force_refresh)
    return pd.Series(discount_to_continuous(s.values), index=s.index, name="r")


_results = []


def check(name, passed, detail=""):
    _results.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    force = "--refresh" in sys.argv
    frames = {name: get(name, force) for name in SERIES}

    print("\nseries   ticker    first        last         rows")
    for name, df in frames.items():
        print(f"{name:<7}  {SERIES[name]:<8}  {df.index[0].date()}   "
              f"{df.index[-1].date()}   {len(df):>6}")
    print()

    for name, df in frames.items():
        got = str(df.index[0].date())
        check(f"{name} start == {EXPECTED_START[name]}", got == EXPECTED_START[name],
              "" if got == EXPECTED_START[name] else f"got {got}")
        check(f"{name} index sorted & unique",
              df.index.is_monotonic_increasing and not df.index.has_duplicates)
        check(f"{name} Close has no NaN", int(df["Close"].isna().sum()) == 0,
              f"{int(df['Close'].isna().sum())} NaN")

    q = qqq()
    check("QQQ close > 0", bool((q["close"] > 0).all()))
    check("QQQ adj_close > 0", bool((q["adj_close"] > 0).all()))
    rel = abs(q["adj_close"].iloc[-1] - q["close"].iloc[-1]) / q["close"].iloc[-1]
    check("QQQ adj_close == close on last row", rel < 1e-6, f"rel {rel:.2e}")
    ndiv = int((q["dividends"] > 0).sum())
    check("QQQ dividend payments recorded", ndiv > 40, f"{ndiv} ex-dates, sum {q['dividends'].sum():.2f}")

    j = pd.concat([nq(), ndx()], axis=1, join="inner").dropna()
    ratio = float((j["nq"] / j["ndx"]).median())
    corr = float(j["nq"].corr(j["ndx"]))
    check("NQ ~ NDX level (median ratio in 0.95-1.05)", 0.95 < ratio < 1.05, f"ratio {ratio:.4f}")
    check("NQ vs NDX corr > 0.99", corr > 0.99, f"corr {corr:.5f}")

    sp = spy()
    check("SPY close > 0", bool((sp["close"] > 0).all()))
    rel_sp = abs(sp["adj_close"].iloc[-1] - sp["close"].iloc[-1]) / sp["close"].iloc[-1]
    check("SPY adj_close == close on last row", rel_sp < 1e-6, f"rel {rel_sp:.2e}")
    ndiv_sp = int((sp["dividends"] > 0).sum())
    check("SPY dividend payments recorded", ndiv_sp > 40, f"{ndiv_sp} ex-dates, sum {sp['dividends'].sum():.2f}")
    je = pd.concat([es(), sp["close"].rename("spy")], axis=1, join="inner").dropna()
    ratio_es = float((je["es"] / je["spy"]).median())
    ecorr = float(je["es"].corr(je["spy"]))
    check("ES ~ 10x SPY (index vs 1/10 ETF)", 9.5 < ratio_es < 10.5, f"ratio {ratio_es:.4f}")
    check("ES vs SPY corr > 0.99", ecorr > 0.99, f"corr {ecorr:.5f}")

    v = vxn()
    check("VXN in [5, 100]", bool(((v >= 5) & (v <= 100)).all()), f"range [{v.min():.2f}, {v.max():.2f}]")
    vi = vix()
    check("VIX in [5, 100]", bool(((vi >= 5) & (vi <= 100)).all()), f"range [{vi.min():.2f}, {vi.max():.2f}]")
    jv = pd.concat([v, vi], axis=1, join="inner").dropna()
    vcorr = float(jv["vxn"].corr(jv["vix"]))
    # Level corr is loosened by cross-era regime shifts (VXN >> VIX in 2001-03);
    # a broken fallback would read ~0. Confirms co-movement for the VIX gap fill.
    check("VXN vs VIX corr > 0.70", vcorr > 0.70, f"corr {vcorr:.4f}")

    ix = irx()
    check("IRX in [-1, 20] percent", bool(((ix >= -1) & (ix <= 20)).all()), f"range [{ix.min():.3f}, {ix.max():.3f}]")
    r = short_rate_continuous()
    check("short rate in [-0.02, 0.20]", bool(((r >= -0.02) & (r <= 0.20)).all()),
          f"range [{r.min():.4f}, {r.max():.4f}], last {r.iloc[-1]:.4f}")

    print("\nentry-day levels (first trading day on/after Jan 1):")
    for yr in (2001, 2011):
        cut = pd.Timestamp(yr, 1, 1)
        qd = q.loc[q.index >= cut].iloc[0]
        nd = nq().loc[nq().index >= cut].iloc[0]
        print(f"  {yr}: QQQ close {qd['close']:.2f} on {q.loc[q.index >= cut].index[0].date()}"
              f" | NQ {nd:.2f} on {nq().loc[nq().index >= cut].index[0].date()}")
    vxn_first = v.index[0].date()
    check("VXN available before 2001 entry window", str(vxn_first) <= "2001-01-31",
          f"VXN starts {vxn_first}; 2001 run entry must be >= this date")

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())

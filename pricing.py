"""Black-Scholes-Merton and Black-76 option pricing, Greeks, and solvers.

Single source of truth for the pricing math. BS uses a continuous dividend
yield q; Black-76 is BS applied to the forward with q = r so that the futures
delta reduces to e^(-rT) N(d1).
"""

import math

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_pdf(x):
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _d1_d2(S, K, T, r, q, sigma):
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    if T <= 0.0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    if sigma <= 0.0:
        # sigma -> 0 limit: discounted forward intrinsic
        fwd = S * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        return disc * (max(fwd - K, 0.0) if kind == "call" else max(K - fwd, 0.0))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if kind == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    pdf = _norm_pdf(d1)
    sqrtT = math.sqrt(T)
    gamma = df_q * pdf / (S * sigma * sqrtT)
    vega = S * df_q * pdf * sqrtT
    if kind == "call":
        delta = df_q * _norm_cdf(d1)
        theta = (-S * df_q * pdf * sigma / (2.0 * sqrtT)
                 - r * K * df_r * _norm_cdf(d2)
                 + q * S * df_q * _norm_cdf(d1))
        rho = K * T * df_r * _norm_cdf(d2)
    else:
        delta = -df_q * _norm_cdf(-d1)
        theta = (-S * df_q * pdf * sigma / (2.0 * sqrtT)
                 + r * K * df_r * _norm_cdf(-d2)
                 - q * S * df_q * _norm_cdf(-d1))
        rho = -K * T * df_r * _norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def black76_price(F, K, T, r, sigma, kind="call"):
    return bs_price(F, K, T, r, sigma, q=r, kind=kind)


def black76_greeks(F, K, T, r, sigma, kind="call"):
    # delta w.r.t. F equals e^(-rT) N(d1) since q = r
    return bs_greeks(F, K, T, r, sigma, q=r, kind=kind)


def _baw_call_critical(X, T, r, b, sigma):
    # Newton solve for the critical exercise price S* (Barone-Adesi-Whaley 1987)
    sig2 = sigma * sigma
    NN = 2.0 * b / sig2
    MM = 2.0 * r / sig2
    KK = 1.0 - math.exp(-r * T)
    q2 = (-(NN - 1.0) + math.sqrt((NN - 1.0) ** 2 + 4.0 * MM / KK)) / 2.0
    q2_inf = (-(NN - 1.0) + math.sqrt((NN - 1.0) ** 2 + 4.0 * MM)) / 2.0
    S_inf = X / (1.0 - 1.0 / q2_inf)
    h2 = -(b * T + 2.0 * sigma * math.sqrt(T)) * X / (S_inf - X)
    Si = X + (S_inf - X) * (1.0 - math.exp(h2))
    sqrtT = math.sqrt(T)
    for _ in range(100):
        d1 = (math.log(Si / X) + (b + 0.5 * sig2) * T) / (sigma * sqrtT)
        eul = bs_price(Si, X, T, r, sigma, q=r - b, kind="call")
        rhs = eul + (1.0 - math.exp((b - r) * T) * _norm_cdf(d1)) * Si / q2
        bi = (math.exp((b - r) * T) * _norm_cdf(d1) * (1.0 - 1.0 / q2)
              + (1.0 - math.exp((b - r) * T) * _norm_pdf(d1) / (sigma * sqrtT)) / q2)
        Si_new = (X + rhs - bi * Si) / (1.0 - bi)
        if abs(Si_new - Si) < 1e-8 * X:
            return Si_new, q2
        Si = Si_new
    return Si, q2


def baw_call(S, K, T, r, sigma, q=0.0):
    """American call via Barone-Adesi-Whaley. b = r - q; collapses to European
    when b >= r (q <= 0), where a call has no early-exercise value."""
    if T <= 0.0:
        return max(S - K, 0.0)
    euro = bs_price(S, K, T, r, sigma, q=q, kind="call")
    b = r - q
    if b >= r or sigma <= 0.0:
        return euro
    Sk, q2 = _baw_call_critical(K, T, r, b, sigma)
    if S >= Sk:
        return S - K
    d1 = (math.log(Sk / K) + (b + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    A2 = (Sk / q2) * (1.0 - math.exp((b - r) * T) * _norm_cdf(d1))
    return euro + A2 * (S / Sk) ** q2


def _bisect(f, lo, hi, xtol=1e-12, ftol=1e-14, max_iter=100):
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("root not bracketed")
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= ftol or 0.5 * (hi - lo) <= xtol:
            return mid
        if flo * fmid < 0.0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return mid


def solve_strike_for_delta(S, T, r, sigma, target_delta=0.95, q=0.0, kind="call"):
    """Strike K such that option delta equals target_delta.

    sigma may be a float or a callable sigma(K) for a strike-dependent surface.
    Call delta is monotonically decreasing in K.
    """
    def delta_at(K):
        sig = sigma(K) if callable(sigma) else sigma
        sig = max(sig, 1e-8)  # floor keeps bracketing valid if a callable dips nonpositive at extremes
        return bs_greeks(S, K, T, r, sig, q=q, kind=kind)["delta"]

    if kind == "call":
        lo, hi = S * 1e-3, S * 5.0
        f = lambda K: delta_at(K) - target_delta
        if f(lo) <= 0.0:
            raise ValueError("target delta exceeds e^(-qT); infeasible")
        if f(hi) >= 0.0:
            raise ValueError("target delta below range at K = 5S")
    else:
        lo, hi = S * 1e-3, S * 5.0
        f = lambda K: delta_at(K) - target_delta
    return _bisect(f, lo, hi, xtol=1e-10)


def implied_vol(price, S, K, T, r, q=0.0, kind="call", lo=1e-6, hi=5.0):
    """Bisection implied vol. Price is monotonically increasing in sigma."""
    f = lambda sig: bs_price(S, K, T, r, sig, q=q, kind=kind) - price
    if f(lo) > 0.0 or f(hi) < 0.0:
        raise ValueError("price outside solvable vol range")
    return _bisect(f, lo, hi, xtol=1e-12)

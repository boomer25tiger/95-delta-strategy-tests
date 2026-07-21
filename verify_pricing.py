"""Verification of pricing.py against known values, parity, and self-consistency."""

import math
import sys

from pricing import (
    bs_price, bs_greeks, black76_price, black76_greeks,
    solve_strike_for_delta, implied_vol, baw_call, _norm_cdf, _d1_d2,
)


def _binom_call(S, K, T, r, q, sigma, N=1000, american=True):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    v = [max(S * u**j * d**(N - j) - K, 0.0) for j in range(N + 1)]
    for i in range(N, 0, -1):
        for j in range(i):
            cont = disc * (p * v[j + 1] + (1 - p) * v[j])
            v[j] = max(cont, S * u**j * d**(i - 1 - j) - K) if american else cont
    return v[0]

_results = []


def check(name, passed, detail=""):
    _results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# 1. Hull reference: S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0 -> call 4.759, put 0.808
S, K, T, r, sig = 42.0, 40.0, 0.5, 0.10, 0.20
call = bs_price(S, K, T, r, sig, q=0.0, kind="call")
put = bs_price(S, K, T, r, sig, q=0.0, kind="put")
check("Hull call == 4.759", abs(call - 4.759) < 1e-3, f"got {call:.6f}")
check("Hull put == 0.808", abs(put - 0.808) < 1e-3, f"got {put:.6f}")

# 2. Put-call parity C - P == S e^(-qT) - K e^(-rT), including q > 0
max_resid = 0.0
for (S_, K_, T_, r_, q_, sig_) in [
    (42.0, 40.0, 0.5, 0.10, 0.0, 0.20),
    (100.0, 110.0, 2.0, 0.03, 0.015, 0.35),
    (500.0, 480.0, 1.0, 0.05, 0.006, 0.28),
    (25.0, 30.0, 0.25, 0.04, 0.0, 0.55),
]:
    c = bs_price(S_, K_, T_, r_, sig_, q=q_, kind="call")
    p = bs_price(S_, K_, T_, r_, sig_, q=q_, kind="put")
    lhs = c - p
    rhs = S_ * math.exp(-q_ * T_) - K_ * math.exp(-r_ * T_)
    max_resid = max(max_resid, abs(lhs - rhs))
check("Put-call parity (machine precision)", max_resid < 1e-12, f"max residual {max_resid:.2e}")

# 3. Black-76 == BS with q = r (price, five Greeks), and delta == e^(-rT) N(d1)
max_pdiff = 0.0
max_gdiff = 0.0
max_ddiff = 0.0
for (F_, K_, T_, r_, sig_) in [
    (100.0, 95.0, 1.0, 0.04, 0.30),
    (15000.0, 14000.0, 2.0, 0.05, 0.25),
    (50.0, 55.0, 0.5, 0.02, 0.40),
]:
    for kind in ("call", "put"):
        p76 = black76_price(F_, K_, T_, r_, sig_, kind=kind)
        pbs = bs_price(F_, K_, T_, r_, sig_, q=r_, kind=kind)
        max_pdiff = max(max_pdiff, abs(p76 - pbs))
        g76 = black76_greeks(F_, K_, T_, r_, sig_, kind=kind)
        gbs = bs_greeks(F_, K_, T_, r_, sig_, q=r_, kind=kind)
        for key in ("delta", "gamma", "vega", "theta", "rho"):
            max_gdiff = max(max_gdiff, abs(g76[key] - gbs[key]))
    d1, _ = _d1_d2(F_, K_, T_, r_, r_, sig_)
    dc = black76_greeks(F_, K_, T_, r_, sig_, kind="call")["delta"]
    max_ddiff = max(max_ddiff, abs(dc - math.exp(-r_ * T_) * _norm_cdf(d1)))
check("Black-76 price == BS(q=r)", max_pdiff < 1e-12, f"max diff {max_pdiff:.2e}")
check("Black-76 Greeks == BS(q=r)", max_gdiff < 1e-12, f"max diff {max_gdiff:.2e}")
check("Black-76 call delta == e^(-rT) N(d1)", max_ddiff < 1e-12, f"max diff {max_ddiff:.2e}")

# 4. 0.95-delta strike solve round-trips its delta (constant and skewed sigma)
S0, T0, r0, q0 = 500.0, 1.5, 0.045, 0.006
Kc = solve_strike_for_delta(S0, T0, r0, 0.28, target_delta=0.95, q=q0, kind="call")
dc = bs_greeks(S0, Kc, T0, r0, 0.28, q=q0, kind="call")["delta"]
check("0.95-delta strike round-trip (flat vol)", abs(dc - 0.95) < 1e-8,
      f"K={Kc:.4f} delta={dc:.10f}")

skew = lambda K: 0.28 - 0.15 * math.log(K / S0)  # deeper ITM (lower K) carries higher IV
Ks = solve_strike_for_delta(S0, T0, r0, skew, target_delta=0.95, q=q0, kind="call")
ds = bs_greeks(S0, Ks, T0, r0, skew(Ks), q=q0, kind="call")["delta"]
check("0.95-delta strike round-trip (skewed vol)", abs(ds - 0.95) < 1e-8,
      f"K={Ks:.4f} delta={ds:.10f}")

# 5. Implied vol round-trip
true_sig = 0.3725
px = bs_price(S0, 480.0, T0, r0, true_sig, q=q0, kind="call")
iv = implied_vol(px, S0, 480.0, T0, r0, q=q0, kind="call")
check("Implied vol round-trip", abs(iv - true_sig) < 1e-6, f"recovered {iv:.8f}")

# 6. Analytic Greeks vs central finite differences
Sg, Kg, Tg, rg, qg, sg = 100.0, 105.0, 1.0, 0.03, 0.01, 0.30
g = bs_greeks(Sg, Kg, Tg, rg, sg, q=qg, kind="call")
h = 1e-4
delta_fd = (bs_price(Sg + h, Kg, Tg, rg, sg, q=qg) - bs_price(Sg - h, Kg, Tg, rg, sg, q=qg)) / (2 * h)
gamma_fd = (bs_price(Sg + h, Kg, Tg, rg, sg, q=qg) - 2 * bs_price(Sg, Kg, Tg, rg, sg, q=qg)
            + bs_price(Sg - h, Kg, Tg, rg, sg, q=qg)) / (h * h)
vega_fd = (bs_price(Sg, Kg, Tg, rg, sg + h, q=qg) - bs_price(Sg, Kg, Tg, rg, sg - h, q=qg)) / (2 * h)
rho_fd = (bs_price(Sg, Kg, Tg, rg + h, sg, q=qg) - bs_price(Sg, Kg, Tg, rg - h, sg, q=qg)) / (2 * h)
theta_fd = -(bs_price(Sg, Kg, Tg + h, rg, sg, q=qg) - bs_price(Sg, Kg, Tg - h, rg, sg, q=qg)) / (2 * h)
check("Delta vs finite diff", abs(g["delta"] - delta_fd) < 1e-6, f"{g['delta']:.8f} vs {delta_fd:.8f}")
check("Gamma vs finite diff", abs(g["gamma"] - gamma_fd) < 1e-4, f"{g['gamma']:.8f} vs {gamma_fd:.8f}")
check("Vega vs finite diff", abs(g["vega"] - vega_fd) < 1e-5, f"{g['vega']:.8f} vs {vega_fd:.8f}")
check("Theta vs finite diff", abs(g["theta"] - theta_fd) < 1e-5, f"{g['theta']:.8f} vs {theta_fd:.8f}")
check("Rho vs finite diff", abs(g["rho"] - rho_fd) < 1e-5, f"{g['rho']:.8f} vs {rho_fd:.8f}")

# 7. American (BAW): collapses to European when q<=0, dominates European and intrinsic,
#    and matches a 1000-step binomial for equity (b=r-q) and futures (b=0) carries
coll = 0.0
for (S_, K_, T_, r_, sig_) in [(100.0, 90.0, 1.0, 0.03, 0.25), (50.0, 55.0, 2.0, 0.05, 0.40)]:
    coll = max(coll, abs(baw_call(S_, K_, T_, r_, sig_, q=0.0)
                         - bs_price(S_, K_, T_, r_, sig_, q=0.0, kind="call")))
check("BAW == European when q=0", coll < 1e-12, f"max dev {coll:.2e}")

dom_ok, intr_ok, max_rel = True, True, 0.0
baw_cases = [
    (100.0, 70.0, 1.0, 0.001, 0.02, 0.25),   # equity deep-ITM, q>r
    (100.0, 60.0, 2.0, 0.001, 0.02, 0.31),   # equity deep-ITM 2y
    (100.0, 100.0, 1.0, 0.010, 0.03, 0.30),  # equity near-ATM
    (100.0, 70.0, 1.0, 0.030, 0.03, 0.30),   # futures carry b=0 (q=r)
]
for (S_, K_, T_, r_, q_, sig_) in baw_cases:
    ba = baw_call(S_, K_, T_, r_, sig_, q=q_)
    eu = bs_price(S_, K_, T_, r_, sig_, q=q_, kind="call")
    bi = _binom_call(S_, K_, T_, r_, q_, sig_, american=True)
    dom_ok = dom_ok and ba >= eu - 1e-9 and ba >= max(S_ - K_, 0.0) - 1e-9
    max_rel = max(max_rel, abs(ba - bi) / bi)
check("BAW >= European and >= intrinsic", dom_ok)
check("BAW matches binomial within 0.5%", max_rel < 5e-3, f"max rel {max_rel*100:.3f}%")

print(f"\n{sum(_results)}/{len(_results)} checks passed")
sys.exit(0 if all(_results) else 1)

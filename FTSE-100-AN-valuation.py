# Valuation of a GBP capital-protected FTSE 100 Asian note

"""
Product
-------
A GBP 100 note matures in five years and pays

    Payoff_T = N * [1 + p * max(A_T / S0 - 1, 0)],

where A_T is the arithmetic mean of the twelve monthly FTSE 100 closes in its final year, S0 is the initial FTSE 100 level, N is GBP 100 notional, 
and p is the participation rate. The initial investment is protected at maturity, only the upside linked to the final-year average is uncertain.

The note is valued as a zero-coupon bond plus an arithmetic Asian call:

    V0 = N * exp(-rT) + p * (N / S0) * C_arithmetic.

The fair participation rate that makes V0 = N follows directly:

    p = S0 * [1 - exp(-rT)] / C_arithmetic.

Simulation framework
--------------------
Under the risk-neutral measure, the FTSE 100 price index follows constant-parameter geometric Brownian motion:

    dS_t = (r - q) * S_t * dt + sigma * S_t * dW_t.

Consequently an exact interval update is

    S_(t+dt) = S_t * exp[(r-q-0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z].

The arithmetic average has no closed-form Black-Scholes price. The code therefore uses Monte Carlo. It validates the engine and reduces variance by 
using the corresponding geometric Asian call as a control variate. The log of the geometric mean is normal, so its discrete-monitoring price is available
analytically. Antithetic variates and an independently estimated control-variate coefficient make the calculation accurate while remaining fast.

Market inputs (at 5 November 2026 market close)
---------------------------------------------------------------
- S0 = 10,782.66:   FTSE 100 Index, ticker FTSE:FSI, GBP close.
- r = 4.51%:        GBP five-year SONIA/OIS swap-rate proxy.
- q = 3.07%:        FTSE 100 trailing dividend-yield proxy.
- sigma = 11.86%:   FTSE 100 one-year realised annualised volatility proxy.

The index is treated as a GBP price-return index, hence q enters the risk-neutral carry. The rate is a flat zero-rate approximation to the GBP OIS
curve. The available sigma is realised rather than five-year implied volatility; a desk implementation should replace q and sigma with FTSE 100
index dividend forwards and the matching implied-volatility surface.

"""

import time
import numpy as np
from scipy.stats import norm

t_start = time.time()
rng = np.random.default_rng(1234)


### Product terms and market data ###

S0 = 10782.66 
r = 0.0451 
q = 0.0307
sigma = 0.1186    

Notional = 100.0
T = 5.0
n_avg = 12
window_start = T - 1.0
t_i = window_start + np.arange(1, n_avg + 1) / 12.0
K = S0

conv_debug = True
N_PAIRS = 500_000
N_PILOT = 50_000
assert 0 < N_PILOT < N_PAIRS

print(f"FTSE:FSI S0={S0:,.2f}; r={r:.2%}; q={q:.2%}; sigma={sigma:.2%}; "
      f"T={T:.0f}y; {n_avg} monthly averaging dates")

### Exact geometric-average benchmark ###

def lognormal_call_price(spot, strike, rate, dividend_yield, vol, dates):
    """Price a call on a discrete geometric average; one date gives BS call."""
    dates = np.atleast_1d(np.asarray(dates, dtype=float))
    n_dates = len(dates)
    payment_time = dates[-1]
    variance = (vol ** 2 / n_dates ** 2) * np.minimum.outer(dates, dates).sum()
    mean_log_g = (np.log(spot) + (rate - dividend_yield - 0.5 * vol ** 2) * dates.mean())
    d1 = (mean_log_g + variance - np.log(strike)) / np.sqrt(variance)
    d2 = d1 - np.sqrt(variance)

    expected_call = (np.exp(mean_log_g + 0.5 * variance) * norm.cdf(d1) - strike * norm.cdf(d2))
    return np.exp(-rate * payment_time) * expected_call


vanilla_bs_price = lognormal_call_price(S0, K, r, q, sigma, np.array([T]))
geom_exact_price = lognormal_call_price(S0, K, r, q, sigma, t_i)

print("\n--- Closed-form benchmarks ---")
print(f"Vanilla European call (terminal FTSE 100) : {vanilla_bs_price:10.2f}")
print(f"Geometric Asian call (12 discrete dates)  : {geom_exact_price:10.2f}")

### Exact GBM Monte Carlo and control variate ###

def simulate_tail_path(spot, vol, shocks):
    """Return final-year monthly levels using exact GBM transitions."""
    # No payoff depends on years 0-4, so one exact draw replaces 48 monthly steps.
    level_at_window = spot * np.exp(
        (r - q - 0.5 * vol ** 2) * window_start
        + vol * np.sqrt(window_start) * shocks[:, 0]
    )

    # Generate the twelve fixing levels used by both arithmetic and geometric legs.
    dt = 1.0 / 12.0
    log_returns = ((r - q - 0.5 * vol ** 2) * dt
                   + vol * np.sqrt(dt) * shocks[:, 1:])
    return np.exp(np.log(level_at_window)[:, None] + np.cumsum(log_returns, axis=1))


def discounted_payoffs(paths):
    """Return discounted arithmetic and geometric call payoffs, plus A_T."""
    arithmetic_average = paths.mean(axis=1)
    geometric_average = np.exp(np.log(paths).mean(axis=1))
    discount = np.exp(-r * T)
    arithmetic_call = discount * np.maximum(arithmetic_average - K, 0.0)
    geometric_call = discount * np.maximum(geometric_average - K, 0.0)
    return arithmetic_call, geometric_call, arithmetic_average


Z = rng.standard_normal((N_PAIRS, 1 + n_avg))
paths_pos = simulate_tail_path(S0, sigma, Z)
paths_neg = simulate_tail_path(S0, sigma, -Z)

arith_pos, geom_pos, avg_pos = discounted_payoffs(paths_pos)
arith_neg, geom_neg, avg_neg = discounted_payoffs(paths_neg)
arith_pair = 0.5 * (arith_pos + arith_neg)
geom_pair = 0.5 * (geom_pos + geom_neg)
avg_pair = 0.5 * (avg_pos + avg_neg)

# Check the simulated geometric payoff against its exact discrete-monitoring price.
geom_mc = geom_pair.mean()
se_geom = geom_pair.std(ddof=1) / np.sqrt(N_PAIRS)

# Estimate beta in a pilot sample, then keep it fixed in the pricing sample.
pilot_arith, main_arith = arith_pair[:N_PILOT], arith_pair[N_PILOT:]
pilot_geom, main_geom = geom_pair[:N_PILOT], geom_pair[N_PILOT:]
N_MAIN = len(main_arith)
beta_star = np.cov(pilot_arith, pilot_geom, ddof=1)[0, 1] / np.var(pilot_geom, ddof=1)

arith_mc_raw = main_arith.mean()
se_raw = main_arith.std(ddof=1) / np.sqrt(N_MAIN)
cv_sample = main_arith - beta_star * (main_geom - geom_exact_price)
arith_mc_cv = cv_sample.mean()
se_cv = cv_sample.std(ddof=1) / np.sqrt(N_MAIN)

print("\n--- Monte Carlo (1,000,000 simulated paths) ---")
print(f"Geometric Asian call (MC)                 : {geom_mc:10.2f} "
      f"+/- {1.96 * se_geom:.2f} (95% CI)")
print(f"Arithmetic Asian call (control variate)   : {arith_mc_cv:10.2f} "
      f"+/- {1.96 * se_cv:.2f} (95% CI); beta={beta_star:.2f}")

# G <= A pathwise, hence a geometric Asian call cannot exceed its arithmetic peer.
assert abs(geom_mc - geom_exact_price) < 5 * se_geom, "Geometric MC validation failed"
assert arith_mc_cv > geom_exact_price, "Arithmetic Asian should exceed geometric Asian"
print("Validation passes: geometric MC matches exact price within CI, arithmetic > geometric")

### Fair participation and risk-neutral expected payoff ###

def note_price(participation, asian_call_price):
    """PV = protected zero-coupon bond plus the participating Asian call."""
    bond_leg = Notional * np.exp(-r * T)
    option_leg = participation * (Notional / S0) * asian_call_price
    return bond_leg + option_leg


if arith_mc_cv <= 0:
    raise ValueError("The Asian call estimate must be positive.")

# Solve V0=N analytically because the note PV is linear in participation.
fair_p = S0 * (1.0 - np.exp(-r * T)) / arith_mc_cv
bond_pv = Notional * np.exp(-r * T)

expected_call_undiscounted = arith_mc_cv * np.exp(r * T)
expected_payoff_q = Notional + fair_p * (Notional / S0) * expected_call_undiscounted
prob_itm = np.mean(0.5 * ((avg_pos[N_PILOT:] > K).astype(float) + (avg_neg[N_PILOT:] > K).astype(float)))

print("\n--- Par participation calibration ---")
print(f"Protected bond PV                         : {bond_pv:10.2f}")
print(f"Fair participation rate                   : {fair_p:10.2%}")
print(f"Risk-neutral E^Q[payoff at T]             : {expected_payoff_q:10.2f}")
print(f"Discounted E^Q[payoff]                    : {expected_payoff_q * np.exp(-r * T):10.2f}") # expected to match notional PV (= 100)
print(f"P(A_T > S0)                               : {prob_itm:10.2%}")

### Convergence diagnostics (comparing raw and control-variate MC estimators) ###

if conv_debug:
    print("\n--- Convergence: raw versus control-variate estimator ---")
    print(f"{'pairs':>10} | {'raw SE':>10} | {'CV SE':>10} | {'reduction':>12}")
    for n_pairs in [2_000, 10_000, 50_000, 200_000, N_MAIN]:
        raw_se = main_arith[:n_pairs].std(ddof=1) / np.sqrt(n_pairs)
        cv_n = main_arith[:n_pairs] - beta_star * (main_geom[:n_pairs] - geom_exact_price)
        cv_se = cv_n.std(ddof=1) / np.sqrt(n_pairs)
        print(f"{n_pairs:>10,} | {raw_se:>10.2f} | {cv_se:>10.2f} | "
            f"{(raw_se / cv_se) ** 2:>11,.0f}x")

### Greeks of the issued contract ###

def reprice_cv(spot, vol, participation):
    """Reprice using common shocks; K and participation remain contractual."""
    positive = simulate_tail_path(spot, vol, Z)
    negative = simulate_tail_path(spot, vol, -Z)
    arith_p, geom_p, _ = discounted_payoffs(positive)
    arith_n, geom_n, _ = discounted_payoffs(negative)
    arith = 0.5 * (arith_p + arith_n)
    geom = 0.5 * (geom_p + geom_n)
    geom_exact = lognormal_call_price(spot, K, r, q, vol, t_i)
    asian_call = (arith - beta_star * (geom - geom_exact)).mean()
    return note_price(participation, asian_call)


# Common random numbers ensure bump differences reflect market sensitivity, not noise.
base_price = reprice_cv(S0, sigma, fair_p)
d_spot = 0.01 * S0
spot_up = reprice_cv(S0 + d_spot, sigma, fair_p)
spot_down = reprice_cv(S0 - d_spot, sigma, fair_p)
delta = (spot_up - spot_down) / (2.0 * d_spot)

d_vol = 0.01
vol_up = reprice_cv(S0, sigma + d_vol, fair_p)
vol_down = reprice_cv(S0, sigma - d_vol, fair_p)
vega = (vol_up - vol_down) / (2.0 * d_vol) / 100.0

print("\n--- Greeks: participation and strike fixed after issuance ---")
print(f"Delta (PV change per one index point)    : {delta:10.4f}")
print(f"Vega (PV change per +1 volatility point) : {vega:10.4f}")

print("\n" + "-" * 125)
print("SUMMARY")
print("-" * 125)
print(f"Structure: GBP 100 protected principal + {fair_p:.2%} participation in a final-year, 12-month arithmetic FTSE 100 average call")
print(f"Par issue price / risk-neutral PV: {note_price(fair_p, arith_mc_cv):.2f}" + " (expected to match notional PV of 100)")
print(f"Risk-neutral expected maturity payoff: {expected_payoff_q:.2f}")
print("-" * 125)

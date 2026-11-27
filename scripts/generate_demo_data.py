"""Generate the clearly-labelled SYNTHETIC demo dataset (not market quotes).

Monthly total-return price indices in PLN and monthly income (dividend/coupon)
per unit, simulated from a multivariate Student-t with fixed, plausible
parameters. Deterministic: fixed seed.

    python scripts/generate_demo_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 11
OUT = Path(__file__).resolve().parents[1] / "data" / "demo"
ASSETS = ["SYN_GLOBAL", "SYN_US", "SYN_EUROPE", "SYN_EM", "SYN_BONDS", "SYN_GOLD", "SYN_CASH"]
MU = np.array([0.075, 0.085, 0.065, 0.080, 0.040, 0.055, 0.032])  # annual arithmetic, PLN
VOL = np.array([0.150, 0.165, 0.175, 0.215, 0.055, 0.160, 0.008])
YIELD = np.array([0.018, 0.015, 0.028, 0.026, 0.035, 0.0, 0.032])  # income yield
CORR = np.array([
    [1.00, 0.93, 0.88, 0.78, 0.05, 0.10, 0.00],
    [0.93, 1.00, 0.78, 0.68, 0.05, 0.05, 0.00],
    [0.88, 0.78, 1.00, 0.72, 0.08, 0.10, 0.00],
    [0.78, 0.68, 0.72, 1.00, 0.05, 0.20, 0.00],
    [0.05, 0.05, 0.08, 0.05, 1.00, 0.25, 0.10],
    [0.10, 0.05, 0.10, 0.20, 0.25, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 1.00],
])
DOF = 6


def main() -> None:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2000-01-31", "2025-12-31", freq="ME")
    n = len(idx)
    cov_m = np.outer(VOL, VOL) * CORR / 12
    mean_log = np.log1p(MU / 12) - 0.5 * np.diag(cov_m)
    chol = np.linalg.cholesky(cov_m)
    z = rng.standard_normal((n, len(ASSETS))) @ chol.T
    g = rng.chisquare(DOF, size=(n, 1)) / DOF
    x = mean_log + z * np.sqrt((DOF - 2) / DOF) / np.sqrt(g)
    total = np.expm1(x)
    prices = 100 * np.cumprod(1 + total, axis=0)
    prices = np.vstack([np.full(len(ASSETS), 100.0), prices])
    dates = idx.insert(0, pd.Timestamp("1999-12-31"))
    price_df = pd.DataFrame(prices, index=dates, columns=ASSETS).round(6)
    # income paid in month t = yield/12 * previous total-return price (already included in the index)
    income = pd.DataFrame(price_df.shift(1).to_numpy() * YIELD / 12, index=dates, columns=ASSETS).iloc[1:].round(6)
    price_df.index.name = income.index.name = "date"
    OUT.mkdir(parents=True, exist_ok=True)
    price_df.to_csv(OUT / "synthetic_monthly_prices.csv")
    income.to_csv(OUT / "synthetic_monthly_income.csv")
    print(f"Zapisano {len(price_df)} miesięcy syntetycznych danych do {OUT}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def toy_returns() -> pd.DataFrame:
    """Deterministic synthetic monthly returns of 4 assets (test-only data)."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2005-01-31", periods=180, freq="ME")
    mu = np.array([0.007, 0.006, 0.003, 0.004])
    vol = np.array([0.045, 0.05, 0.012, 0.04])
    corr = np.array([[1, 0.8, 0.1, 0.0], [0.8, 1, 0.15, 0.05], [0.1, 0.15, 1, 0.2], [0.0, 0.05, 0.2, 1]])
    cov = np.outer(vol, vol) * corr
    x = rng.multivariate_normal(mu, cov, size=len(idx))
    return pd.DataFrame(x, index=idx, columns=["EQ1", "EQ2", "BOND", "GOLD"])


@pytest.fixture
def toy_classes() -> dict:
    return {"EQ1": "global_equity", "EQ2": "us_equity", "BOND": "bonds", "GOLD": "gold"}

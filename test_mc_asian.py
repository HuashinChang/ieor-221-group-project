from itertools import combinations
from math import hypot

import numpy as np
import pytest
import torch
from scipy.stats import norm

from mc_asian import MCPricer


TOTAL_PATHS = 2**24
VARIANCE_REDUCTION = (
    ("plain", False, None),
    ("antithetic", True, None),
    ("stock control", False, "stock_prices"),
    ("geometric Asian control", False, "geom_asian"),
    ("antithetic + stock control", True, "stock_prices"),
    ("antithetic + geometric Asian control", True, "geom_asian"),
)
PRICE_TOLERANCE = 0.1  # Absolute price difference for a spot of 100.
CI_MULTIPLIER = norm.ppf(0.995)  # Two-sided 99% confidence interval.
MAX_CI_HALF_WIDTH_FRACTION = 0.5


@pytest.mark.parametrize(
    ("strike", "dividend", "rate"),
    [
        pytest.param(90, 0.02, 0.03, id="ITM"),
        pytest.param(100, 0.08, 0.05, id="ATM"),
        pytest.param(110, 0.04, 0.02, id="OTM"),
    ],
)
@pytest.mark.parametrize(
    "dtm",
    [10, 63, 125, 250, 500]
)
@pytest.mark.parametrize(
    "vol",
    [0.01, 0.1, 0.2, 0.4]
)
@pytest.mark.parametrize(
    "root_seed",
    [0, 12321, 45343]
)
def test_mc_policy_prices_are_equivalent(strike, dtm, vol, dividend, rate, root_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    # Distinct seeds let us combine the independent price standard errors.
    for index, (policy, antithetic, ctrl_var) in enumerate(VARIANCE_REDUCTION):
        # Match total paths; one antithetic sample is a pair of paths.
        n_samples = TOTAL_PATHS // 2 if antithetic else TOTAL_PATHS
        pricer = MCPricer(
            stock_spot=100,
            dividend=dividend,
            vol=vol,
            rate=rate,
            n_samples=n_samples,
            seed=root_seed + 3141 * index,
            device=device,
        )
        results[policy] = pricer.run_mc(
            strike, dtm, antithetic=antithetic, ctrl_var=ctrl_var
        )

    for left, right in combinations(results, 2):
        left_price, left_se = results[left]
        right_price, right_se = results[right]
        difference = left_price - right_price
        half_width = CI_MULTIPLIER * hypot(left_se, right_se)
        comparison = (
            f"{left} vs {right}: price difference={difference:+.5f}, "
            f"99% CI half-width={half_width:.5f}, tolerance={PRICE_TOLERANCE:.5f}"
        )
        assert abs(difference) + half_width < PRICE_TOLERANCE, comparison
        assert half_width < MAX_CI_HALF_WIDTH_FRACTION * PRICE_TOLERANCE, comparison

from itertools import combinations
from math import hypot

import numpy as np
import pytest
import torch
from scipy.stats import norm

from mc_asion import MCPricer


TOTAL_PATHS = 2**19
VARIANCE_REDUCTION = (
    ("plain", False, None),
    ("antithetic", True, None),
    ("stock control", False, "stock_prices"),
    ("geometric Asian control", False, "geom_asian"),
    ("antithetic + stock control", True, "stock_prices"),
    ("antithetic + geometric Asian control", True, "geom_asian"),
)
PRICE_TOLERANCE = 0.25  # Absolute price difference for a spot of 100.
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


@pytest.mark.parametrize(
    ("policy", "antithetic", "ctrl_var"),
    VARIANCE_REDUCTION,
    ids=[policy for policy, _, _ in VARIANCE_REDUCTION],
)
def test_mc_mean_and_standard_error_use_independent_samples(
    policy, antithetic, ctrl_var, monkeypatch
):
    # Ten total paths become five independent antithetic pairs.
    shocks = np.array(
        [[-1.8, -0.7, 0.1, 0.9, 1.5], [0.2, -1.1, 1.3, -0.5, 0.6]]
    )
    total_paths = 2 * shocks.shape[1]
    n_samples = total_paths // 2 if antithetic else total_paths
    if not antithetic:
        shocks = np.concatenate((shocks, -shocks), axis=1)
    draws = iter(shocks)
    draw_sizes = []

    def fixed_randn(size, *, device, dtype, generator):
        draw_sizes.append(size)
        return torch.tensor(next(draws), device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fixed_randn)
    pricer = MCPricer(
        stock_spot=100,
        dividend=0.08,
        vol=0.35,
        rate=0.05,
        n_samples=n_samples,
        seed=42,
        device=torch.device("cpu"),
    )
    strike = 100
    dtm = len(shocks)
    price, standard_error = pricer.run_mc(
        strike, dtm, antithetic=antithetic, ctrl_var=ctrl_var
    )

    path_shocks = np.concatenate((shocks, -shocks), axis=1) if antithetic else shocks
    drift = (pricer.rate - pricer.dividend - pricer.vol**2 / 2) / pricer.day_count
    increments = drift + pricer.vol * path_shocks / np.sqrt(pricer.day_count)
    paths = pricer.stock_spot * np.exp(np.cumsum(increments, axis=0))
    assert paths.shape == (dtm, total_paths)
    discount = np.exp(-pricer.rate * dtm / pricer.day_count)
    payoffs = discount * np.maximum(paths.mean(axis=0) - strike, 0)

    if ctrl_var is not None:
        if ctrl_var == "stock_prices":
            control = paths[-1]
            control_mean = pricer.stock_spot * np.exp(
                (pricer.rate - pricer.dividend) * dtm / pricer.day_count
            )
        else:
            control = discount * np.maximum(
                np.exp(np.log(paths).mean(axis=0)) - strike, 0
            )
            # Independent discrete-monitoring formula from the observation times.
            times = np.arange(1, dtm + 1) / pricer.day_count
            log_mean = np.log(pricer.stock_spot) + (
                pricer.rate - pricer.dividend - pricer.vol**2 / 2
            ) * times.mean()
            log_variance = pricer.vol**2 * np.minimum.outer(times, times).mean()
            log_std = np.sqrt(log_variance)
            d2 = (log_mean - np.log(strike)) / log_std
            control_mean = discount * (
                np.exp(log_mean + log_variance / 2) * norm.cdf(d2 + log_std)
                - strike * norm.cdf(d2)
            )
        centered_control = control - control.mean()
        beta = np.dot(payoffs - payoffs.mean(), centered_control) / np.dot(
            centered_control, centered_control
        )
        payoffs = payoffs - beta * (control - control_mean)

    if antithetic:
        payoffs = (payoffs[:n_samples] + payoffs[n_samples:]) / 2

    assert draw_sizes == [n_samples] * dtm
    assert price == pytest.approx(payoffs.mean(), rel=1e-10, abs=1e-12)
    assert standard_error == pytest.approx(
        payoffs.std(ddof=1) / np.sqrt(n_samples), rel=1e-10, abs=1e-12
    )

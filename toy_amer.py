import numpy as np
from scipy.stats import norm
import pytest


def CRR_american_call(
    *,
    stock_spot: float,
    vol: float,
    ttm: float,
    zero_rate: float,
    dividend: float,
    strike: float,
    n_steps: int,
) -> float:
    dt = ttm / n_steps

    up_factor = np.exp(vol * np.sqrt(dt))
    down_factor = 1 / up_factor

    up_prob = ((np.exp((zero_rate - dividend) * dt) - down_factor) 
        / (up_factor - down_factor))
    down_prob = 1 - up_prob

    stock_prices = _make_ternimal_stock_prices(stock_spot, up_factor, n_steps)
    amer_prices = np.maximum(stock_prices - strike, 0)
    
    for _ in  range(n_steps):
        expect_vals = amer_prices[:-1] * down_prob + amer_prices[1:] * up_prob
        stock_prices = stock_prices[:-1] * up_factor
        exercise_payoffs = np.maximum(stock_prices - strike, 0)
        amer_prices = np.maximum(
            exercise_payoffs, 
            np.exp(-zero_rate * dt) * expect_vals)

    return amer_prices[0]


def _make_ternimal_stock_prices(
    stock_spot: float,
    up_factor: float,
    n_steps: int
) -> np.ndarray:
    """result[i] = u^i * d^{n-i} * S0"""
    down_factor = 1 / up_factor
    up_pows = np.arange(n_steps + 1)
    down_pows = n_steps - up_pows
    return stock_spot * np.power(up_factor, up_pows) * np.power(down_factor, down_pows)


def bsm_call_price(
    stock_spot: float,
    strike: float,
    vol: float,
    zero_rate: float,
    ttm: float,
    dividend: float,
) -> float:
    d1 = (
        np.log(stock_spot / strike) 
        + (zero_rate - dividend + 0.5 * vol**2) * ttm) / (vol * np.sqrt(ttm))
    d2 = d1 - vol * np.sqrt(ttm)
    return (
        np.exp(-dividend * ttm) * stock_spot * norm.cdf(d1) 
        - np.exp(-zero_rate * ttm) * strike * norm.cdf(d2)
    )


@pytest.mark.parametrize(
        "vol",
        [0.01, 0.1, 0.2, 0.3, 0.5]
)
@pytest.mark.parametrize(
    "zero_rate",
    [0.01, 0.05, 0.1]
)
@pytest.mark.parametrize(
    "strike",
    [90, 100, 110],
)
@pytest.mark.parametrize(
    "ttm",
    [0.1, 1, 5]
)
def test_same_as_euro_when_no_dividend(vol, zero_rate, strike, ttm):
    STOCK_SPOT = 100.0
    amer_price = CRR_american_call(
        stock_spot=STOCK_SPOT,
        vol=vol,
        zero_rate=zero_rate,
        strike=strike,
        ttm=ttm,
        dividend=0.0,
        n_steps=10000,
        )
    euro_price = bsm_call_price(
        stock_spot=STOCK_SPOT,
        ttm=ttm,
        vol=vol,
        strike=strike,
        zero_rate=zero_rate,
        dividend=0.0
    )
    assert amer_price == pytest.approx(euro_price, rel=1e-4, abs=1e-3)

if __name__ == "__main__":
    pytest.main([__file__, "-q"])


    
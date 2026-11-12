import numpy as np
from scipy.stats import norm
from typing import Literal, Callable


def crr_american_call(
    *,
    stock_spot: float,
    vol: float,
    ttm: float,
    rate: float,
    dividend: float,
    strike: float,
    n_steps: int,
    smoothing: bool = False,
) -> tuple[float, float, float, float]:
    """
    Returns:
        A tuple (`npv`, `delta`, `gamma`, `theta`).
    """
    dt = ttm / n_steps

    up_factor = np.exp(vol * np.sqrt(dt))
    down_factor = 1 / up_factor

    up_prob = ((np.exp((rate - dividend) * dt) - down_factor) 
        / (up_factor - down_factor))
    down_prob = 1 - up_prob

    stock_prices = _make_ternimal_stock_prices(stock_spot, up_factor, n_steps)
    amer_prices = np.maximum(stock_prices - strike, 0)

    if smoothing:
        stock_prices = stock_prices[:-1] * up_factor
        bsm_vanilla = bsm_call_price(
            stock_spot=stock_prices,
            strike=strike,
            vol=vol,
            rate=rate,
            ttm=dt,
            dividend=dividend
        )
        exercise_payoffs = np.maximum(stock_prices - strike, 0)
        amer_prices = np.maximum(bsm_vanilla, exercise_payoffs)
        start = n_steps - 2
    else:
        start = n_steps - 1
    
    for i in range(start, -1, -1):
        expect_vals = amer_prices[:-1] * down_prob + amer_prices[1:] * up_prob
        stock_prices = stock_prices[:-1] * up_factor
        exercise_payoffs = np.maximum(stock_prices - strike, 0)
        amer_prices = np.maximum(
            exercise_payoffs, 
            np.exp(-rate * dt) * expect_vals)

        if i == 2:
            payoffs_2 = amer_prices
            stocks_2 = stock_prices
        if i == 1:
            payoffs_1 = amer_prices
            stocks_1 = stock_prices

    delta = (payoffs_1[1] - payoffs_1[0]) / (stocks_1[1] - stocks_1[0])
    deltas = (payoffs_2[1:] - payoffs_2[:-1]) / (stocks_2[1:] - stocks_2[:-1])
    gamma = (deltas[1] - deltas[0]) / (stocks_1[1] - stocks_1[0])
    theta = (payoffs_2[1] - amer_prices[0]) / (2 * dt)

    return amer_prices.item(), delta.item(), gamma.item(), theta.item()

def bbsr_american_call(
    *,
    stock_spot: float,
    vol: float,
    ttm: float,
    rate: float,
    dividend: float,
    strike: float,
    n_steps: int,
):
    inputs = {
        'stock_spot': stock_spot,
        'vol': vol,
        'ttm': ttm,
        'rate': rate,
        'dividend': dividend,
        'strike': strike
    }
    res_n = crr_american_call(**inputs, n_steps=n_steps, smoothing=True)
    res_2n = crr_american_call(**inputs, n_steps=int(2*n_steps), smoothing=True)
    return (res_2n[i] * 2 - res_n[i] for i in range(len(res_n)))


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
    stock_spot: float | np.ndarray,
    strike: float,
    vol: float,
    rate: float,
    ttm: float,
    dividend: float,
) -> float | np.ndarray:
    d1 = (
        np.log(stock_spot / strike) 
        + (rate - dividend + 0.5 * vol**2) * ttm) / (vol * np.sqrt(ttm))
    d2 = d1 - vol * np.sqrt(ttm)
    return (
        np.exp(-dividend * ttm) * stock_spot * norm.cdf(d1) 
        - np.exp(-rate * ttm) * strike * norm.cdf(d2)
    )

def cent_fdm(
    *,
    stock_spot: float,
    vol: float,
    ttm: float,
    rate: float,
    dividend: float,
    strike: float,
    n_steps: int,
    bump_size: float = 1e-4,
    bump_param: Literal['vol', 'rate'],
    tree_method: Literal['CRR', 'BBSR'] = 'CRR',
):
    inputs = {
        'stock_spot': stock_spot,
        'vol': vol,
        'ttm': ttm,
        'rate': rate,
        'strike': strike,
        'dividend': dividend,
        'n_steps': n_steps
    }

    pricer = select_pricer(tree_method)

    inputs[bump_param] += bump_size
    up_price, _, _, _ = pricer(**inputs)

    inputs[bump_param] -= 2 * bump_size
    down_price, _, _, _ = pricer(**inputs)

    return (up_price - down_price) / (2 * bump_size)


def select_pricer(method: Literal['CRR', 'BBSR']) -> Callable[..., tuple[float, float, float, float]]:
    if method == 'BBSR':
        return bbsr_american_call
    if method == 'CRR':
        return crr_american_call
    raise TypeError('Invalid input string.')
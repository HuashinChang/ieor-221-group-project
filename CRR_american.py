import numpy as np
from scipy.stats import norm
from typing import Literal


def CRR_american_call(
    *,
    stock_spot: float,
    vol: float,
    ttm: float,
    rate: float,
    dividend: float,
    strike: float,
    n_steps: int,
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
    
    for i in range(n_steps -1, -1, -1):
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
    rate: float,
    ttm: float,
    dividend: float,
) -> float:
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
    bump_size: float = 1e-3,
    bump_param: Literal['vol', 'rate']
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

    inputs[bump_param] += bump_size
    up_price, _, _, _ = CRR_american_call(**inputs)

    inputs[bump_param] -= 2 * bump_size
    down_price, _, _, _ = CRR_american_call(**inputs)

    return (up_price - down_price) / (2 * bump_size)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', type=int, default=10_000)
    parser.add_argument('--rate-bump-size', type=float, default=1e-3)
    parser.add_argument('--vol-bump-size', type=float, default=1e-3)
    arg = parser.parse_args()

    inputs = {
        'stock_spot': 100.0,
        'vol': 0.35,
        'ttm': 5,
        'rate': 0.5,
        'strike': 100.0,
        'dividend': 0.08,
        'n_steps': arg.n_steps
    }
    amer_price, delta, gamma, theta = CRR_american_call(**inputs)
    vega = cent_fdm(**inputs, bump_size=arg.vol_bump_size, bump_param='vol')
    rho = cent_fdm(**inputs, bump_size=arg.rate_bump_size, bump_param='rate')

    print(f'NPV: {amer_price:.4f}')
    print(f'Delta: {delta:.4f}')
    print(f'Gamma: {gamma:.4f}')
    print(f'Theta: {theta:.4f}')
    print(f'Vega: {vega:.4f}')
    print(f'Rho: {rho:.4f}')

    inputs.pop('n_steps')
    vanilla_price = bsm_call_price(**inputs)

    print("For reference, the price of the corresponding" 
          + f"vanilla call option is {vanilla_price:.4f}")


if __name__ == "__main__":
    main()


    
import numpy as np
from scipy.stats import norm
from typing import Literal, Callable


def CRR_american_call(
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

def BBSR_american_call(
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
    res_n = CRR_american_call(**inputs, n_steps=n_steps, smoothing=True)
    res_2n = CRR_american_call(**inputs, n_steps=int(2*n_steps), smoothing=True)
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

    pricer = _select_pricer(tree_method)

    inputs[bump_param] += bump_size
    up_price, _, _, _ = pricer(**inputs)

    inputs[bump_param] -= 2 * bump_size
    down_price, _, _, _ = pricer(**inputs)

    return (up_price - down_price) / (2 * bump_size)


def _select_pricer(method: Literal['CRR', 'BBSR']) -> Callable[..., tuple[float, float, float, float]]:
    if method == 'BBSR':
        return BBSR_american_call
    if method == 'CRR':
        return CRR_american_call
    raise TypeError('Invalid input string.')


def _export_convergence(rows, method, output_dir):
    import csv

    columns = ("N steps", "NPV", "Delta", "Gamma", "Theta", "Vega", "Rho")
    formatted_rows = [
        [str(n_steps), *(f"{value:.5f}" for value in values)]
        for n_steps, *values in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{method.lower()}_convergence.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(formatted_rows)

    print(" ".join(f"{column:>12}" for column in columns))
    for row in formatted_rows:
        print(" ".join(f"{value:>12}" for value in row))


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Calculate price and Greeks at doubling tree step counts; save CSV."
    )
    parser.add_argument(
        "--start-n-steps", type=int, default=2**8,
        help="starting step count (default: 256); multiply by 2 up to 32768",
    )
    parser.add_argument("--rate-bump-size", type=float, default=1e-4)
    parser.add_argument("--vol-bump-size", type=float, default=1e-4)
    parser.add_argument("--method", choices=["CRR", "BBSR"], default="CRR")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("plots")
    )
    args = parser.parse_args()
    max_n_steps = 32768
    if not 4 <= args.start_n_steps <= max_n_steps:
        parser.error(f"--start-n-steps must be between 4 and {max_n_steps}")

    step_counts = []
    n_steps = args.start_n_steps
    while n_steps < max_n_steps:
        step_counts.append(n_steps)
        n_steps *= 2
    step_counts.append(max_n_steps)

    inputs = {
        "stock_spot": 100.0,
        "vol": 0.35,
        "ttm": 5,
        "rate": 0.05,
        "strike": 100.0,
        "dividend": 0.08,
    }
    pricer = _select_pricer(args.method)
    rows = []
    for n_steps in step_counts:
        print(f"Calculating {args.method} at N={n_steps:,}...", flush=True)
        price, delta, gamma, theta = pricer(**inputs, n_steps=n_steps)
        vega = cent_fdm(
            **inputs, n_steps=n_steps, bump_size=args.vol_bump_size,
            bump_param="vol", tree_method=args.method,
        )
        rho = cent_fdm(
            **inputs, n_steps=n_steps, bump_size=args.rate_bump_size,
            bump_param="rate", tree_method=args.method,
        )
        rows.append((n_steps, price, delta, gamma, theta, vega, rho))
        
    print("")
    _export_convergence(rows, args.method, args.output_dir)
    vanilla_price = bsm_call_price(**inputs)
    print(f"\nCorresponding European call price: {vanilla_price:.5f}")


if __name__ == "__main__":
    main()

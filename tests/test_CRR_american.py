import pytest
import math
from scipy.stats import norm
from CRR_american import CRR_american_call, cent_fdm

def bsm_prices_greeks(stock_spot, vol, rate, ttm, strike, dividend):
    d1 = (
        math.log(stock_spot / strike) 
        + (rate - dividend + 0.5 * vol**2) * ttm) / (vol * math.sqrt(ttm))
    d2 = d1 - vol * math.sqrt(ttm)
    dvd_disc = math.exp(-dividend * ttm)
    disc = math.exp(-rate * ttm)

    price = stock_spot * dvd_disc * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    delta = dvd_disc * norm.cdf(d1)
    gamma = dvd_disc * norm.pdf(d1) / (stock_spot * vol * math.sqrt(ttm))
    vega = stock_spot * dvd_disc * math.sqrt(ttm) * norm.pdf(d1)
    theta = (
        -0.5 * vega * vol / ttm 
        + dividend * stock_spot * dvd_disc * norm.cdf(d1)
        - rate * strike * disc * norm.cdf(d2)
        )
    rho = strike * ttm * disc * norm.cdf(d2)
    return price, delta, gamma, theta, vega, rho

@pytest.mark.parametrize("vol", [0.01, 0.1, 0.2, 0.35, 0.5])
@pytest.mark.parametrize("rate", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("strike", [90, 100, 110], ids=['OTM', 'ATM', 'ITM'])
@pytest.mark.parametrize("ttm", [0.1, 1, 5])
def test_same_as_euro_when_no_dividend(vol, rate, strike, ttm):
    STOCK_SPOT = 100.0
    n_steps = 20000
    bump_size = 1e-3
    price_rel_error = 1e-4
    price_abs_error = 1e-3
    inputs = {
        'stock_spot': STOCK_SPOT,
        'vol': vol,
        'rate': rate,
        'strike': strike,
        'ttm': ttm,
        'dividend': 0.0,
        'n_steps': n_steps,
    }
    a_price, a_delta, a_gamma, a_theta = CRR_american_call(**inputs)
    a_vega = cent_fdm(**inputs, bump_param='vol', bump_size=bump_size)
    a_rho = cent_fdm(**inputs, bump_param='rate', bump_size=bump_size)

    inputs.pop('n_steps')
    e_price, e_delta, e_gamma, e_theta, e_vega, e_rho = bsm_prices_greeks(
        **inputs
    )
    assert a_price == pytest.approx(
        e_price, rel=price_rel_error, abs=price_abs_error
    )

    price_error = max(price_abs_error, price_rel_error * abs(e_price))
    dt = ttm / n_steps
    up_factor = math.exp(vol * math.sqrt(dt))
    down_factor = 1 / up_factor
    stock_width = STOCK_SPOT * (up_factor - down_factor)
    upper_stock_width = up_factor * stock_width
    lower_stock_width = down_factor * stock_width

    delta_error = 2 * price_error / stock_width
    gamma_error = (
        2 * price_error
        * (1 / upper_stock_width + 1 / lower_stock_width)
        / stock_width
    )
    theta_error = price_error / dt
    vega_error = price_error / bump_size
    rho_error = price_error / bump_size

    assert a_delta == pytest.approx(e_delta, rel=0, abs=delta_error)
    assert a_gamma == pytest.approx(e_gamma, rel=0, abs=gamma_error)
    assert a_theta == pytest.approx(e_theta, rel=0, abs=theta_error)
    assert a_vega == pytest.approx(e_vega, rel=0, abs=vega_error)
    assert a_rho == pytest.approx(e_rho, rel=0, abs=rho_error)

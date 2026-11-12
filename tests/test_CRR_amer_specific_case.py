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

# [0.01, 0.1, 0.2, 0.35, 0.5]

@pytest.mark.parametrize(
    ("vol", "rate", "strike", "ttm"),
    [
        (0.01, 0.1, 110, 1),
        (0.01, 0.01, 100, 5),
        (0.01, 0.01, 110, 5)
    ]
)
def test_same_as_euro_when_no_dividend(vol, rate, strike, ttm):
    bump_size = 1e-3
    inputs = {
        'stock_spot': 100.0,
        'vol': vol,
        'rate': rate,
        'strike': strike,
        'ttm': ttm,
        'dividend': 0.0,
        'n_steps': 150000,
    }
    a_price, a_delta, a_gamma, a_theta = CRR_american_call(**inputs)
    a_vega = cent_fdm(**inputs, bump_param='vol', bump_size=bump_size)
    a_rho = cent_fdm(**inputs, bump_param='rate', bump_size=bump_size)

    inputs.pop('n_steps')
    e_price, e_delta, e_gamma, e_theta, e_vega, e_rho = bsm_prices_greeks(
        **inputs
    )
    assert a_price == pytest.approx(e_price, rel=1e-4, abs=1e-5), (
        f"price: CRR={a_price:.12g}, BSM={e_price:.12g}, "
        f"abs_error={abs(a_price - e_price):.6e}"
    )
    assert a_delta == pytest.approx(e_delta, rel=1e-3, abs=1e-4), (
        f"delta: CRR={a_delta:.12g}, BSM={e_delta:.12g}, "
        f"abs_error={abs(a_delta - e_delta):.6e}"
    )
    assert a_gamma == pytest.approx(e_gamma, rel=5e-3, abs=1e-5), (
        f"gamma: CRR={a_gamma:.12g}, BSM={e_gamma:.12g}, "
        f"abs_error={abs(a_gamma - e_gamma):.6e}"
    )
    assert a_theta == pytest.approx(e_theta, rel=5e-3, abs=1e-2), (
        f"theta: CRR={a_theta:.12g}, BSM={e_theta:.12g}, "
        f"abs_error={abs(a_theta - e_theta):.6e}"
    )
    assert a_vega == pytest.approx(e_vega, rel=1e-2, abs=1e-2), (
        f"vega: CRR={a_vega:.12g}, BSM={e_vega:.12g}, "
        f"abs_error={abs(a_vega - e_vega):.6e}"
    )
    assert a_rho == pytest.approx(e_rho, rel=1e-2, abs=1e-2), (
        f"rho: CRR={a_rho:.12g}, BSM={e_rho:.12g}, "
        f"abs_error={abs(a_rho - e_rho):.6e}"
    )

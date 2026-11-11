"""Compare geometric-CV arithmetic Asian calls with independent QuantLib MC."""

from math import hypot, sqrt

import pytest
import QuantLib as ql
import torch
from scipy.stats import norm

from mc_asian import MCPricer


N_SAMPLES = 2**20
PRICE_TOLERANCE = 0.1  # Absolute price units for a spot of 100.
CI_MULTIPLIER = norm.ppf(0.995)  # Two-sided, per-case 99% confidence interval.
MAX_CI_HALF_WIDTH = PRICE_TOLERANCE / 2


def _quantlib_arithmetic_asian(
    pricer: MCPricer, strike: float, dtm: int, seed: int
) -> tuple[float, float]:
    """Return QuantLib's price and standard error on the same model time grid."""
    with ql.SavedSettings():
        evaluation_date = ql.Date(1, 1, 2026)
        ql.Settings.instance().evaluationDate = evaluation_date
        day_counter = ql.Actual365Fixed()
        calendar = ql.NullCalendar()

        # One synthetic calendar day represents one MCPricer trading step.
        # Scaling gives r_ql * n/365 = r * n/day_count, likewise for q and vol².
        time_scale = 365 / pricer.day_count
        risk_free_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(evaluation_date, pricer.rate * time_scale, day_counter)
        )
        dividend_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(evaluation_date, pricer.dividend * time_scale, day_counter)
        )
        volatility = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                evaluation_date, calendar, pricer.vol * sqrt(time_scale), day_counter
            )
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(pricer.stock_spot)),
            dividend_curve,
            risk_free_curve,
            volatility,
        )

        # Match run_mc: equally weighted future closes, excluding today's spot.
        fixing_dates = [evaluation_date + step for step in range(1, dtm + 1)]
        option = ql.DiscreteAveragingAsianOption(
            ql.Average.Arithmetic,
            0.0,
            0,
            fixing_dates,
            ql.PlainVanillaPayoff(ql.Option.Call, strike),
            ql.EuropeanExercise(fixing_dates[-1]),
        )
        option.setPricingEngine(
            ql.MCDiscreteArithmeticAPEngine(
                process,
                "pseudorandom",
                brownianBridge=False,
                antitheticVariate=False,
                controlVariate=True,
                requiredSamples=pricer.n_samples,
                seed=seed,
            )
        )
        return option.NPV(), option.errorEstimate()


@pytest.mark.parametrize(
    ("strike", "dividend", "rate"),
    [
        pytest.param(90, 0.02, 0.03, id="ITM"),
        pytest.param(100, 0.08, 0.05, id="ATM"),
        pytest.param(110, 0.04, 0.02, id="OTM"),
    ],
)
@pytest.mark.parametrize("dtm", [10, 63, 125])
@pytest.mark.parametrize("vol", [0.01, 0.1, 0.2])
@pytest.mark.parametrize("root_seed", [0, 12321])
def test_geom_cv_price_is_practically_equivalent_to_quantlib(
    strike, dtm, vol, dividend, rate, root_seed
):
    pricer = MCPricer(
        stock_spot=100,
        dividend=dividend,
        vol=vol,
        rate=rate,
        n_samples=N_SAMPLES,
        seed=root_seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    price, standard_error = pricer.run_mc(
        strike, dtm, antithetic=False, ctrl_var="geom_asian"
    )
    reference_price, reference_se = _quantlib_arithmetic_asian(
        pricer, strike, dtm, seed=root_seed + 1_000_003
    )

    # Independent streams: include uncertainty from both MC estimates.
    # QuantLib uses CV coefficient 1; MCPricer estimates it from the samples.
    difference = price - reference_price
    half_width = CI_MULTIPLIER * hypot(standard_error, reference_se)
    comparison = (
        f"MCPricer={price:.6f} (SE={standard_error:.6f}), "
        f"QuantLib={reference_price:.6f} (SE={reference_se:.6f}), "
        f"difference={difference:+.6f}, 99% CI half-width={half_width:.6f}, "
        f"tolerance={PRICE_TOLERANCE:.6f}"
    )
    assert half_width < MAX_CI_HALF_WIDTH, comparison
    assert abs(difference) + half_width < PRICE_TOLERANCE, comparison

from scipy.stats import chi2
import torch
import math
import numpy as np
from tqdm import tqdm
import pytest
from mc_asian import MCPricer


INPUT = {
    "stock_spot": 100.0,
    "vol": 0.35,
    "rate": 0.05,
    "dividend": 0.08,
    "n_samples": 2**19,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "day_count": 250,
    "verbose": False
}
OPTION = {"strike": 100.0, 'dtm': 125}
N_REP = 500
GATE = {
    "alpha": 0.05, # (1 - 2*alpha) CI
    "UPPER": 1.2 # L = 1 / U
}
STABILITY_CONST = 0.005
# for checking if n_samples is large enough to make estimator of se stable
# stable if SD(se_native) / mean(se_native) < STABILITY_CONST

@pytest.mark.parametrize("root_seed", [165, 41619, 98889], ids=["sd1", "sd2", "sd3"])
@pytest.mark.parametrize("av", [True, False])
@pytest.mark.parametrize("cv", ["geom_asian", "stock_price", None])
def test_mc_native_se_not_underestimate(av, cv, root_seed):
    root_gen = torch.Generator().manual_seed(root_seed)
    seeds = torch.randint(
            0, 2**63 - 1, (N_REP,), 
            generator=root_gen, device='cpu').tolist()
    pricer = MCPricer(**INPUT)
    prices = []
    varns = []

    av_msg = "True" if av else "False"
    cv_msg = cv if cv is not None else "None"
    print(f"\nAntithetic: {av_msg}, control car: {cv_msg}, " +
          f"root seed = {root_seed}, n rep: {N_REP}.")
    
    for seed in tqdm(seeds, unit="rep"):
        pricer.seed = seed
        price, se = pricer.run_mc(
            **OPTION, antithetic=av, ctrl_var=cv, pilot_usage=0.1
        )
        prices.append(price)
        varns.append(se**2) 
    prices = np.array(prices)
    varns = np.array(varns)

    ses = np.sqrt(varns)
    stable_ratio = np.std(ses, ddof=1).item() / np.mean(ses).item() 

    s_emp = np.std(prices, ddof=1).item()
    s_native = math.sqrt(np.mean(varns).item())    
    r_est = s_emp / s_native 
    # r_est > 1 => native se underestimates
    df = N_REP - 1
    lower = r_est * math.sqrt(df / chi2.ppf(1 - GATE['alpha'], df=df))
    upper = r_est * math.sqrt(df / chi2.ppf(GATE['alpha'], df=df))
    print(f"Stability ratio: {stable_ratio}, "+ 
          f"{(1 - 2*GATE['alpha'])*100:.1f}%CI: [{lower:.4f}, {upper:.4f}]")

    assert stable_ratio < STABILITY_CONST
    assert upper < GATE["UPPER"]
    assert lower > (1 / GATE["UPPER"])
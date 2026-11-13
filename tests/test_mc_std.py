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
    "n_samples": 2**20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "day_count": 250,
    "verbose": False
}
OPTION = {"strike": 100.0, 'dtm': 125}
N_REP = 500
ROOT_SEED = 4564198
GATE = {
    "alpha": 0.05, # (1 - 2*alpha) CI
    "width": 0.2 # U = 1 + 0.5*width
}

@pytest.mark.parametrize("av", [True, False])
@pytest.mark.parametrize("cv", ["geom_asian", "stock_price", None])
def test_mc_native_std_not_underestimate(av, cv):
    root_gen = torch.Generator().manual_seed(ROOT_SEED)
    seeds = torch.randint(
            0, 2**63 - 1, (N_REP,), 
            generator=root_gen, device='cpu').item()
    pricer = MCPricer(**INPUT)
    prices = []
    varns = []
    for seed in tqdm(seeds, unit="rep"):
        price.seed = seed
        price, std = pricer.run_mc(
            **OPTION, antithetic=av, ctrl_var=cv, pilot_usage=0.1
        )
        prices.append(price)
        varns.append(std**2) 
    prices = np.array(prices)
    varns = np.array(varns)
    s_emp = np.std(prices, ddof=1).item()
    s_native = math.sqrt(np.mean(varns).item())
    
    r_est = s_emp / s_native 
    # r_est > 1 => native std underestimates
    df = N_REP - 1
    lower = r_est * math.sqrt(df / chi2.ppf(1 - GATE['alpha'], df=df))
    upper = r_est * math.sqrt(df / chi2.ppf(GATE['alpha'], df=df))
    av_msg = "True" if av else "False"
    cv_msg = cv if cv is not None else "None"
    print(f"Antithetic: {av_msg}, Control Var: {cv_msg}, Replicates: {N_REP} " + 
          f"{(1 - 2*GATE['alpha'])*100:.1f}%CI: [{lower:.4f}, {upper:.4f}]")
    
    assert upper < 1 + GATE["width"] * 0.5
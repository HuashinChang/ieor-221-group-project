import torch
import math
from dataclasses import dataclass
from typing import Literal, Iterable
from random import randint
from scipy.stats import norm
from tqdm import tqdm

@dataclass
class MCPricer:
    """
    An Monte Carlo pricer for fixed-strike arithmetic Asian option.
    The underlying stock dynamic is BSM, 
    and the average is computed with the daily closing price.

    ``n_samples`` referes to number of *independent* samples
    """
    stock_spot: float
    dividend: float
    vol: float
    rate: float
    n_samples: int = 2**20
    seed: int | None = None
    device: torch.device = torch.device("cpu")
    day_count: int = 250
    verbose: bool = False

    def _make_gen(self) -> torch.Generator:
        seed = self.seed if self.seed is not None else randint(0, 2**63 - 1)
        return torch.Generator(device=self.device).manual_seed(seed)

    def run_mc(
        self, 
        strike: float, 
        dtm: int,
        antithetic: bool = False,
        ctrl_var: Literal['stock_prices', 'geom_asian'] | None = None,
    ) -> tuple[float, float]:

        N = int(self.n_samples) * 2 if antithetic else int(self.n_samples)
        stock_prices = self.stock_spot * torch.ones(N, device=self.device)
        price_sums = log_price_sums = torch.zeros_like(stock_prices)

        gen = self._make_gen()

        for _ in self._iterable(dtm):
            stock_prices = self._evolve(stock_prices, gen, antithetic)
            price_sums = price_sums + stock_prices
            log_price_sums = log_price_sums + torch.log(stock_prices)

        price_a_avg = price_sums / dtm
        price_g_avg = torch.exp(log_price_sums / dtm)

        discount = math.exp(-self.rate * dtm / self.day_count)
        disc_a_payoffs = torch.where(price_a_avg > strike, price_a_avg - strike, 0) * discount

        if ctrl_var == 'stock_prices':
            ctrl_sampled = stock_prices
            ctrl_ana_val = self.stock_spot * math.exp((self.rate - self.dividend)*(dtm / self.day_count))
        elif ctrl_var == 'geom_asian':
            ctrl_sampled = torch.where(price_g_avg > strike, price_g_avg - strike, 0) * discount
            ctrl_ana_val = self._analytic_geom_asian(strike, dtm)
        else:
            ctrl_sampled = torch.zeros_like(stock_prices)
            ctrl_ana_val = 0
 
        if antithetic:
            half = int(self.n_samples)
            disc_a_payoffs = (disc_a_payoffs[:half] + disc_a_payoffs[half:]) / 2
            ctrl_sampled = (ctrl_sampled[:half] + ctrl_sampled[half:]) / 2

        payoffs = self._apply_ctrl_var(
                orig_payoffs=disc_a_payoffs,
                ctrl_sampled=ctrl_sampled,
                ctrl_ana_val=ctrl_ana_val
            )
        
        npv = torch.mean(payoffs)
        std = torch.std(payoffs, correction=1) / math.sqrt(self.n_samples)
        return npv.item(), std.item()

    def _iterable(self, dtm: int) -> Iterable:
        if self.verbose:
            return tqdm(
                range(dtm),
                desc="Simulating...",
                bar_format="{desc}: {n_fmt}/{total_fmt}|{bar}| {rate_fmt}",
                unit="step",
                unit_scale=True,
            )
        return range(dtm)

    def _evolve(
            self,
            stock_prices: torch.Tensor,
            generator: torch.Generator,
            antithetic: bool
        ) -> torch.Tensor:
        Z = torch.randn(
            int(self.n_samples), 
            device=self.device, 
            dtype=torch.float64, 
            generator=generator
        )
        if antithetic:
            Z = torch.cat([Z, -Z])
        
        drift = (self.rate - self.dividend - 0.5 * self.vol**2) / self.day_count
        diffusion = self.vol * Z / math.sqrt(self.day_count)
        return stock_prices * torch.exp(drift + diffusion)

    def _analytic_geom_asian(self, strike: float, dtm: int) -> float:
        ytm = dtm / self.day_count

        A = 0.5 * (dtm + 1) * ytm / dtm
        log_mean = math.log(self.stock_spot) + (self.rate - self.dividend - 0.5 * self.vol**2) * A
        log_var = self.vol**2 * ytm * (dtm + 1) * (2*dtm + 1) / (6 * dtm**2)

        d1 = (log_mean - math.log(strike) + log_var) / math.sqrt(log_var)
        d2 = d1 - math.sqrt(log_var)

        discount = math.exp(-self.rate * ytm)
        return discount * (math.exp(log_mean + 0.5*log_var) * norm.cdf(d1) - strike * norm.cdf(d2))

    def _apply_ctrl_var(
        self,
        orig_payoffs: torch.Tensor, 
        ctrl_sampled: torch.Tensor,
        ctrl_ana_val: float, 
    ) -> tuple[float, float]:
        orig_mean = torch.mean(orig_payoffs)
        ctrl_mean = torch.mean(ctrl_sampled)
        ctrl_m2 = torch.sum((ctrl_sampled - ctrl_mean)**2)
        if ctrl_m2 == 0:
            est_corr = 0
        else:
            est_corr = torch.sum((orig_payoffs - orig_mean) * (ctrl_sampled - ctrl_mean)) / ctrl_m2

        return orig_payoffs - est_corr * (ctrl_sampled - ctrl_ana_val)
        

if __name__ == "__main__":
    mkt_cond = {
        'stock_spot': 100,
        'dividend': 0.08,
        'vol': 0.35,
        'rate': 0.05
    }
    contract = {
        'dtm': 125,
        'strike': 100
    }
    mc_spec = {
        'seed': 0,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'n_samples': 2**20
    }
    pricer = MCPricer(**mkt_cond, **mc_spec)

    print("Running plain MC...")
    npv, std = pricer.run_mc(**contract)
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    print("\nRunning control variate (stock_prices) MC...")
    npv, std = pricer.run_mc(**contract, ctrl_var='stock_prices')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    print("\nRunning control variante (geomtric Asian) MC...")
    npv, std = pricer.run_mc(**contract, ctrl_var='geom_asian')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    pricer.n_samples = int(mc_spec['n_samples'] / 2)
    print("\nRunning antithetic MC...")
    npv, std = pricer.run_mc(**contract, antithetic=True)
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    print("\nRunning anthithetic with control variate (stock_prices) MC...")
    npv, std = pricer.run_mc(**contract, antithetic=True, ctrl_var='stock_prices')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    print("\nRunning anthithetic with control variate (geomtric Asian) MC...")
    npv, std = pricer.run_mc(**contract, antithetic=True, ctrl_var='geom_asian')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")
            
            
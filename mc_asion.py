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
    """
    stock_spot: float
    dividend: float
    vol: float
    rate: float
    n_paths: int = 1_000_000
    seed: int | None = None
    device: torch.device = torch.device("cuda")
    day_count: int = 250

    def _make_gen(self) -> torch.Generator:
        seed = self.seed if self.seed is not None else randint(0, 2**63 - 1)
        return torch.Generator(device=self.device).manual_seed(seed)

    def run_mc(
            self, 
            strike: float, 
            dtm: int, 
            policy: Literal['plain', 'antithetic', 'ctrl_var']
        ) -> tuple[float, float]:
        """
        When simulation policy is 'antithetic', the total simulated path is **doubled**.
        """
        is_antithetic = policy == 'antithetic'
        N = int(self.n_paths) * 2 if is_antithetic else int(self.n_paths)
        stock_price = self.stock_spot * torch.ones(N, device=self.device)
        price_sums = log_price_sums = torch.zeros_like(stock_price)

        gen = self._make_gen()

        for _ in self._iterable(dtm):
            stock_price = self._evolve(stock_price, gen, is_antithetic)
            price_sums = price_sums + stock_price
            log_price_sums = log_price_sums + torch.log(stock_price)

        price_a_avg = price_sums / dtm
        price_g_avg = torch.exp(log_price_sums / dtm)

        discount = math.exp(-self.rate * dtm / self.day_count)
        disc_a_payoffs = torch.where(price_a_avg > strike, price_a_avg - strike, 0) * discount

        if policy == 'ctrl_var':
            disc_g_payoffs = torch.where(price_g_avg > strike, price_g_avg - strike, 0) * discount
            return self._ctrl_var_returns(disc_a_payoffs, disc_g_payoffs, strike, dtm)

        if is_antithetic:
            half = int(self.n_paths)
            disc_a_payoffs = (disc_a_payoffs[:half] + disc_a_payoffs[half:]) / 2
        
        npv = torch.mean(disc_a_payoffs)
        std = torch.std(disc_a_payoffs, correction=1) / math.sqrt(self.n_paths)
        return npv.item(), std.item()

    @staticmethod
    def _iterable(dtm: int) -> Iterable:
        return tqdm(
            range(dtm),
            desc="Simulating...",
            bar_format="{desc}: {n_fmt}/{total_fmt}|{bar}| {rate_fmt}",
            unit="step",
            unit_scale=True,
        )

    def _evolve(
            self,
            stock_price: torch.Tensor,
            generator: torch.Generator,
            antithetic: bool
        ) -> torch.Tensor:
        Z = torch.randn(
            int(self.n_paths), 
            device=self.device, 
            dtype=torch.float64, 
            generator=generator
        )
        if antithetic:
            Z = torch.cat([Z, -Z])
        
        drift = (self.rate - self.dividend - 0.5 * self.vol**2) / self.day_count
        diffusion = self.vol * Z / math.sqrt(self.day_count)
        return stock_price * torch.exp(drift + diffusion)

    def _analytic_geom_asian(self, strike: float, dtm: int) -> float:
        ytm = dtm / self.day_count

        A = 0.5 * (dtm + 1) * ytm / dtm
        log_mean = math.log(self.stock_spot) + (self.rate - self.dividend - 0.5 * self.vol**2) * A
        log_var = self.vol**2 * ytm * (dtm + 1) * (2*dtm + 1) / (6 * dtm**2)

        d1 = (log_mean - math.log(strike) + log_var) / math.sqrt(log_var)
        d2 = d1 - math.sqrt(log_var)

        discount = math.exp(-self.rate * ytm)
        return discount * (math.exp(log_mean + 0.5*log_var) * norm.cdf(d1) - strike * norm.cdf(d2))

    def _ctrl_var_returns(
        self,
        disc_a_payoffs: torch.Tensor, 
        disc_g_payoffs: torch.Tensor,
        strike: float,
        dtm: int,
    ) -> tuple[float, float]:
        arith_mean = torch.mean(disc_a_payoffs)
        geom_mean = torch.mean(disc_g_payoffs)
        geom_m2 = torch.sum((disc_g_payoffs - geom_mean)**2)
        if geom_m2 == 0:
            est_corr = 0
        else:
            est_corr = torch.sum((disc_a_payoffs - arith_mean) * (disc_g_payoffs - geom_mean)) / geom_m2

        geom_ana = self._analytic_geom_asian(strike, dtm)

        ctrl_var_payoffs = disc_a_payoffs - est_corr * (disc_g_payoffs - geom_ana)
        npv = torch.mean(ctrl_var_payoffs)
        std = torch.std(ctrl_var_payoffs, correction=1) / math.sqrt(self.n_paths)
        return npv.item(), std.item()


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
        'device': torch.device('cuda'),
        'n_paths': 2**24
    }
    pricer = MCPricer(**mkt_cond, **mc_spec)

    print("Running plain MC...")
    npv, std = pricer.run_mc(**contract, policy='plain')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    # the underlying implementation for antithetic
    # mc doubles the number of simulated path,
    # so we half the n_paths to make it the same as other methods
    pricer.n_paths = 0.5 * pricer.n_paths
    print("\nRunning antithetic MC...")
    npv, std = pricer.run_mc(**contract, policy='antithetic')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")

    pricer.n_paths = 2 * pricer.n_paths
    print("\nRunning control variande method MC...")
    print("control policy is using geometric asian option")
    npv, std = pricer.run_mc(**contract, policy='ctrl_var')
    print(f"npv is {npv:.4f} with mc std {std:.4f}")
        
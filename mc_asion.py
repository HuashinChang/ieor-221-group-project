import torch
from dataclasses import dataclass
from typing import Literal
from random import randint

@dataclass(frozen=True)
class MCPricer:
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

    def run_simulation(
            self, 
            strike: float, 
            dtm: int, 
            policy: Literal['plain', 'antithetic', 'ctrl_var']
        ):
        is_antithetic = policy == 'antithetic'
        N = self.n_paths * 2 if is_antithetic else self.n_paths
        stock_price = self.stock_spot * torch.ones(N, device=self.device)
        price_sums = stock_price
        log_price_sums = torch.log(stock_price)

        gen = self._make_gen()
        
        for _ in range(dtm):
            stock_price = self._evolve(stock_price, gen, is_antithetic)
            price_sums = price_sums + stock_price
            log_price_sums = log_price_sums + torch.log(stock_price)

        price_a_avg = price_sums / dtm
        price_g_avg = torch.exp(log_price_sums / dtm)

        discount = torch.exp(-self.rate * dtm / self.day_count)
        disc_a_payoffs = torch.where(price_a_avg > strike, price_a_avg - strike, 0) * discount

        if policy is 'ctrl_var':
            disc_g_payoffs = torch.where(price_g_avg > strike, price_g_avg - strike, 0) * discount
            return self._ctrl_var_returns(disc_a_payoffs, disc_g_payoffs)

        if is_antithetic:
            disc_a_payoffs = (disc_a_payoffs[:self.n_paths] + disc_a_payoffs[self.n_paths:]) / 2
        
        npv = torch.mean(disc_a_payoffs)
        std = torch.std(disc_a_payoffs, correction=1) / torch.sqrt(self.n_paths)
        return npv.item(), std.item()

    def _evolve(self, stock_price, generator, antithetic: bool = False):
        Z = torch.randn(
            self.n_paths, 
            device=self.device, 
            dtype=torch.float64, 
            generator=generator
        )
        if antithetic:
            Z = torch.cat([Z, -Z])
        
        drift = (self.rate - self.dividend - 0.5 * self.vol**2) / self.day_count
        diffusion = self.vol * Z / torch.sqrt(self.day_count)
        return stock_price * torch.exp(drift + diffusion)

    def _analytic_geom_asian(self) -> float:
        pass

    def _ctrl_var_returns(
        self,
        disc_a_payoffs: torch.Tensor, 
        disc_g_payoffs: torch.Tensor,
    ) -> tuple[float, float]:
        arith_mean = torch.mean(disc_a_payoffs)
        geom_mean = torch.mean(disc_g_payoffs)
        est_corr = (
            torch.sum((disc_a_payoffs - arith_mean) * (disc_g_payoffs - geom_mean)) 
            / torch.sum((disc_g_payoffs - geom_mean)**2))
        geom_ana = self._analytic_geom_asian()

        npv = arith_mean - est_corr * (geom_mean - geom_ana)
        std = torch.std(arith_mean / self.n_paths) * torch.sqrt(1 - est_corr**2)
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
        'n_path': 1_000_000
    }
    pricer = MCPricer(**mkt_cond, **mc_spec)
    
        
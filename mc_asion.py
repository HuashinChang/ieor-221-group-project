import torch
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class MCPricer:
    stock_spot: float
    dividend: float
    vol: float
    rate: float
    n_paths: int = 1_000_000
    device: torch.device = torch.device("cuda")


    def _simulation(
            self, 
            strike: float, 
            dtm: int, 
            policy: Literal['plain', 'antithetic', 'ctrl_var']
        ):
        stock = self.stock_spot * torch.ones(self.n_paths, device=self.device)
        stock_arith_avg = stock
        log_stock_geom_avg = torch.log(stock)


        for i in range(dtm):
            stock = self._evolve(stock)
            stock_arith_avg = (stock_arith_avg * i + stock) / (i + 1)
            log_stock_geom_avg = (
                (log_stock_geom_avg) * i + torch.log(stock)) / (i + 1)

        stock_geom_avg = torch.exp(log_stock_geom_avg)

        arith_payoffs = torch.where(stock_arith_avg > strike, stock_arith_avg - strike, 0)
        geom_payoffs = torch.where(stock_geom_avg > strike, stock_geom_avg - strike, 0)

        if policy == 'plain':
            pass
        

    def _sampling(self):
        pass

    def _evolve(self):
        pass




if __name__ == "__main__":
    pricer = MCPricer(
        stock_spot=100.0,
        dividend=0.08,
        rate=0.05,
    ) 
        
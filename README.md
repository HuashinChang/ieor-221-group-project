The main implementations are in `tree_american.py` and `mc_asian.py`. The dependencies are `numpy`, `torch>=2.0`, `scipy`, and `tqdm`.

Tested with Python 3.14.8 on Linux. To rerun the tests, run `pytest` from the project directory. The tests additionally require `pytest` and `QuantLib`. MC tests use CUDA when available and otherwise run on CPU. The full test suite may take a while to complete.

### Reproduce answers

Run the following command from the project directory to reproduce the answer tables using the default parameters:

```bash
python answers.py
```
For MC simulation, CUDA is used when available; otherwise, the script runs on CPU.

You can also run a specific problem with your choice of methods and parameters. For example, use

```bash
python answers.py p1 --method BBS --start-exp 11 --max-exp 15
```
to run a BBS tree with step counts doubling from 2048 to 32768, or use

```bash
python answers.py p2 --n-paths $((2**24)) --device "cuda" --seed 21
```
to run MC on CUDA with a random seed of 21 and $2^{24}$ simulated paths. See `python answers.py -h` for more information.

### Other plots
`matplotlib` is required. Run `plot_tree_asymptotic.py` for CRR:

```bash
python plot_tree_asymptotic.py
```

This produces graphs for inspecting the asymptotic behavior of the CRR tree.

Run `plot_mc_method_se_compare.py` to compare MC standard errors and estimated variance reduction across methods:

```bash
python plot_mc_method_se_compare.py
```

To examine the accuracy of the reported MC standard errors, run

```bash
python plot_mc_se_ci.py
```

This script may take a while to run. It compares the reported MC standard errors with the empirical standard deviation of price estimates from repeated simulations.

To see how the MC price estimates and confidence intervals change as the number of simulated paths increases, run

```bash
python plot_mc_asymptotic.py
```

All of the above scripts accept command-line options. Run `python <script_name>.py -h` for details. By default, figures are saved in `plots/`, and any CSV outputs are saved in `csv/`.

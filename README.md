The main implementations are in `tree_american.py` and `mc_ansian.py`. The dependencies are `numpy`, `torch>=2.0`, `scipy`, `tqdm`.

Tested with Python 3.14.8 on Linux. Run `pytest` from the project directory to replicate the test results. A CUDA device and `quantlib` are required for tests, and it takes a while to run the full test cases.

### Reproduce answers

Run from the project directory to reproduce the answer with default parameters:

```bash
python answers.py
```
For MC simulation, CUDA is used when available; otherwise, the script runs on CPU.

You can also run a specific problem with your choice of methods and parameters. For example, use

```bash
python answers.py p1 --method BBSR --start-exp 11 --max-exp 15
```
to run a BBSR tree at doubling step counts from 2048 to 32768, or use

```bash
python answers.py p2 --n-paths $((2**24)) --device "cuda"
```
to run MC with 2**24 paths on CUDA. See `-h` for more information.

### Other plots
`matplotlib` is required. Run `plot_tree_asymptotic.py` for CRR:

```bash
python plot_tree_asymptotic.py
```

This produces graphs for inspecting the asymptotic behavior of the CRR tree.

Run the file `plot_mc_amethod_std_compare.py` to generate variance-reduction figures comparing the mc variance from different methods:

```bash
python plot_mc_amethod_std_compare.py
```

To inspect deeper into the behavior of the MC standard error calculated from the sampled paths, run 

```bash
python plot_mc_std_ci.py
```

This takes a while to run, and it provides a basic statistic check for the approximation quality of the native MC std to the true MC standard deviation.

To see how the result of MC estimator changes with respect to number of samples, run

```bash
python plot_mc_asymptotic.py
```

All of above script can be adjust using different arguments. See `-h` for more details. Figures are saved in `plots/`, and CSV results are saved in `csv/`. 

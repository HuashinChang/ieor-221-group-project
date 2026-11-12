The dependencies are `torch>=2.0`, `scipy`, `tqdm`, and `matplotlib`.

Tested with Python 3.14.8 on Linux. Run `pytest` from the project directory to replicate the test results. A CUDA device and `quantlib` are required for tests, and it takes a while to run the full test cases.

### Problem 1
Implemented in `CRR_american.py`. Run from the project directory to reproduce answers to the Problem 1:

```bash
python CRR_american.py
```
BBSR tree is also implemented, use `--method` to selected tree method.

Run the convergence comparison for CRR, CRR with Black–Scholes smoothing, and BBSR:

```bash
python plot_tree_convergence.py
```

This uses the Problem 1 American-call parameters and saves price convergence and
absolute-error panels to `plots/tree_convergence.png`. Adjacent even/odd step counts
show tree oscillations. Errors use a numerical BBSR reference at N = 10,000, not an
exact solution. BBSR combines smoothed trees at N and 2N, so the horizontal axis
compares base step counts rather than equal computational cost.

To choose step counts or the numerical reference resolution:

```bash
python plot_tree_convergence.py --n-steps 50 51 100 101 200 201 500 501 --reference-steps 10000
```

The default output directory for both plotting scripts is `plots/` beside the
scripts, regardless of the working directory. The tree script also accepts
`--output-dir` to override its destination.

### Problem 2 
The main implementation of Problem 2 is in `mc_asian.py`. Run from the project directory to reproduce answers to the Problem 2:

```bash
python mc_asian.py
```
CUDA is used when available; otherwise, the script runs on CPU (it may take some time if you run it on CPU).

You can also run the file `plot_mc_asian_std.py` to generate variance-reduction figures comparing the mc variance from different methods:

```bash
python plot_mc_asian_std.py
```
The produced figures are saved in `plots/`: `mc_std_itm.png`, `mc_std_atm.png`,
`mc_std_otm.png`, and `mc_variance_reduction.png`.

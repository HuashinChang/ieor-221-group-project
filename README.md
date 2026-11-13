### Reproduce answers
The dependencies are `numpy`, `torch>=2.0`, `scipy`, `tqdm`, `csv`.
Run from the project directory to reproduce the answer with default parameters:

```bash
python answer.py
```
The results is saved in `plots/`. For MC simulation, CUDA is used when available; otherwise, the script runs on CPU.

You can also run a specific problem with your choice of methods and parameters. For example, use

```bash
python answers.py p1 --method BBSR --start-exp 11 --max-exp 15
```
to run a BBSR tree at doubling step counts from 2048 to 32768, or use

```bash
python answers.py p2 --n-paths $((2**24)) --device "cuda"
```
to run MC with 2**24 paths on CUDA.

CRR tree is implemented in `tree_american.py`, and the 

### Tests
Tested with Python 3.14.8 on Linux. Run `pytest` from the project directory to replicate the test results. A CUDA device and `quantlib` are required for tests, and it takes a while to run the full test cases.

### Other plots
`matplotlib` is required. Run `plot_tree_asymptotic.py` for CRR:

```bash
python plot_tree_asymptotic.py
```

This produces graphs for inspecting the asymptotic behavior of the CRR tree.

Run the file `plot_mc_asian_std.py` to generate variance-reduction figures comparing the mc variance from different methods:

```bash
python plot_mc_asian_std.py
```
All produced figures are saved in `plots/`.

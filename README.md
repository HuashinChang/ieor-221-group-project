The main implementation of Problem 2 is in `mc_asian.py`. 

The dependencies are `torch>=2.0`, `scipy`, `tqdm`, and `matplotlib`.
Tested with Python 3.14.8 on Linux.

### Result Reproduction

Run from the project directory to reproduce answers to the Problem 2:

```bash
python mc_asian.py
```
CUDA is used when available; otherwise, the script runs on CPU (it may take some time if you run it on CPU).

You can also run the file `plot_mc_asian_std.py` to generate variance-reduction figures comparing the mc variance from different methods:

```bash
python plot_mc_asian_std.py
```
The produced figures are saved in `mc_std_plits/`

### Tests

You can use tests in `tests/` check practical price equivalence 
- across variance-reduction methods and
- against QuantLib's BSM arithmetic Asian Monte Carlo engine.

Tests take some time to run, and a CUDA device is needed. 
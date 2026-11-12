The dependencies are `torch>=2.0`, `scipy`, `tqdm`, and `matplotlib`.

Tested with Python 3.14.8 on Linux. Run `pytest` from the project directory to replicate the test results. A CUDA device and `quantlib` are required for tests, and it takes a while to run the full test cases.

### Problem 1
Implemented in `CRR_american.py`. Run from the project directory to reproduce answers to the Problem 1:

```bash
python CRR_american.py
```

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
The produced figures are saved in `mc_std_plits/`
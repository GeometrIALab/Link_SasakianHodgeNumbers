# Environment

All scripts in `code/` and `certificates/` run with Python 3.9 or later and two packages, `numpy` and `sympy`. The isolation certificates also need the computer algebra system Singular.

## Python virtual environment

From the root of the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

Then run the scripts from their own folders with the environment active, for example:

```bash
cd code && python k3_links.py
cd ../certificates && python six_remaining_7d.py
```

To leave the environment, run `deactivate`. The `.venv` folder is listed in `.gitignore`.

The results in `certificates/results/` were produced on these versions, on an Apple M1 laptop under macOS 26:
* Python 3.10.13, numpy 2.2.6 and sympy 1.14.0 for the scripts in this repository;
* numpy 1.26.4 for the original dataset runs.

Several scripts use 8 worker processes. The number is set in the scripts, and for `isolation_certificates.py` by the variable `NPROC`.

## Singular (only for `certificates/isolation_certificates*.py`)

Any recent Singular works; the results were produced with Singular 4.4.1. Some ways to install it:

```bash
conda create -p ./singular-env -c conda-forge singular      # conda / mamba, no administrator rights needed
brew install singular                                        # macOS with Homebrew
sudo apt install singular                                    # Debian / Ubuntu
```

Pass the path of the binary through the variable `SINGULAR`, for example:

```bash
cd certificates
SINGULAR=../singular-env/bin/Singular python isolation_certificates_k3gb.py
SINGULAR=../singular-env/bin/Singular NPROC=4 python isolation_certificates.py --only six7d_dense
```

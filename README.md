[![DOI](https://zenodo.org/badge/851852346.svg)](https://zenodo.org/doi/10.5281/zenodo.13652743)

# Optimal control of parametrized linear systems using reduced basis methods and machine learning
In this repository, we provide the code used for the numerical experiments on the extended adaptive model hierarchy.

## Installation
On a system with `git` (`sudo apt install git`), `python3` (`sudo apt install python3-dev`) and
`venv` (`sudo apt install python3-venv`) installed, the following commands should be sufficient
to install the `adaptive-ml-control` package with all required dependencies in a new virtual environment:
```
git clone https://github.com/HenKlei/ADAPTIVE-REDUCED-OPT-CONTROL.git
cd ADAPTIVE-REDUCED-OPT-CONTROL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Running the experiments
To reproduce the results, we provide the original scripts creating the results
in the directory [`adaptive_ml_control/examples/`](adaptive_ml_control/examples/).

## Questions
If you have any questions, feel free to contact me via email at <hendrik.kleikamp@uni-muenster.de>.

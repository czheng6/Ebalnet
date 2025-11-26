# Ebalnet: Adapting Neural Networks for Entropy Balancing

A novel neural network architecture for causal effect estimation that combines deep learning with entropy balancing to achieve double robustness and semiparametric efficiency.

## Overview

Ebalnet learns a representation that can linearly predict both the propensity score and outcomes, enabling entropy balancing estimators to achieve:
- **Double Robustness**: Consistent estimation when either the propensity score or outcome model is correctly specified
- **Semiparametric Efficiency**: Achieves the variance bound for ATT estimation
- **Stable Finite-Sample Performance**: Leverages entropy balancing's robustness properties

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
ebalnet/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── ebal_util.py          # Core Ebalnet implementation
│   └── baseline_methods.py   # Baseline methods (IPW, TARNet, CFRNet, Dragonnet)
```

## Quick Start

### Training Ebalnet

```python
from src.ebal_util import NNEbal

# Define model parameters
params = {
    "input_dim": X.shape[1],
    "use_adam": True,
    "verbose": False,
    "act_fn": "gelu",
    "epochs": 500,
    "num_layers": 5,
    "embedding_dim": 16,
    "neurons_per_layer": 100,
    "reg_l2": 5,
    "learning_rate": 1e-3,
}

# Initialize and train
model = NNEbal(params)
model.fit(X, treatment, y)

# Estimate ATT
att_estimate = model.predict_att(X, treatment, y)
```

### Using Baseline Methods

```python
from src.baseline_methods import IPW, TARNet, CFRNet_WASS, Dragonnet

# IPW with logistic regression
ipw = IPW(random_seed=42)
ipw.fit(X, treatment, y)
att_ipw = ipw.predict_att(X, treatment, y)

# TARNet
tarnet = TARNet(input_dim=X.shape[1], epochs=300, verbose=False)
tarnet.fit(X, treatment, y)
att_tarnet = tarnet.predict_att(X, treatment, y)

# CFRNet with Wasserstein regularization
cfrnet = CFRNet_WASS(input_dim=X.shape[1], alpha=1.0, epochs=300)
cfrnet.fit(X, treatment, y)
att_cfrnet = cfrnet.predict_att(X, treatment, y)

# Dragonnet
dragonnet = Dragonnet(input_dim=X.shape[1], epochs=300)
dragonnet.fit(X, treatment, y)
att_dragonnet = dragonnet.predict_att(X, treatment, y)
```

### Entropy Balancing (without neural network)

```python
from src.ebal_util import ebal_bin
import pandas as pd

ebal = ebal_bin(effect='ATT', PCA=True, print_level=-1)
result = ebal.ebalance(treatment, pd.DataFrame(X), y)
weights = result['w']
```

## Methods Implemented

| Method | Description | Reference |
|--------|-------------|-----------|
| **Ebalnet** | Neural network + entropy balancing | This paper |
| **EB** | Entropy Balancing | Hainmueller (2012) |
| **IPW** | Inverse Probability Weighting | Rosenbaum & Rubin (1983) |
| **TARNet** | Treatment-Agnostic Representation Network | Shalit et al. (2017) |
| **CFRNet WASS** | Counterfactual Regression with Wasserstein | Shalit et al. (2017) |
| **Dragonnet** | Three-headed network with targeted regularization | Shi et al. (2019) |

## Benchmark Datasets

The package includes data loaders for standard causal inference benchmarks:

### IHDP (Infant Health and Development Program)
- Semi-synthetic dataset with 1000 realizations
- 25 covariates, ~750 samples per realization
- Ground truth available for evaluation

### JOBS (LaLonde Job Training)
- Real-world dataset combining experimental and observational data
- 8 covariates, ~3000 samples
- Ground truth ATT from randomized portion

## Running Experiments

Reproduce the results from the paper:

```bash
# Full experiment (takes several hours)
python run_experiments.py --ihdp_realizations 1000 --jobs_bootstrap 100

# Quick test run
python run_experiments.py --ihdp_realizations 10 --jobs_bootstrap 5

# Run only IHDP
python run_experiments.py --dataset ihdp --ihdp_realizations 100

# Run only JOBS
python run_experiments.py --dataset jobs --jobs_bootstrap 50

# Custom settings
python run_experiments.py \
    --dataset both \
    --ihdp_realizations 1000 \
    --jobs_bootstrap 100 \
    --data_dir data \
    --output_dir results \
    --nn_epochs 300
```

Results will be saved to:
- `results/experiment_summary.csv` - Summary table
- `results/ihdp_detailed_results.csv` - Per-realization IHDP results
- `results/jobs_detailed_results.csv` - Per-bootstrap JOBS results


## References

- Hainmueller, J. (2012). Entropy balancing for causal effects. Political Analysis.
- Zhao, Q. (2016). Entropy balancing is doubly robust. Journal of Causal Inference.
- Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect. ICML.
- Shi, C., Blei, D., & Veitch, V. (2019). Adapting neural networks for the estimation of treatment effects. NeurIPS.

## License

MIT License


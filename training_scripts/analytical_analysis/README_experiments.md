# Analytical Experiments

This directory contains the synthetic-data generation and inverse-problem analysis scripts used for the analytical appendix experiments.

## Current Files

- `generate_synthetic_dataset_v2.py`: synthetic rotor-dynamics dataset generator
- `data_loader_v2.py`: helpers for loading generated simulations
- `nlls_baseline_and_sensitivity_experiments.py`: NLLS and sensitivity-analysis experiment suite
- `visualize_dataset_v2.py`: dataset inspection and visualization
- `README_experiments.md`: this file

## Important Repo-State Note

Both `generate_synthetic_dataset_v2.py` and `nlls_baseline_and_sensitivity_experiments.py` import `RotatingMachinerySimulator` from `baseForV2.py`, but that file is not present in this folder in the current repository snapshot. To run these scripts, `baseForV2.py` must be available on the Python path.

## Synthetic Dataset Generation

Current CLI usage:

```bash
python training_scripts/analytical_analysis/generate_synthetic_dataset_v2.py \
  --n_configs 100 \
  --output_dir training_scripts/analytical_analysis/data_v2 \
  --conservativeness random \
  --time_revolutions 100
```

### Current Arguments

- `--n_configs`: number of parameter configurations [default: `100`]
- `--n_workers`: number of parallel workers [default: CPU count]
- `--output_dir`: destination folder [default: `training_scripts/analytical_analysis/data_v2`]
- `--conservativeness`: one of `very_conservative`, `conservative`, `moderate`, `aggressive`, `very_aggressive`, `random`, `mass_constrained` [default: `random`]
- `--time_revolutions`: revolutions per simulation [default: `100`]

### Generated Dataset Layout

The generator writes one folder per simulation under the output directory, typically with names such as `simulation_000000`, plus metadata summarizing the dataset configuration.

## NLLS And Sensitivity Experiments

Run the experiment suite with:

```bash
python training_scripts/analytical_analysis/nlls_baseline_and_sensitivity_experiments.py
```

The current script does not expose a CLI parser. It runs the full experiment suite directly when executed.

### What It Currently Does

1. Runs an NLLS baseline over the 9-parameter inverse problem.
2. Runs multi-scenario sensitivity analysis across the predefined scenarios:
   - `baseline`
   - `high_stiffness`
   - `low_damping`
   - `high_unbalance`
3. Saves CSV summaries and PNG plots under `results_experiments/`.

### Current Output Paths

- `results_experiments/data/nlls_parameter_statistics.csv`
- `results_experiments/data/nlls_overall_statistics.csv`
- `results_experiments/data/sensitivity_<scenario>.csv`
- `results_experiments/data/correlation_<scenario>.csv`
- `results_experiments/plots/sensitivity_<scenario>.png`
- `results_experiments/plots/sensitivity_comparison.png`

## Key Current Defaults

From the checked-in script state:

- `SIMULATION_REVOLUTIONS = 50`
- `POINTS_PER_REVOLUTION = 64`
- `OMEGA_RADS = 800.0`
- `N_RANDOM_STARTS = 3`
- `PERTURBATION_FACTOR = 0.5`

## Dependencies

Typical dependencies for this folder include:

- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `pandas`

You also need access to `baseForV2.py` because both main scripts import it directly.

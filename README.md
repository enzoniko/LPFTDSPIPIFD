# JOSAFAT: Physics-Informed Fault Diagnosis Framework

**Paper:** "Linking Physical Fidelity to Downstream Performance in Physics-Informed Fault Diagnosis"  
**Journal:** IEEE Access (2026)  
**Authors:** Enzo Nicolás Spotorno, Josafat Leal Filho, Antônio Augusto Fröhlich

This repository contains the complete codebase for reproducing the results presented in our IEEE Access paper on physics-informed neural networks (PINNs) for bearing fault diagnosis.

---

## 📋 Abstract

A critical question in Scientific Machine Learning is whether improving a Physics-Informed Neural Network's (PINN) physical fidelity translates into downstream task performance. We investigate this link using a three-stage diagnostic framework: PINN-based residual generation, Extreme Value Theory anomaly detection, and Siamese Neural Network (SNN) open-set recognition. To overcome optimization costs, we propose a hierarchical three-phase evaluation protocol that progressively filters models based on physical fidelity (Phase 1), linear separability (Phase 2), and downstream robustness (Phase 3). Applying this to a ring bearing system, we evaluated adaptive PINNs against rigorous, architecturally identical data-driven baselines. Results revealed a paradox: unconstrained baselines achieved superior linear separability on known faults yet over-generalized on novel ones. In contrast, PINNs selected for high physical fidelity produced residuals with lower initial separability but a significantly richer latent structure and topological compactness. We demonstrate that the physics loss acts as an inductive bias for invariance, preventing shortcut features. This results in enhanced learnability, quantified by the improvement in cluster quality, leading to a state-of-the-art OSR F1-score of 84.0%. This work empirically establishes that optimizing for physical fidelity improves system robustness and provides a tractable methodology for guiding PINN-based applications.


---

## 🗂️ Repository Structure

```
.
├── Models/                          # Core neural network architectures
│   ├── __init__.py
│   └── basicPINNv8.py               # Main PINN model used in all experiments
│
├── Data/                            # Data loading utilities
│   ├── __init__.py
│   ├── LoadData.py                  # Legacy data paths (still used by some scripts)
│   └── LoadDatav3.py                # Current data loader for v3 dataset
│
├── training_scripts/                # Stage 1: Model training
│   ├── common_utils.py              # Shared utilities for all training scripts
│   ├── pinn_preprocessing.py        # Data preprocessing pipeline
│   │
│   ├── # Adaptive Weighting Schemes (8 methods)
│   ├── relobralo_training.py        # ReLoBRaLo loss weighting
│   ├── constant_weight_pinn_training.py  # Static weight baseline
│   ├── brdr_training.py             # BRDR scheme
│   ├── alpinn_training.py           # AL-PINN scheme
│   ├── adaptive_lbpin_training.py   # lbPINN scheme
│   ├── dwpinn_training.py           # dwPINN scheme
│   ├── gradnorm_training.py         # GradNorm scheme
│   ├── pecann_training.py           # PECANN scheme
│   │
│   ├── # Data-Driven Baselines
│   ├── data_driven_training.py      # MLP baseline training
│   ├── data_driven_preprocessing_4ch.py  # 4-channel preprocessing
│   ├── data_driven_regularization_optimization.py  # HPO for MLP-Reg
│   │
│   ├── # Hyperparameter Optimization
│   ├── bayesian_hyperparameter_optimization.py  # Optuna HPO for PINNs
│   ├── synthetic_bayesian_optimization.py       # HPO for synthetic data
│   ├── train_best_models.py         # Train best models from HPO
│   ├── validate_best_models.py      # Validate trained models
│   ├── run_smoke_tests.py           # Test harness for all training scripts
│   │
│   ├── # Metrics & Analysis
│   ├── extract_comparison_metrics.py           # Tables 1-5 metrics
│   ├── extract_parameter_estimation_errors.py  # Parameter estimation analysis
│   ├── residual_distribution_analysis.py       # Appendix 2 analysis
│   ├── pinn_evt_anomaly_evaluation.py          # Stage 2 EVT evaluation (Table 6)
│   ├── pinn_paper_spectrograms.py              # Figure 5 spectrograms
│   │
│   └── analytical_analysis/         # Synthetic data experiments
│       ├── README_experiments.md
│       ├── generate_synthetic_dataset_v2.py
│       ├── data_loader_v2.py
│       ├── nlls_baseline_and_sensitivity_experiments.py
│       └── visualize_dataset_v2.py
│
├── anomaly_detection/               # Stage 2: EVT-based anomaly detection
│   ├── __init__.py, cli.py, evaluation.py, feature_extraction.py, preprocessing.py, utils.py
│   ├── methods/
│   │   ├── __init__.py, base.py, evt.py, isolation_forest.py
│   └── scripts/
│       └── run_all_models.py
│
├── siamese_analysis/                # Stage 3: Siamese Neural Networks
│   ├── __init__.py, cli.py, README.md
│   ├── data/
│   │   ├── __init__.py, dataset.py, feature_extractors.py, parallel_features.py, processor.py
│   ├── evaluation/
│   │   ├── __init__.py, evaluator.py
│   ├── models/
│   │   ├── __init__.py, siamese.py
│   └── training/
│       ├── __init__.py, randomized_search.py, trainer.py
│
├── utils/                           # Utility functions
│   ├── data_utils.py                # Data preparation utilities
│   └── residuals_utils.py           # Residual extraction utilities
│
├── main_ACCESS.tex                  # Paper manuscript source
├── README.md                        # Repository overview and reproduction guide
└── scripts/                         # Analysis & visualization scripts
    ├── extract_bayesian_raw_metrics_corrected.py  # Tables 1-4 raw metrics
    ├── visualize_bayesian_results.py              # Bayesian HPO comparison
    ├── generate_normal_residual_statistics_table.py  # Appendix residual table
    ├── osr_umap_metrics_automation.py             # Table 7 + OSR metrics
    ├── pinn_direct_umap_visualization.py          # UMAP embeddings
    ├── pinn_to_siamese_wrapper.py                 # PINN→Siamese pipeline bridge
    ├── enhanced_half_violin_plots.py              # Residual distribution plots
    ├── robust_composite_analysis.py               # Composite metrics
    ├── individual_model_metrics.py                # Per-model metrics
    ├── run_all_models_parallel.py                 # Parallel model execution
    ├── run_siamese_analysis.py                    # Siamese launcher
    ├── siamese_config.py                          # Siamese configuration
    ├── siamese_residuals.py                       # Siamese residual processing
    ├── analyze_overnight_results.py               # Overnight experiment analysis
    └── overnight_siamese_analysis.py              # Overnight Siamese runner
```

---

## 🔧 Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9+
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alekfrohlich/hybrid_pinns.git
   cd hybrid_pinns
   ```

2. **Create and activate a Python environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   This repository currently does not include a `pyproject.toml` or `requirements.txt`, so install the packages needed for the workflows you plan to run.

   ```bash
   pip install torch numpy scipy matplotlib scikit-learn tqdm optuna umap-learn pandas seaborn
   ```

   Depending on the scripts you use, you may also need optional packages such as `PyWavelets` and `tsfresh`.

4. **Quick sanity checks:**
   ```bash
   python -m anomaly_detection.cli --help
   python -m siamese_analysis.cli --help
   python training_scripts/run_smoke_tests.py --help
   ```

5. **Optional smoke test run:**

   The smoke test executes training scripts and requires synthetic data under `training_scripts/analytical_analysis/data_v2`. Real-data checks additionally require `Data/v3/X_normal_v3.pth` and `Data/v3/Y_normal_v3.pth`.

   ```bash
   python training_scripts/run_smoke_tests.py --skip-real
   ```

---

## 🚀 Reproduction Guide

### Stage 1: PINN Training & Physical Fidelity Evaluation

#### 1.1 Train Models with Adaptive Weighting Schemes

The paper evaluates 8 adaptive loss weighting schemes plus data-driven baselines:

```bash
# ReLoBRaLo (our proposed method)
python training_scripts/relobralo_training.py

# Baselines
python training_scripts/constant_weight_pinn_training.py
python training_scripts/brdr_training.py
python training_scripts/alpinn_training.py
python training_scripts/adaptive_lbpin_training.py
python training_scripts/dwpinn_training.py
python training_scripts/gradnorm_training.py
python training_scripts/pecann_training.py

# Data-driven MLP baselines
python training_scripts/data_driven_training.py
python training_scripts/data_driven_regularization_optimization.py
```

#### 1.2 Hyperparameter Optimization

```bash
# Bayesian optimization for PINN architectures
python training_scripts/bayesian_hyperparameter_optimization.py

# Train and validate best models from HPO
python training_scripts/train_best_models.py
python training_scripts/validate_best_models.py
```

#### 1.3 Extract Comparison Metrics (Tables 1-5)

```bash
python training_scripts/extract_comparison_metrics.py
```

#### 1.4 Generate Spectrograms (Figure 5)

```bash
python training_scripts/pinn_paper_spectrograms.py
```

---

### Stage 2: EVT-Based Anomaly Detection

#### 2.1 Run EVT Evaluation (Table 6)

```bash
python training_scripts/pinn_evt_anomaly_evaluation.py
```

#### 2.2 Use Anomaly Detection Module

```bash
# Run all anomaly detection methods
python -m anomaly_detection.scripts.run_all_models

# CLI usage
python -m anomaly_detection.cli --help
```

---

### Stage 3: Siamese Network for Fault Classification

#### 3.1 Prepare PINN Residuals for Siamese Pipeline

```bash
python scripts/pinn_to_siamese_wrapper.py --config scripts/siamese_config.py
```

#### 3.2 Run Siamese Analysis

```bash
# Main launcher
python scripts/run_siamese_analysis.py

# CLI with hyperparameter optimization
python -m siamese_analysis.cli --residuals residuals_dict.pth --output-dir siamese_results --num-trials 20
```

#### 3.3 Overnight Experiments

```bash
python scripts/overnight_siamese_analysis.py
python scripts/analyze_overnight_results.py
```

---

## 📊 Generating Paper Assets

### Tables

| Table | Script | Description |
|-------|--------|-------------|
| Table 1-4 | `scripts/extract_bayesian_raw_metrics_corrected.py` | Bayesian HPO raw metrics |
| Table 5 | `training_scripts/extract_comparison_metrics.py` | Model comparison metrics |
| Table 6 | `training_scripts/pinn_evt_anomaly_evaluation.py` | EVT anomaly detection results |
| Table 7 | `scripts/osr_umap_metrics_automation.py` | OSR metrics & significance tests |

### Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 5 | `training_scripts/pinn_paper_spectrograms.py` | Time-frequency spectrograms |
| Appendix 2 | `scripts/generate_normal_residual_statistics_table.py` | Residual distribution statistics |
| UMAP Embeddings | `scripts/pinn_direct_umap_visualization.py` | UMAP visualizations |
| Residual Distributions | `scripts/enhanced_half_violin_plots.py` | Half-violin plots |

### Additional Analysis Scripts

```bash
# Bayesian HPO visualization
python scripts/visualize_bayesian_results.py

# Composite analysis
python scripts/robust_composite_analysis.py

# Individual model metrics
python scripts/individual_model_metrics.py

# Parallel model execution
python scripts/run_all_models_parallel.py
```

---

## 🧪 Synthetic Data Experiments

For analytical analysis using synthetic data (Appendix):

```bash
cd training_scripts/analytical_analysis

# Generate synthetic dataset
python generate_synthetic_dataset_v2.py

# Run NLLS baseline
python nlls_baseline_and_sensitivity_experiments.py

# Visualize dataset
python visualize_dataset_v2.py
```

---

## 📝 Key Modules

### Models/basicPINNv8.py

The core PINN architecture used throughout the paper. Features:
- Configurable MLP with customizable depth/width
- Multiple activation functions (tanh, ReLU, SiLU)
- Weight initialization strategies
- Automatic differentiation for PDE residuals

### anomaly_detection/

Stage 2 EVT-based anomaly detection module:
- Extreme Value Theory threshold estimation
- Isolation Forest baseline
- Feature extraction from residuals
- Evaluation metrics (precision, recall, F1)

### siamese_analysis/

Stage 3 Siamese Neural Network module:
- Triplet-based training
- Randomized hyperparameter search
- Fault classification with cross-validation
- Embedding visualization (UMAP, t-SNE)

---

## 📦 Data

The experimental dataset consists of bearing vibration data under various fault conditions:
- Normal operation
- Imbalance faults
- Bearing faults (inner race, outer race, rolling element)
- Multiple rotation speeds

Data loading is handled by `Data/LoadDatav3.py`. Some legacy scripts still reference `Data/LoadData.py`, so keep the dataset layout compatible with both loaders when reproducing older experiments.

---

## 🔍 Code Organization Rationale

This repository follows a **three-stage diagnostic framework**:

1. **Stage 1 (training_scripts/)**: Train PINNs with different loss weighting schemes and extract physical fidelity metrics
2. **Stage 2 (anomaly_detection/)**: Use EVT to detect anomalies from PINN residuals
3. **Stage 3 (siamese_analysis/)**: Classify faults using Siamese networks on residual features

The `scripts/` directory contains analysis and visualization tools that orchestrate the three stages and generate paper assets (tables, figures). The repository root is the working directory for all commands shown above.

---

## 🤝 Contributing

This is a research codebase accompanying our IEEE Access paper. For questions or issues related to reproduction, please open an issue on GitHub.

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

This work was supported by FUNDEP grants Rota 2030/Linha VI 29271.02.01/2022.01-00 and 29271.03.01/2023.04-00.

---

## 🔗 Links

- **Paper:** https://doi.org/10.1109/ACCESS.2026.3663988


# Siamese Residual Analysis

This package implements Stage 3 of the pipeline: Siamese-network-based analysis and open-set style evaluation over residual features.

## Current Structure

- `cli.py`: main command-line entry point
- `data/dataset.py`: triplet construction and dataset utilities
- `data/processor.py`: residual preprocessing and hierarchical labeling
- `data/feature_extractors.py`: handcrafted and spectral feature extraction
- `data/parallel_features.py`: parallel feature processing helpers
- `models/siamese.py`: Siamese network model definitions
- `training/trainer.py`: training loop and fold handling
- `training/randomized_search.py`: randomized search over training/model settings
- `evaluation/evaluator.py`: reports, plots, and downstream evaluation

## Important Repo-State Note

The current `cli.py` still contains imports from `siamese_analysis_v3.*`, while this repository exposes the package under `siamese_analysis/`. This README reflects the code that is present in the repo, but the CLI may still require local import fixes before it runs end-to-end.

## CLI Usage

Use either the package entry point or the thin wrapper in `scripts/`:

```bash
python -m siamese_analysis.cli --residuals residuals_dict.pth --output-dir siamese_results
```

```bash
python scripts/run_siamese_analysis.py --residuals residuals_dict.pth --output-dir siamese_results
```

## Current CLI Arguments

### Inputs

- `--residuals`: residual `.pth` file [default: `residuals_dict.pth`]
- `--source`: `direct` or `hybrid` [default: `direct`]
- `--output-dir`: output folder [default is a long experiment-style directory name from the current script]

### Data Processing

- `--fft-length` [default: `128`]
- `--num-channels` [default: `4`]
- `--num-triplets` [default: `20000`]
- `--labels-to-use`: optional subset of labels
- `--known-classes`: list of known classes used for training
- `--include-physical`: include physical residual channels when available

### Feature Extraction

- `--advanced-features`: enabled by default in the current parser
- `--sampling-rate` [default: `50000`]
- `--include-tsfresh`
- `--wavelet` [default: `db4`]
- `--wavelet-level` [default: `3`]
- `--max-sequence-length`
- `--feature-workers`
- `--feature-batch-size` [default: `10`]

### Training And Search

- `--batch-size` [default: `256`]
- `--epochs` [default: `20`]
- `--learning-rate` [default: `0.01`]
- `--n-folds` [default: `3`]
- `--patience` [default: `10`]
- `--data-loader-workers` [default: `20`]
- `--num-trials` [default: `1`]
- `--num-top-models` [default: `1`]
- `--num-repeat-runs` [default: `1`]
- `--load-best-model-from`: reuse a previously selected configuration and weights

### Evaluation And Visualization

- `--threshold` [default: `0.5`]
- `--use-knn`: enabled by default in the current parser
- `--knn-k-values` [default: `5`]
- `--reference-samples` [default: `30`]
- `--evaluate-baselines`: enabled by default in the current parser
- `--baseline-test-size` [default: `0.3`]
- `--baseline-max-samples` [default: `500`]
- `--visualization-level` [default: `all`]
- `--tsne-perplexity` [default: `30 90 150`]
- `--umap-n-neighbors` [default: `30 50 100`]
- `--umap-min-dist` [default: `0.25 0.5 0.8`]
- `--reuse-embeddings`

### Runtime

- `--device` [default: `cuda`]
- `--seed` [default: `42`]
- `--verbose` [default: `2`]

## Behavior Summary

The current pipeline is designed to:

1. Load residual dictionaries from direct or hybrid pipelines.
2. Preprocess and label samples hierarchically.
3. Generate triplets for Siamese training.
4. Run randomized search over model/training settings.
5. Evaluate the selected configuration with baseline probes, KNN-based classification, and embedding visualizations.

## Dependencies

Typical dependencies for this package include:

- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `umap-learn`
- `pandas`

Advanced feature extraction may additionally require `PyWavelets` and `tsfresh`.
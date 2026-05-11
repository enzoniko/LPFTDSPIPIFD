# Anomaly Detection Module

This package implements Stage 2 of the pipeline: anomaly detection on residual signals produced by PINN, hybrid, or data-driven models.

## Current Structure

- `cli.py`: command-line entry point
- `preprocessing.py`: residual loading and format normalization
- `feature_extraction.py`: statistical, FFT, and optional extended features
- `evaluation.py`: grid search and metrics
- `methods/evt.py`: Extreme Value Theory detector
- `methods/isolation_forest.py`: Isolation Forest baseline
- `utils.py`: result serialization and plotting helpers

## Supported Residual Sources

The current CLI accepts:

- `direct`: direct PINN residual dictionaries
- `hybrid`: hybrid RNN-PINN residual dictionaries
- `data_driven`: data-driven model residuals

## CLI Usage

```bash
python -m anomaly_detection.cli --residuals path/to/residuals.pth --source direct
```

### Arguments

- `--residuals`: path to the residual `.pth` file (required)
- `--source`: `direct`, `hybrid`, or `data_driven` [default: `direct`]
- `--output`: JSON file for the selected best results [default: `anomaly_detection_results.json`]
- `--workers`: worker count for grid search [default: `8`]
- `--sample-rate`: sample rate used by feature extraction [default: `50000`]
- `--methods`: one or more of `evt`, `isolation_forest`, or `all` [default: `evt`]
- `--plot-dir`: directory for ROC curves and confusion matrices [default: `anomaly_detection_plots`]
- `--skip-plots`: disable plot generation

## What The CLI Does

1. Loads and preprocesses the residual file.
2. Splits the `normal` class into train and test partitions.
3. Runs grid search for the selected methods.
4. Saves best parameters and metrics to JSON.
5. Optionally writes ROC curves and confusion matrices.

## Programmatic Usage

```python
import numpy as np

from anomaly_detection import (
    EVTDetector,
    evaluate_detector,
    preprocess_residuals,
)

processed = preprocess_residuals('residuals.pth', source_type='direct')

normal_samples = processed.get('normal', [])
split_idx = int(0.8 * len(normal_samples))
normal_train = normal_samples[:split_idx]
normal_test = normal_samples[split_idx:]

test_samples = {'normal': normal_test}
for label, sequences in processed.items():
    if label != 'normal':
        test_samples[label] = sequences

detector = EVTDetector(agg_func=np.max, threshold_u_quantile=0.95, quantile=0.99)
detector.fit(normal_train)

metrics = evaluate_detector(detector, test_samples)
print(metrics)
```

## Notes

- The preprocessing code still enriches `direct` residuals with rotation-speed metadata through `Data.LoadData` when available.
- The default CLI behavior currently optimizes and runs EVT. Isolation Forest is available, but you must request it explicitly with `--methods isolation_forest` or `--methods all`.

## Dependencies

Core dependencies used by this module:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `torch`
- `pandas`

Optional feature paths may additionally require `PyWavelets` and `tsfresh`.
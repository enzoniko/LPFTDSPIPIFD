#!/usr/bin/env python3
"""
Comprehensive Siamese Results Analyzer - Level 1 Focus with Statistical Significance

This script analyzes results from the siamese analysis pipeline, focusing on
Level 1 (Data Types) only for 2 key methods:

1. KNN k=5 classification
2. Logistic Regression (linear probe)

For each method at level 1, it extracts comprehensive metrics:
- F1 Macro scores with bootstrap confidence intervals
- Expected Calibration Error (ECE) with bin details
- Open-set AUROC (where applicable)
- Per-speed performance analysis (6 speed buckets)
- Best hyperparameters and CV scores

STATISTICAL ANALYSIS FEATURES:
- Paired t-tests and Wilcoxon signed-rank tests for method comparisons
- ANOVA for model comparisons within methods
- Effect size calculations (Cohen's d)
- Multiple testing correction (Bonferroni)
- Overall method comparison across all experiments

The script automatically detects result directories and provides detailed
performance comparisons with statistical significance testing.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from statsmodels.stats.multitest import multipletests
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All possible fault classes from the dataset
ALL_FAULT_CLASSES = [
    "normal",
    "overhang_ball_fault",
    "overhang_cage_fault", 
    "overhang_outer_race_fault",
    "underhang_ball_fault",
    "underhang_cage_fault",
    "underhang_outer_race_fault",
    "horizontal_misalignment_fault",
    "vertical_misalignment_fault",
    "imbalance_fault"
]

class SiameseResultsAnalyzer:
    """Analyzer for comprehensive Siamese results including multiple methods and levels"""

    def __init__(self, base_dir: str = "final_overnight_osr_experiments"):
        """
        Initialize the analyzer

        Args:
            base_dir: Base directory containing the results
        """
        self.base_dir = base_dir

        # Data structure: {(model, num_known_classes, method, level): [(f1_score, directory_path, additional_metrics)]}
        # method can be: 'knn_k5', 'logistic_regression'
        # level can be: 1, 2
        self.results = defaultdict(list)

        # Residual statistics: {model_name: {'mae': [mae_ch1, mae_ch2, ...], 'std': [std_ch1, std_ch2, ...]}}
        self.residual_stats = {}

        # Correlation results
        self.correlation_results = {}
        
    def find_result_directories(self) -> List[str]:
        """
        Find all directories that match the result directory patterns

        Returns:
            List of directory paths
        """
        # Look for directories that contain model results
        # Pattern: {model}_{k}classes_{fault_abbrev}_{timestamp}
        all_dirs = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                # Check if this looks like a result directory (contains training.log and search_results)
                if (os.path.exists(os.path.join(item_path, "training.log")) and
                    os.path.exists(os.path.join(item_path, "search_results"))):
                    all_dirs.append(item_path)

        logger.info(f"Found {len(all_dirs)} result directories")
        if all_dirs:
            logger.debug("Directories found:")
            for d in all_dirs[:5]:  # Show first 5
                logger.debug(f"  {os.path.basename(d)}")
            if len(all_dirs) > 5:
                logger.debug(f"  ... and {len(all_dirs) - 5} more")
        return all_dirs
    
    def extract_model_info(self, directory_name: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract model name and number of known classes from directory name

        Args:
            directory_name: Name of the directory

        Returns:
            Tuple of (model_name, num_known_classes) or (None, None) if not found
        """
        # New pattern: {model}_{k}classes_{fault_abbrev}_{timestamp}
        # Model names can contain underscores and dots (e.g., residuals_data_driven_standard.pth)
        # So we look for the pattern that contains digits followed by "classes"
        match = re.search(r'^(.+?)_(\d+)classes', directory_name)
        if match:
            model_name = match.group(1)
            num_classes = int(match.group(2))
            return model_name, num_classes
        return None, None
    
    def extract_known_classes_from_log(self, log_path: str) -> Optional[List[str]]:
        """
        Extract known classes from training log
        
        Args:
            log_path: Path to the training.log file
            
        Returns:
            List of known classes or None if not found
        """
        if not os.path.exists(log_path):
            return None
            
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for the line that shows known classes
            # Pattern: "Using known classes for training: ['normal', 'fault1', 'fault2']"
            match = re.search(r"Using known classes for training:\s*\[(.*?)\]", content)
            if match:
                classes_str = match.group(1)
                # Parse the list of classes
                classes = []
                for class_match in re.finditer(r"'([^']+)'", classes_str):
                    classes.append(class_match.group(1))
                return classes
            
        except Exception as e:
            logger.warning(f"Error reading log file {log_path}: {e}")
        
        return None
    
    def parse_classification_report(self, report_path: str) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Parse classification report and extract metrics
        
        Args:
            report_path: Path to classification_report_data_types.txt
            
        Returns:
            Dictionary mapping class names to their metrics
        """
        if not os.path.exists(report_path):
            return None
            
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the classification report table
            lines = content.strip().split('\n')
            metrics = {}
            
            # Find the start of the table (after headers)
            table_started = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip header lines
                if 'precision' in line and 'recall' in line and 'f1-score' in line:
                    table_started = True
                    continue
                
                if not table_started:
                    continue
                
                # Skip summary lines (accuracy, macro avg, weighted avg)
                if any(keyword in line.lower() for keyword in ['accuracy', 'macro avg', 'weighted avg']):
                    break
                
                # Parse data lines
                parts = line.split()
                if len(parts) >= 4:
                    class_name = parts[0]
                    try:
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1_score = float(parts[3])
                        support = int(parts[4]) if len(parts) > 4 else 0
                        
                        metrics[class_name] = {
                            'precision': precision,
                            'recall': recall,
                            'f1-score': f1_score,
                            'support': support
                        }
                    except ValueError:
                        continue
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error parsing classification report {report_path}: {e}")
        
        return None
    
    def calculate_unknown_weighted_f1(self, known_classes: List[str], 
                                    classification_metrics: Dict[str, Dict[str, float]]) -> Optional[float]:
        """
        Calculate weighted average F1-score for unknown classes
        
        Args:
            known_classes: List of known class names
            classification_metrics: Metrics from classification report
            
        Returns:
            Weighted average F1-score for unknown classes
        """
        unknown_f1_scores = []
        unknown_supports = []
        
        for class_name, metrics in classification_metrics.items():
            # Check if this class is unknown (not in known_classes)
            if class_name not in known_classes:
                f1_score = metrics.get('f1-score', 0.0)
                support = metrics.get('support', 0)
                
                if support > 0:  # Only include classes with actual samples
                    unknown_f1_scores.append(f1_score)
                    unknown_supports.append(support)
        
        if not unknown_f1_scores:
            return None
        
        # Calculate weighted average
        total_support = sum(unknown_supports)
        if total_support == 0:
            return None
            
        weighted_f1 = sum(f1 * support for f1, support in zip(unknown_f1_scores, unknown_supports)) / total_support
        return weighted_f1
    
    def extract_f1_from_classification_report(self, report_path: str) -> Optional[float]:
        """
        Extract macro F1 score from classification report

        Args:
            report_path: Path to classification report file

        Returns:
            Macro F1 score or None if not found
        """
        if not os.path.exists(report_path):
            return None

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for macro avg line and extract F1 score
            lines = content.split('\n')
            for line in lines:
                if 'macro avg' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            f1_score = float(parts[-2])  # F1 score is usually the second-to-last column
                            return f1_score
                        except ValueError:
                            continue
        except Exception as e:
            logger.warning(f"Error reading classification report {report_path}: {e}")

        return None


    def extract_knn_metrics(self, knn_dir: str) -> Dict[str, Dict[str, float]]:
        """
        Extract KNN metrics for k=5 from both levels 1 and 2

        Args:
            knn_dir: Path to knn_results directory

        Returns:
            Dictionary mapping (level) to metrics
        """
        metrics = {}
        k_dir = os.path.join(knn_dir, "knn_k5")

        if not os.path.exists(k_dir):
            return metrics

        # Extract Level 1 metrics (data_types)
        report_path_l1 = os.path.join(k_dir, "classification_report_data_types.txt")
        if os.path.exists(report_path_l1):
            level1_metrics = self.extract_detailed_knn_metrics(report_path_l1)
            if level1_metrics:
                metrics[1] = level1_metrics

        # Extract Level 2 metrics (level2)
        report_path_l2 = os.path.join(k_dir, "classification_report_level2.txt")
        if os.path.exists(report_path_l2):
            level2_metrics = self.extract_detailed_knn_metrics(report_path_l2)
            if level2_metrics:
                metrics[2] = level2_metrics

        return metrics

    def extract_detailed_knn_metrics(self, report_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract detailed KNN metrics from classification report

        Args:
            report_path: Path to classification report

        Returns:
            Dictionary with extracted metrics
        """
        if not os.path.exists(report_path):
            return None

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metrics = {}

            # Extract F1 macro from classification report
            lines = content.split('\n')
            for line in lines:
                if 'macro avg' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            f1_macro = float(parts[-2])  # F1 score is usually the second-to-last column
                            metrics['f1_macro'] = f1_macro
                        except ValueError:
                            continue
                    break

            # Extract Bootstrap Confidence Intervals
            ci_match = re.search(r'Mean F1:\s*([0-9.]+).*?95%\s*CI:\s*\[([0-9.]+),\s*([0-9.]+)\]', content, re.DOTALL)
            if ci_match:
                metrics['bootstrap_mean_f1'] = float(ci_match.group(1))
                metrics['bootstrap_ci_lower'] = float(ci_match.group(2))
                metrics['bootstrap_ci_upper'] = float(ci_match.group(3))

            # Extract Expected Calibration Error
            ece_match = re.search(r'Expected Calibration Error \(ECE\):\s*([0-9.]+)', content)
            if ece_match:
                metrics['ece'] = float(ece_match.group(1))

            # Extract Open-Set AUROC
            auroc_match = re.search(r'Open-Set AUROC:\s*([0-9.]+)', content)
            if auroc_match:
                metrics['open_set_auroc'] = float(auroc_match.group(1))

            return metrics if metrics else None

        except Exception as e:
            logger.warning(f"Error extracting KNN metrics from {report_path}: {e}")
            return None

    def extract_baseline_metrics(self, baseline_dir: str) -> Dict[Tuple[str, int], Dict[str, float]]:
        """
        Extract metrics from baseline comparison CSV and individual reports

        Args:
            baseline_dir: Path to linear_probe_analysis directory

        Returns:
            Dictionary mapping (method, level) to metrics
        """
        metrics = {}

        # Extract from individual report files for detailed metrics
        for level in [1, 2]:
            level_dir = os.path.join(baseline_dir, f"level_{level}")

            if not os.path.exists(level_dir):
                continue

            # Logistic Regression
            lr_report = os.path.join(level_dir, "logistic_regression_report.txt")
            if os.path.exists(lr_report):
                lr_metrics = self.extract_logistic_regression_metrics(lr_report)
                if lr_metrics:
                    metrics[('logistic_regression', level)] = lr_metrics


        return metrics

    def extract_logistic_regression_metrics(self, report_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract detailed Logistic Regression metrics

        Args:
            report_path: Path to logistic regression report

        Returns:
            Dictionary with extracted metrics
        """
        if not os.path.exists(report_path):
            return None

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metrics = {}

            # Extract Best Parameters (for completeness)
            param_match = re.search(r'Best Parameters:\s*(\{.+\})', content)
            if param_match:
                metrics['best_parameters'] = param_match.group(1)

            # Extract Best CV Score
            cv_match = re.search(r'Best CV Score \(F1 Macro\):\s*(nan|\d+\.\d+)', content)
            if cv_match:
                val = cv_match.group(1)
                metrics['best_cv_score'] = float(val) if val != 'nan' else None

            # Extract Test F1 Macro
            f1_match = re.search(r'Test F1 Macro:\s*(\d+\.\d+)', content)
            if f1_match:
                metrics['f1_macro'] = float(f1_match.group(1))

            # Extract Bootstrap Mean F1
            bootstrap_match = re.search(r'Mean F1 \(Bootstrap\):\s*(\d+\.\d+)', content)
            if bootstrap_match:
                metrics['bootstrap_mean_f1'] = float(bootstrap_match.group(1))

            # Extract Confidence Intervals
            ci_match = re.search(r'95%\s*Confidence Interval:\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]', content)
            if ci_match:
                metrics['bootstrap_ci_lower'] = float(ci_match.group(1))
                metrics['bootstrap_ci_upper'] = float(ci_match.group(2))

            # Extract per-speed performance (detailed)
            speed_pattern = r'(\d+-\d+ Hz):\s*F1-macro=([\d.]+),\s*Accuracy=([\d.]+),\s*N=(\d+)'
            speed_matches = re.findall(speed_pattern, content)
            if speed_matches:
                metrics['per_speed_performance'] = []
                for speed_range, f1_macro, accuracy, n_samples in speed_matches:
                    metrics['per_speed_performance'].append({
                        'speed_range': speed_range,
                        'f1_macro': float(f1_macro),
                        'accuracy': float(accuracy),
                        'n_samples': int(n_samples)
                    })
                # Also add summary stats
                speed_f1_scores = [float(match[1]) for match in speed_matches]
                metrics['speed_avg_f1'] = np.mean(speed_f1_scores)
                metrics['speed_std_f1'] = np.std(speed_f1_scores)

            return metrics if 'f1_macro' in metrics else None

        except Exception as e:
            logger.warning(f"Error extracting Logistic Regression metrics from {report_path}: {e}")
            return None

    def precalculate_residual_statistics(self):
        """
        Use pre-calculated MAE and Standard deviation of each residual channel for each residuals model
        (from healthy_residuals_ranking_tables.tex)
        """
        logger.info("Using pre-calculated residual statistics...")

        # Pre-calculated MAE values from healthy_residuals_ranking_tables.tex (excluding "old" models)
        mae_data = {
            'adaptive_lbpin': [13.7352, 46.4318, 12.4865, 44.3507, 0.2488, 0.4890, 0.1899, 0.1535],
            'alpinn': [9.9468, 38.2677, 9.3683, 36.5966, 0.2789, 8.7571, 0.2364, 0.5145],
            'brdr': [0.6657, 0.7763, 0.1280, 1.2736, 9.0504, 13.8538, 0.5046, 0.3066],
            'constant_weight': [0.2693, 0.0379, 0.0252, 0.1240, 0.6073, 5.9639, 0.8603, 0.2634],
            'residuals_data_driven_reg': [1.8627, 0.3510, 0.0266, 0.6511, 1.8627, 0.3510, 0.0266, 0.6511],
            'residuals_data_driven_standard': [1.9226, 0.3147, 0.0811, 0.2354, 1.9226, 0.3147, 0.0811, 0.2354],
            'dwpinn': [18.8689, 56.4198, 14.1976, 49.2277, 0.1620, 0.2220, 0.1787, 2.5879],
            'gradnorm': [6.0422, 3.6300, 0.5243, 1.5733, 548.7162, 662.3445, 202.7312, 38.6405],
            'pecann': [6119.4868, 2597.6294, 617.8529, 3651.4194, 16759.5957, 14202.2773, 102798.9688, 163742.0156],
            'relobralo': [0.2674, 0.0423, 0.0252, 0.1277, 0.6496, 0.1287, 0.0061, 0.0119]
        }

        # Pre-calculated STD values from healthy_residuals_ranking_tables.tex (excluding "old" models)
        std_data = {
            'adaptive_lbpin': [0.3575, 0.7181, 0.1315, 0.7044, 0.1650, 0.1558, 0.2095, 0.0782],
            'alpinn': [0.4354, 1.1590, 0.2728, 1.1300, 0.1651, 0.1535, 0.1225, 0.3810],
            'brdr': [0.3166, 0.0465, 0.0302, 0.1615, 0.1615, 0.1575, 0.3134, 0.1720],
            'constant_weight': [0.3166, 0.0453, 0.0301, 0.1521, 0.1646, 0.1291, 0.2438, 0.2761],
            'residuals_data_driven_reg': [0.3226, 0.0519, 0.0322, 0.1730, 0.3226, 0.0519, 0.0322, 0.1730],
            'residuals_data_driven_standard': [0.3253, 0.0483, 0.0319, 0.1668, 0.3253, 0.0483, 0.0319, 0.1668],
            'dwpinn': [0.3224, 0.5250, 0.0719, 0.4743, 0.1642, 0.1555, 0.1967, 0.0944],
            'gradnorm': [3.7063, 2.3845, 0.6435, 1.8092, 0.2718, 0.4110, 35.0420, 34.7434],
            'pecann': [293.6779, 112.9084, 32.0254, 155.3133, 2.0708, 1.8761, 3046.8591, 2771.0713],
            'relobralo': [0.3134, 0.0452, 0.0300, 0.1551, 0.1648, 0.1559, 0.0073, 0.0060]
        }

        # Map model names to the keys used in the data structures (excluding "old" models)
        model_name_mapping = {
            'adaptive_lbpin': 'adaptive_lbpin',
            'alpinn': 'alpinn',
            'brdr': 'brdr',
            'constant_weight': 'constant_weight',
            'residuals_data_driven_reg': 'residuals_data_driven_reg',
            'residuals_data_driven_standard': 'residuals_data_driven_standard',
            'dwpinn': 'dwpinn',
            'gradnorm': 'gradnorm',
            'pecann': 'pecann',
            'relobralo': 'relobralo'
        }

        for model_key, model_name in model_name_mapping.items():
            if model_key in mae_data and model_key in std_data:
                self.residual_stats[model_name] = {
                    'mae': mae_data[model_key],
                    'std': std_data[model_key],
                    'num_channels': len(mae_data[model_key])
                }
                logger.info(f"Loaded residual stats for {model_name}: {len(mae_data[model_key])} channels")

        logger.info(f"Loaded pre-calculated residual statistics for {len(self.residual_stats)} models")

    def bootstrap_correlation(self, x_data: np.ndarray, y_data: np.ndarray, n_bootstraps: int = 1000,
                            correlation_type: str = 'pearson') -> Dict[str, float]:
        """
        Calculate correlation with bootstrap confidence intervals and p-values

        Args:
            x_data: First variable data
            y_data: Second variable data
            n_bootstraps: Number of bootstrap samples
            correlation_type: 'pearson', 'spearman', or 'kendall'

        Returns:
            Dictionary with correlation coefficient, confidence intervals, and p-value
        """
        if len(x_data) != len(y_data) or len(x_data) < 3:
            return {'correlation': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p_value': np.nan}

        # Calculate observed correlation and p-value
        if correlation_type == 'pearson':
            observed_corr, p_value = stats.pearsonr(x_data, y_data)
        elif correlation_type == 'spearman':
            observed_corr, p_value = stats.spearmanr(x_data, y_data)
        elif correlation_type == 'kendall':
            observed_corr, p_value = stats.kendalltau(x_data, y_data)
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")

        # Bootstrap
        bootstrap_corrs = []
        n_samples = len(x_data)

        for _ in range(n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]

            try:
                if correlation_type == 'pearson':
                    corr, _ = stats.pearsonr(x_boot, y_boot)
                elif correlation_type == 'spearman':
                    corr, _ = stats.spearmanr(x_boot, y_boot)
                else:  # kendall
                    corr, _ = stats.kendalltau(x_boot, y_boot)
                bootstrap_corrs.append(corr)
            except:
                continue

        if not bootstrap_corrs:
            return {'correlation': observed_corr, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p_value': p_value}

        bootstrap_corrs = np.array(bootstrap_corrs)
        ci_lower = np.percentile(bootstrap_corrs, 2.5)
        ci_upper = np.percentile(bootstrap_corrs, 97.5)

        return {
            'correlation': observed_corr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value
        }

    def calculate_channel_performance_correlations(self):
        """
        Calculate correlations between channel MAE/STD and performance metrics across models
        """
        logger.info("Calculating channel-performance correlations...")

        self.correlation_results['channel_performance'] = {}

        # Collect data across all models for cross-model correlations
        all_models_data = {
            'mae_values': [],
            'std_values': [],
            'lr_f1_means': [],
            'lr_bootstrap_means': [],
            'lr_ci_lowers': [],
            'lr_ci_uppers': [],
            'knn_f1_means': [],
            'knn_bootstrap_means': [],
            'knn_ci_lowers': [],
            'knn_ci_uppers': [],
            'model_names': []
        }

        # First, collect average performance metrics for each model
        model_performance = {}
        for key, f1_data in self.results.items():
            if key[3] != 1:  # Only level 1
                continue

            model_name = key[0]
            method = key[2]
            f1_scores = [f1 for f1, _, _ in f1_data]

            if not f1_scores:
                continue

            if model_name not in model_performance:
                model_performance[model_name] = {'lr': [], 'knn': []}

            if f1_data and len(f1_data[0]) >= 3:
                additional_metrics = f1_data[0][2]
                if method == 'logistic_regression' and additional_metrics:
                    model_performance[model_name]['lr'].append({
                        'f1': np.mean(f1_scores),
                        'bootstrap_mean': additional_metrics.get('bootstrap_mean_f1', np.mean(f1_scores)),
                        'ci_lower': additional_metrics.get('bootstrap_ci_lower', np.mean(f1_scores)),
                        'ci_upper': additional_metrics.get('bootstrap_ci_upper', np.mean(f1_scores))
                    })
                elif method == 'knn_k5' and additional_metrics:
                    model_performance[model_name]['knn'].append({
                        'f1': np.mean(f1_scores),
                        'bootstrap_mean': additional_metrics.get('bootstrap_mean_f1', np.mean(f1_scores)),
                        'ci_lower': additional_metrics.get('bootstrap_ci_lower', np.mean(f1_scores)),
                        'ci_upper': additional_metrics.get('bootstrap_ci_upper', np.mean(f1_scores))
                    })

        # Calculate average performance across different configurations for each model
        for model_name, perf_data in model_performance.items():
            if perf_data['lr'] and perf_data['knn']:
                # Average across configurations
                lr_avg = {
                    'f1': np.mean([x['f1'] for x in perf_data['lr']]),
                    'bootstrap_mean': np.mean([x['bootstrap_mean'] for x in perf_data['lr']]),
                    'ci_lower': np.mean([x['ci_lower'] for x in perf_data['lr']]),
                    'ci_upper': np.mean([x['ci_upper'] for x in perf_data['lr']])
                }
                knn_avg = {
                    'f1': np.mean([x['f1'] for x in perf_data['knn']]),
                    'bootstrap_mean': np.mean([x['bootstrap_mean'] for x in perf_data['knn']]),
                    'ci_lower': np.mean([x['ci_lower'] for x in perf_data['knn']]),
                    'ci_upper': np.mean([x['ci_upper'] for x in perf_data['knn']])
                }

                if model_name in self.residual_stats:
                    all_models_data['model_names'].append(model_name)
                    all_models_data['lr_f1_means'].append(lr_avg['f1'])
                    all_models_data['lr_bootstrap_means'].append(lr_avg['bootstrap_mean'])
                    all_models_data['lr_ci_lowers'].append(lr_avg['ci_lower'])
                    all_models_data['lr_ci_uppers'].append(lr_avg['ci_upper'])
                    all_models_data['knn_f1_means'].append(knn_avg['f1'])
                    all_models_data['knn_bootstrap_means'].append(knn_avg['bootstrap_mean'])
                    all_models_data['knn_ci_lowers'].append(knn_avg['ci_lower'])
                    all_models_data['knn_ci_uppers'].append(knn_avg['ci_upper'])

        # Only proceed if we have enough models
        if len(all_models_data['model_names']) >= 3:
            all_models_data['lr_f1_means'] = np.array(all_models_data['lr_f1_means'])
            all_models_data['lr_bootstrap_means'] = np.array(all_models_data['lr_bootstrap_means'])
            all_models_data['lr_ci_lowers'] = np.array(all_models_data['lr_ci_lowers'])
            all_models_data['lr_ci_uppers'] = np.array(all_models_data['lr_ci_uppers'])
            all_models_data['knn_f1_means'] = np.array(all_models_data['knn_f1_means'])
            all_models_data['knn_bootstrap_means'] = np.array(all_models_data['knn_bootstrap_means'])
            all_models_data['knn_ci_lowers'] = np.array(all_models_data['knn_ci_lowers'])
            all_models_data['knn_ci_uppers'] = np.array(all_models_data['knn_ci_uppers'])

            # Calculate correlations for each channel
            for ch_idx in range(8):  # 8 channels
                channel_key = f'channel_{ch_idx+1}'

                # Get MAE and STD values for this channel across models
                mae_values = []
                std_values = []
                for model_name in all_models_data['model_names']:
                    if model_name in self.residual_stats:
                        mae_values.append(self.residual_stats[model_name]['mae'][ch_idx])
                        std_values.append(self.residual_stats[model_name]['std'][ch_idx])

                mae_values = np.array(mae_values)
                std_values = np.array(std_values)

                # Calculate correlations
                correlations = {}

                # MAE correlations
                correlations['mae_lr_f1_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['lr_f1_means'], correlation_type='pearson')
                correlations['mae_lr_f1_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['lr_f1_means'], correlation_type='spearman')
                correlations['mae_lr_f1_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['lr_f1_means'], correlation_type='kendall')

                correlations['mae_lr_bootstrap_mean_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['lr_bootstrap_means'], correlation_type='pearson')
                correlations['mae_lr_bootstrap_mean_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['lr_bootstrap_means'], correlation_type='spearman')
                correlations['mae_lr_bootstrap_mean_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['lr_bootstrap_means'], correlation_type='kendall')

                correlations['mae_lr_ci_lower_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['lr_ci_lowers'], correlation_type='pearson')
                correlations['mae_lr_ci_lower_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['lr_ci_lowers'], correlation_type='spearman')
                correlations['mae_lr_ci_lower_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['lr_ci_lowers'], correlation_type='kendall')

                correlations['mae_lr_ci_upper_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['lr_ci_uppers'], correlation_type='pearson')
                correlations['mae_lr_ci_upper_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['lr_ci_uppers'], correlation_type='spearman')
                correlations['mae_lr_ci_upper_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['lr_ci_uppers'], correlation_type='kendall')

                correlations['mae_knn_f1_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['knn_f1_means'], correlation_type='pearson')
                correlations['mae_knn_f1_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['knn_f1_means'], correlation_type='spearman')
                correlations['mae_knn_f1_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['knn_f1_means'], correlation_type='kendall')

                correlations['mae_knn_bootstrap_mean_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['knn_bootstrap_means'], correlation_type='pearson')
                correlations['mae_knn_bootstrap_mean_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['knn_bootstrap_means'], correlation_type='spearman')
                correlations['mae_knn_bootstrap_mean_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['knn_bootstrap_means'], correlation_type='kendall')

                correlations['mae_knn_ci_lower_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['knn_ci_lowers'], correlation_type='pearson')
                correlations['mae_knn_ci_lower_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['knn_ci_lowers'], correlation_type='spearman')
                correlations['mae_knn_ci_lower_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['knn_ci_lowers'], correlation_type='kendall')

                correlations['mae_knn_ci_upper_pearson'] = self.bootstrap_correlation(mae_values, all_models_data['knn_ci_uppers'], correlation_type='pearson')
                correlations['mae_knn_ci_upper_spearman'] = self.bootstrap_correlation(mae_values, all_models_data['knn_ci_uppers'], correlation_type='spearman')
                correlations['mae_knn_ci_upper_kendall'] = self.bootstrap_correlation(mae_values, all_models_data['knn_ci_uppers'], correlation_type='kendall')

                # STD correlations
                correlations['std_lr_f1_pearson'] = self.bootstrap_correlation(std_values, all_models_data['lr_f1_means'], correlation_type='pearson')
                correlations['std_lr_f1_spearman'] = self.bootstrap_correlation(std_values, all_models_data['lr_f1_means'], correlation_type='spearman')
                correlations['std_lr_f1_kendall'] = self.bootstrap_correlation(std_values, all_models_data['lr_f1_means'], correlation_type='kendall')

                correlations['std_lr_bootstrap_mean_pearson'] = self.bootstrap_correlation(std_values, all_models_data['lr_bootstrap_means'], correlation_type='pearson')
                correlations['std_lr_bootstrap_mean_spearman'] = self.bootstrap_correlation(std_values, all_models_data['lr_bootstrap_means'], correlation_type='spearman')
                correlations['std_lr_bootstrap_mean_kendall'] = self.bootstrap_correlation(std_values, all_models_data['lr_bootstrap_means'], correlation_type='kendall')

                correlations['std_lr_ci_lower_pearson'] = self.bootstrap_correlation(std_values, all_models_data['lr_ci_lowers'], correlation_type='pearson')
                correlations['std_lr_ci_lower_spearman'] = self.bootstrap_correlation(std_values, all_models_data['lr_ci_lowers'], correlation_type='spearman')
                correlations['std_lr_ci_lower_kendall'] = self.bootstrap_correlation(std_values, all_models_data['lr_ci_lowers'], correlation_type='kendall')

                correlations['std_lr_ci_upper_pearson'] = self.bootstrap_correlation(std_values, all_models_data['lr_ci_uppers'], correlation_type='pearson')
                correlations['std_lr_ci_upper_spearman'] = self.bootstrap_correlation(std_values, all_models_data['lr_ci_uppers'], correlation_type='spearman')
                correlations['std_lr_ci_upper_kendall'] = self.bootstrap_correlation(std_values, all_models_data['lr_ci_uppers'], correlation_type='kendall')

                correlations['std_knn_f1_pearson'] = self.bootstrap_correlation(std_values, all_models_data['knn_f1_means'], correlation_type='pearson')
                correlations['std_knn_f1_spearman'] = self.bootstrap_correlation(std_values, all_models_data['knn_f1_means'], correlation_type='spearman')
                correlations['std_knn_f1_kendall'] = self.bootstrap_correlation(std_values, all_models_data['knn_f1_means'], correlation_type='kendall')

                correlations['std_knn_bootstrap_mean_pearson'] = self.bootstrap_correlation(std_values, all_models_data['knn_bootstrap_means'], correlation_type='pearson')
                correlations['std_knn_bootstrap_mean_spearman'] = self.bootstrap_correlation(std_values, all_models_data['knn_bootstrap_means'], correlation_type='spearman')
                correlations['std_knn_bootstrap_mean_kendall'] = self.bootstrap_correlation(std_values, all_models_data['knn_bootstrap_means'], correlation_type='kendall')

                correlations['std_knn_ci_lower_pearson'] = self.bootstrap_correlation(std_values, all_models_data['knn_ci_lowers'], correlation_type='pearson')
                correlations['std_knn_ci_lower_spearman'] = self.bootstrap_correlation(std_values, all_models_data['knn_ci_lowers'], correlation_type='spearman')
                correlations['std_knn_ci_lower_kendall'] = self.bootstrap_correlation(std_values, all_models_data['knn_ci_lowers'], correlation_type='kendall')

                correlations['std_knn_ci_upper_pearson'] = self.bootstrap_correlation(std_values, all_models_data['knn_ci_uppers'], correlation_type='pearson')
                correlations['std_knn_ci_upper_spearman'] = self.bootstrap_correlation(std_values, all_models_data['knn_ci_uppers'], correlation_type='spearman')
                correlations['std_knn_ci_upper_kendall'] = self.bootstrap_correlation(std_values, all_models_data['knn_ci_uppers'], correlation_type='kendall')

                self.correlation_results['channel_performance'][channel_key] = correlations

        logger.info("Completed channel-performance correlation calculations")

    def calculate_method_performance_correlations(self):
        """
        Calculate correlations between LR F1 and KNN F1 performance metrics
        """
        logger.info("Calculating method performance correlations...")

        self.correlation_results['method_performance'] = {}

        # Collect data across all models and known_classes combinations
        lr_f1_scores = []
        lr_bootstrap_means = []
        lr_ci_lowers = []
        lr_ci_uppers = []
        knn_f1_scores = []
        knn_bootstrap_means = []
        knn_ci_lowers = []
        knn_ci_uppers = []

        for key, f1_data in self.results.items():
            if key[3] != 1:  # Only level 1
                continue

            f1_scores = [f1 for f1, _, _ in f1_data]
            if not f1_scores:
                continue

            mean_f1 = np.mean(f1_scores)
            method = key[2]

            # Get additional metrics from the first entry
            if f1_data and len(f1_data[0]) >= 3:
                additional_metrics = f1_data[0][2]

                if method == 'logistic_regression' and additional_metrics:
                    lr_f1_scores.append(mean_f1)
                    lr_bootstrap_means.append(additional_metrics.get('bootstrap_mean_f1', mean_f1))
                    lr_ci_lowers.append(additional_metrics.get('bootstrap_ci_lower', mean_f1))
                    lr_ci_uppers.append(additional_metrics.get('bootstrap_ci_upper', mean_f1))

                elif method == 'knn_k5' and additional_metrics:
                    knn_f1_scores.append(mean_f1)
                    knn_bootstrap_means.append(additional_metrics.get('bootstrap_mean_f1', mean_f1))
                    knn_ci_lowers.append(additional_metrics.get('bootstrap_ci_lower', mean_f1))
                    knn_ci_uppers.append(additional_metrics.get('bootstrap_ci_upper', mean_f1))

        # Convert to arrays
        lr_f1_scores = np.array(lr_f1_scores)
        lr_bootstrap_means = np.array(lr_bootstrap_means)
        lr_ci_lowers = np.array(lr_ci_lowers)
        lr_ci_uppers = np.array(lr_ci_uppers)
        knn_f1_scores = np.array(knn_f1_scores)
        knn_bootstrap_means = np.array(knn_bootstrap_means)
        knn_ci_lowers = np.array(knn_ci_lowers)
        knn_ci_uppers = np.array(knn_ci_uppers)

        # Calculate correlations only if we have matching data lengths
        min_length = min(len(lr_f1_scores), len(knn_f1_scores))
        if min_length >= 3:
            lr_f1_scores = lr_f1_scores[:min_length]
            lr_bootstrap_means = lr_bootstrap_means[:min_length]
            lr_ci_lowers = lr_ci_lowers[:min_length]
            lr_ci_uppers = lr_ci_uppers[:min_length]
            knn_f1_scores = knn_f1_scores[:min_length]
            knn_bootstrap_means = knn_bootstrap_means[:min_length]
            knn_ci_lowers = knn_ci_lowers[:min_length]
            knn_ci_uppers = knn_ci_uppers[:min_length]

            correlations = {}

            # F1 correlations
            correlations['f1_pearson'] = self.bootstrap_correlation(lr_f1_scores, knn_f1_scores, correlation_type='pearson')
            correlations['f1_spearman'] = self.bootstrap_correlation(lr_f1_scores, knn_f1_scores, correlation_type='spearman')
            correlations['f1_kendall'] = self.bootstrap_correlation(lr_f1_scores, knn_f1_scores, correlation_type='kendall')

            # Bootstrap mean correlations
            correlations['bootstrap_mean_pearson'] = self.bootstrap_correlation(lr_bootstrap_means, knn_bootstrap_means, correlation_type='pearson')
            correlations['bootstrap_mean_spearman'] = self.bootstrap_correlation(lr_bootstrap_means, knn_bootstrap_means, correlation_type='spearman')
            correlations['bootstrap_mean_kendall'] = self.bootstrap_correlation(lr_bootstrap_means, knn_bootstrap_means, correlation_type='kendall')

            # CI lower correlations
            correlations['ci_lower_pearson'] = self.bootstrap_correlation(lr_ci_lowers, knn_ci_lowers, correlation_type='pearson')
            correlations['ci_lower_spearman'] = self.bootstrap_correlation(lr_ci_lowers, knn_ci_lowers, correlation_type='spearman')
            correlations['ci_lower_kendall'] = self.bootstrap_correlation(lr_ci_lowers, knn_ci_lowers, correlation_type='kendall')

            # CI upper correlations
            correlations['ci_upper_pearson'] = self.bootstrap_correlation(lr_ci_uppers, knn_ci_uppers, correlation_type='pearson')
            correlations['ci_upper_spearman'] = self.bootstrap_correlation(lr_ci_uppers, knn_ci_uppers, correlation_type='spearman')
            correlations['ci_upper_kendall'] = self.bootstrap_correlation(lr_ci_uppers, knn_ci_uppers, correlation_type='kendall')

            self.correlation_results['method_performance'] = correlations

        logger.info("Completed method performance correlation calculations")


    def process_single_directory(self, directory: str) -> bool:
        """
        Process a single results directory and extract metrics for level 1 only

        Args:
            directory: Path to the directory

        Returns:
            True if successfully processed, False otherwise
        """
        dir_name = os.path.basename(directory)
        logger.debug(f"Processing directory: {dir_name}")

        # Extract model name and number of known classes
        model_name, num_known_classes = self.extract_model_info(dir_name)
        if not model_name or num_known_classes is None:
            logger.warning(f"Could not extract model info from {dir_name}")
            return False

        success_count = 0

        # 1. Extract KNN k=5 metrics for level 1 only
        knn_dir = os.path.join(directory, "knn_results", "knn_k5")
        if os.path.exists(knn_dir):
            # Level 1 (data types)
            report_path_l1 = os.path.join(knn_dir, "classification_report_data_types.txt")
            metrics_l1 = self.extract_detailed_knn_metrics(report_path_l1)
            if metrics_l1 and 'f1_macro' in metrics_l1:
                key = (model_name, num_known_classes, 'knn_k5', 1)
                self.results[key].append((metrics_l1['f1_macro'], directory, metrics_l1))
                success_count += 1

        # 2. Extract baseline metrics for level 1 only
        baseline_dir = os.path.join(directory, "linear_probe_analysis", "level_1")
        if os.path.exists(baseline_dir):
            # Logistic Regression
            lr_report_path = os.path.join(baseline_dir, "logistic_regression_report.txt")
            lr_metrics = self.extract_logistic_regression_metrics(lr_report_path)
            if lr_metrics and 'f1_macro' in lr_metrics:
                key = (model_name, num_known_classes, 'logistic_regression', 1)
                self.results[key].append((lr_metrics['f1_macro'], directory, lr_metrics))
                success_count += 1

        logger.debug(f"Processed {dir_name}: extracted {success_count} metrics")
        return success_count > 0

    def compute_statistical_significance(self) -> Dict[str, Any]:
        """
        Compute statistical significance tests between methods and models

        Returns:
            Dictionary containing statistical test results
        """
        stats_results = {
            'method_comparisons': [],
            'model_comparisons': [],
            'overall_method_comparison': {}
        }

        # 1. Compare methods within each model and known_classes combination
        models = set(key[0] for key in self.results.keys())
        known_classes_values = set(key[1] for key in self.results.keys())

        for model in models:
            for num_known in known_classes_values:
                # Get F1 scores for both methods
                knn_scores = []
                lr_scores = []

                # Extract scores for KNN k=5
                knn_key = (model, num_known, 'knn_k5', 1)
                if knn_key in self.results:
                    knn_scores = [f1 for f1, _, _ in self.results[knn_key]]

                # Extract scores for Logistic Regression
                lr_key = (model, num_known, 'logistic_regression', 1)
                if lr_key in self.results:
                    lr_scores = [f1 for f1, _, _ in self.results[lr_key]]

                # Only compare if we have data for both methods
                if len(knn_scores) >= 2 and len(lr_scores) >= 2:  # Need at least 2 samples for basic comparison
                    try:
                        # Check if arrays are constant (all values the same)
                        knn_constant = np.allclose(knn_scores, knn_scores[0])
                        lr_constant = np.allclose(lr_scores, lr_scores[0])

                        # Paired t-test (only if we have enough samples and not all constant)
                        if len(knn_scores) >= 3 and len(lr_scores) >= 3 and not (knn_constant and lr_constant):
                            t_stat, p_value = stats.ttest_rel(knn_scores, lr_scores)
                        else:
                            # Cannot perform meaningful t-test
                            t_stat, p_value = np.nan, np.nan

                        # Wilcoxon signed-rank test (non-parametric) - can handle constant data
                        if len(knn_scores) >= 2 and len(lr_scores) >= 2:
                            w_stat, w_p_value = stats.wilcoxon(knn_scores, lr_scores)
                        else:
                            w_p_value = np.nan

                        # Effect size (Cohen's d)
                        mean_diff = np.mean(lr_scores) - np.mean(knn_scores)
                        if knn_constant and lr_constant:
                            # Both arrays are constant, effect size is undefined
                            cohens_d = 0.0
                        else:
                            pooled_std = np.sqrt((np.std(knn_scores)**2 + np.std(lr_scores)**2) / 2)
                            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                        stats_results['method_comparisons'].append({
                            'model': model,
                            'known_classes': num_known,
                            'knn_mean': np.mean(knn_scores),
                            'lr_mean': np.mean(lr_scores),
                            'mean_diff': mean_diff,
                            't_statistic': t_stat,
                            't_p_value': p_value,
                            'wilcoxon_p_value': w_p_value,
                            'cohens_d': cohens_d,
                            'sample_size': len(knn_scores),
                            'significant_ttest': p_value < 0.05 if not np.isnan(p_value) else False,
                            'significant_wilcoxon': w_p_value < 0.05 if not np.isnan(w_p_value) else False
                        })
                    except Exception as e:
                        logger.warning(f"Could not compute statistics for {model}, {num_known} classes: {e}")

        # 2. Compare models within each method and known_classes combination
        methods = ['knn_k5', 'logistic_regression']

        for method in methods:
            for num_known in known_classes_values:
                model_scores = {}

                # Collect scores for all models with this method and known_classes
                for key, f1_data in self.results.items():
                    if key[1] == num_known and key[2] == method and key[3] == 1:
                        model_name = key[0]
                        scores = [f1 for f1, _, _ in f1_data]
                        if len(scores) >= 2:  # Need at least 2 samples
                            model_scores[model_name] = scores

                # Only perform ANOVA if we have at least 3 models with data
                if len(model_scores) >= 3:
                    try:
                        # Prepare data for ANOVA
                        all_scores = []
                        group_labels = []
                        for model_name, scores in model_scores.items():
                            all_scores.extend(scores)
                            group_labels.extend([model_name] * len(scores))

                        # One-way ANOVA
                        f_stat, p_value = stats.f_oneway(*model_scores.values())

                        stats_results['model_comparisons'].append({
                            'method': method,
                            'known_classes': num_known,
                            'models_compared': list(model_scores.keys()),
                            'f_statistic': f_stat,
                            'anova_p_value': p_value,
                            'significant_anova': p_value < 0.05,
                            'model_means': {model: np.mean(scores) for model, scores in model_scores.items()}
                        })
                    except Exception as e:
                        logger.warning(f"Could not compute ANOVA for {method}, {num_known} classes: {e}")

        # 3. Overall method comparison across all experiments
        all_knn_scores = []
        all_lr_scores = []

        for key, f1_data in self.results.items():
            if key[2] == 'knn_k5' and key[3] == 1:
                all_knn_scores.extend([f1 for f1, _, _ in f1_data])
            elif key[2] == 'logistic_regression' and key[3] == 1:
                all_lr_scores.extend([f1 for f1, _, _ in f1_data])

        if len(all_knn_scores) >= 10 and len(all_lr_scores) >= 10:  # Need substantial sample sizes
            try:
                # Independent t-test (not paired since different experimental conditions)
                t_stat, p_value = stats.ttest_ind(all_knn_scores, all_lr_scores)

                # Mann-Whitney U test (non-parametric)
                u_stat, mw_p_value = stats.mannwhitneyu(all_knn_scores, all_lr_scores, alternative='two-sided')

                # Effect size
                mean_diff = np.mean(all_lr_scores) - np.mean(all_knn_scores)
                cohens_d = mean_diff / np.sqrt((np.var(all_knn_scores) + np.var(all_lr_scores)) / 2)

                stats_results['overall_method_comparison'] = {
                    'knn_mean': np.mean(all_knn_scores),
                    'lr_mean': np.mean(all_lr_scores),
                    'mean_diff': mean_diff,
                    'knn_n': len(all_knn_scores),
                    'lr_n': len(all_lr_scores),
                    't_statistic': t_stat,
                    't_p_value': p_value,
                    'mann_whitney_p_value': mw_p_value,
                    'cohens_d': cohens_d,
                    'significant_ttest': p_value < 0.05,
                    'significant_mannwhitney': mw_p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Could not compute overall method comparison: {e}")

        return stats_results

    def analyze_all_directories(self):
        """Process all result directories"""
        directories = self.find_result_directories()

        successful = 0
        failed = 0

        for directory in directories:
            if self.process_single_directory(directory):
                successful += 1
            else:
                failed += 1

        logger.info(f"Processed {successful} directories successfully, {failed} failed")

        # Pre-calculate residual statistics
        self.precalculate_residual_statistics()

        # Compute statistical significance
        self.stats_results = self.compute_statistical_significance()

        # Calculate correlations
        self.calculate_channel_performance_correlations()
        self.calculate_method_performance_correlations()
    
    def print_results(self):
        """Print the comprehensive analysis results for level 1 only"""
        if not self.results:
            print("No results found!")
            return

        print("\n" + "="*100)
        print("COMPREHENSIVE SIAMESE ANALYSIS RESULTS")
        print("2 Methods, Level 1 (Data Types), Full Metrics")
        print("="*100)

        # Get all unique models and methods (only level 1)
        models = sorted(set(key[0] for key in self.results.keys()))
        methods = ['knn_k5', 'logistic_regression']  # Focus on these 2 methods
        levels = [1]  # Only level 1

        print(f"Found {len(models)} models, {len(methods)} methods, level 1 only")

        # Print results by model and method
        for model in models:
            print(f"\n{'='*80}")
            print(f"MODEL: {model.upper()}")
            print(f"{'='*80}")

            # Get all results for this model
            model_results = {k: v for k, v in self.results.items() if k[0] == model}

            if not model_results:
                print("   No results available")
                continue

            # Group by method
            for method in methods:
                method_results = {k: v for k, v in model_results.items() if k[2] == method}

                if not method_results:
                    continue

                print(f"\nMETHOD: {method.upper().replace('_', ' ')}")
                print("-" * 60)

                # Level 1 (Data Types) only
                level_results = {k: v for k, v in method_results.items() if k[3] == 1}

                if not level_results:
                    continue

                print(f"\n  Level 1 (Data Types):")

                # Group by number of known classes
                known_classes_groups = {}
                for key, values in level_results.items():
                    num_known = key[1]
                    if num_known not in known_classes_groups:
                        known_classes_groups[num_known] = []
                    known_classes_groups[num_known].extend(values)

                # Sort by number of known classes
                for num_known in sorted(known_classes_groups.keys()):
                    f1_data = known_classes_groups[num_known]
                    if not f1_data:
                        continue

                    # Extract F1 scores and additional metrics
                    f1_scores = [f1 for f1, _, _ in f1_data]
                    directories = [dir_path for _, dir_path, _ in f1_data]
                    additional_metrics = [metrics for _, _, metrics in f1_data]

                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    min_f1 = np.min(f1_scores)
                    max_f1 = np.max(f1_scores)
                    median_f1 = np.median(f1_scores)
                    count = len(f1_scores)

                    print(f"    {num_known} Known Classes: F1 = {mean_f1:.4f} ± {std_f1:.4f} (n={count})")
                    print(f"       Range: {min_f1:.4f} - {max_f1:.4f}, Median: {median_f1:.4f}")

                    # Show detailed metrics based on method
                    if method == 'knn_k5':
                        self._print_knn_metrics(additional_metrics)
                    elif method == 'logistic_regression':
                        self._print_logistic_regression_metrics(additional_metrics)

                    # Find best and worst directories
                    min_idx = np.argmin(f1_scores)
                    max_idx = np.argmax(f1_scores)
                    min_dir = os.path.basename(directories[min_idx])
                    max_dir = os.path.basename(directories[max_idx])
                    print(f"       Best:  {max_dir} (F1={max_f1:.4f})")
                    print(f"       Worst: {min_dir} (F1={min_f1:.4f})")

        # Create comprehensive summary table
        self._print_summary_table()

        # Print statistical significance results
        self._print_statistical_significance()

        print("\n" + "="*100)

    def _print_knn_metrics(self, additional_metrics):
        """Print KNN-specific metrics"""
        if not additional_metrics:
            return

        # Average across all runs
        avg_metrics = {}
        for metrics in additional_metrics:
            for metric_name, value in metrics.items():
                if metric_name != 'f1_macro' and value is not None and pd.notna(value):
                    if metric_name not in avg_metrics:
                        avg_metrics[metric_name] = []
                    avg_metrics[metric_name].append(value)

        if avg_metrics:
            print("       Detailed metrics:")
            for metric_name in ['bootstrap_mean_f1', 'ci_lower', 'ci_upper', 'ece', 'open_set_auroc']:
                if metric_name in avg_metrics and avg_metrics[metric_name]:
                    mean_val = np.mean(avg_metrics[metric_name])
                    if 'ci_' in metric_name:
                        continue  # Handle CI together
                    elif metric_name == 'bootstrap_mean_f1':
                        if 'ci_lower' in avg_metrics and 'ci_upper' in avg_metrics:
                            ci_lower = np.mean(avg_metrics['ci_lower'])
                            ci_upper = np.mean(avg_metrics['ci_upper'])
                            print(f"         Bootstrap F1: {mean_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
                        else:
                            print(f"         Bootstrap F1: {mean_val:.4f}")
                    elif metric_name == 'ece':
                        print(f"         ECE: {mean_val:.4f}")
                    elif metric_name == 'open_set_auroc':
                        print(f"         Open-set AUROC: {mean_val:.4f}")

    def _print_logistic_regression_metrics(self, additional_metrics):
        """Print Logistic Regression-specific metrics"""
        if not additional_metrics:
            return

        # Average across all runs
        avg_metrics = {}
        for metrics in additional_metrics:
            for metric_name, value in metrics.items():
                if metric_name != 'f1_macro' and value is not None and pd.notna(value):
                    if metric_name not in avg_metrics:
                        avg_metrics[metric_name] = []
                    avg_metrics[metric_name].append(value)

        if avg_metrics:
            print("       Detailed metrics:")
            for metric_name in ['bootstrap_mean_f1', 'ci_lower', 'ci_upper', 'speed_avg_f1', 'speed_std_f1']:
                if metric_name in avg_metrics and avg_metrics[metric_name]:
                    mean_val = np.mean(avg_metrics[metric_name])
                    if 'ci_' in metric_name:
                        continue  # Handle CI together
                    elif metric_name == 'bootstrap_mean_f1':
                        if 'ci_lower' in avg_metrics and 'ci_upper' in avg_metrics:
                            ci_lower = np.mean(avg_metrics['ci_lower'])
                            ci_upper = np.mean(avg_metrics['ci_upper'])
                            print(f"         Bootstrap F1: {mean_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
                        else:
                            print(f"         Bootstrap F1: {mean_val:.4f}")
                    elif metric_name == 'speed_avg_f1':
                        if 'speed_std_f1' in avg_metrics:
                            std_val = np.mean(avg_metrics['speed_std_f1'])
                            print(f"         Speed F1: {mean_val:.4f} ± {std_val:.4f}")
                        else:
                            print(f"         Speed F1: {mean_val:.4f}")

    def _print_knn_metrics(self, additional_metrics):
        """Print detailed KNN metrics"""
        if not additional_metrics or not any(additional_metrics):
            return

        # Aggregate metrics across runs
        ece_values = []
        auroc_values = []
        bootstrap_means = []
        ci_lowers = []
        ci_uppers = []

        for metrics in additional_metrics:
            if metrics and 'ece' in metrics:
                ece_values.append(metrics['ece'])
            if metrics and 'open_set_auroc' in metrics:
                auroc_values.append(metrics['open_set_auroc'])
            if metrics and 'bootstrap_mean_f1' in metrics:
                bootstrap_means.append(metrics['bootstrap_mean_f1'])
            if metrics and 'bootstrap_ci_lower' in metrics:
                ci_lowers.append(metrics['bootstrap_ci_lower'])
            if metrics and 'bootstrap_ci_upper' in metrics:
                ci_uppers.append(metrics['bootstrap_ci_upper'])

        if ece_values:
            print(f"       ECE: {np.mean(ece_values):.4f} ± {np.std(ece_values):.4f}")
        if auroc_values:
            print(f"       Open-set AUROC: {np.mean(auroc_values):.4f} ± {np.std(auroc_values):.4f}")
        if bootstrap_means and ci_lowers and ci_uppers:
            print(f"       Bootstrap CI: [{np.mean(ci_lowers):.4f}, {np.mean(ci_uppers):.4f}]")

    def _print_logistic_regression_metrics(self, additional_metrics):
        """Print detailed Logistic Regression metrics"""
        if not additional_metrics or not any(additional_metrics):
            return

        # Aggregate metrics across runs
        bootstrap_means = []
        ci_lowers = []
        ci_uppers = []
        speed_avgs = []
        speed_stds = []

        for metrics in additional_metrics:
            if metrics and 'bootstrap_mean_f1' in metrics:
                bootstrap_means.append(metrics['bootstrap_mean_f1'])
            if metrics and 'bootstrap_ci_lower' in metrics:
                ci_lowers.append(metrics['bootstrap_ci_lower'])
            if metrics and 'bootstrap_ci_upper' in metrics:
                ci_uppers.append(metrics['bootstrap_ci_upper'])
            if metrics and 'speed_avg_f1' in metrics:
                speed_avgs.append(metrics['speed_avg_f1'])
            if metrics and 'speed_std_f1' in metrics:
                speed_stds.append(metrics['speed_std_f1'])

        if bootstrap_means and ci_lowers and ci_uppers:
            print(f"       Bootstrap CI: [{np.mean(ci_lowers):.4f}, {np.mean(ci_uppers):.4f}]")
        if speed_avgs:
            print(f"       Speed F1: {np.mean(speed_avgs):.4f} ± {np.mean(speed_stds):.4f}")

    def _print_summary_table(self):
        """Print a comprehensive summary table of all results"""
        print(f"\nCOMPREHENSIVE SUMMARY TABLE")
        print("-" * 120)

        # Collect all results for summary
        summary_data = []

        for (model, num_known, method, level), f1_data in self.results.items():
            if not f1_data:
                continue

            f1_scores = [f1 for f1, _, _ in f1_data]
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            min_f1 = np.min(f1_scores)
            max_f1 = np.max(f1_scores)
            median_f1 = np.median(f1_scores)
            count = len(f1_scores)

            # Only include the 2 methods for level 1
            if method in ['knn_k5', 'logistic_regression'] and level == 1:
                summary_data.append({
                    'Model': model,
                    'Known_Classes': num_known,
                    'Method': method.upper().replace('_', ' '),
                    'Level': level,
                    'Mean_F1': mean_f1,
                    'Std_F1': std_f1,
                    'Min_F1': min_f1,
                    'Max_F1': max_f1,
                    'Median_F1': median_f1,
                    'Count': count
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            # Sort by Model, Method, Level, Known_Classes
            df = df.sort_values(['Model', 'Method', 'Level', 'Known_Classes'])

            # Print with formatting
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(df.to_string(index=False, float_format='%.4f'))

        # Overall statistics
        self._print_overall_statistics()

    def _print_overall_statistics(self):
        """Print overall statistics across all results"""
        print("\n" + "="*100)
        print("OVERALL STATISTICS")

        total_results = sum(len(scores) for scores in self.results.values())
        total_combinations = len(self.results)

        print(f"   Total metrics extracted: {total_results}")
        print(f"   Unique (model, method, level, known_classes) combinations: {total_combinations}")

        # Count by categories
        models = set(key[0] for key in self.results.keys())
        methods = set(key[2] for key in self.results.keys())
        levels = set(key[3] for key in self.results.keys())

        print(f"   Models analyzed: {len(models)} ({', '.join(sorted(models))})")
        print(f"   Methods evaluated: {len(methods)} ({', '.join(sorted(methods))})")
        print(f"   Hierarchy levels: {len(levels)} ({', '.join(f'L{l}' for l in sorted(levels))})")

        if total_results > 0:
            # Overall F1 statistics
            all_f1_data = [(f1, path) for f1_list in self.results.values() for f1, path, _ in f1_list]
            all_f1_scores = [f1 for f1, _ in all_f1_data]

            overall_mean = np.mean(all_f1_scores)
            overall_std = np.std(all_f1_scores)
            overall_min = np.min(all_f1_scores)
            overall_max = np.max(all_f1_scores)
            overall_median = np.median(all_f1_scores)

            print(f"\n   Overall F1 Statistics:")
            print(f"   Mean: {overall_mean:.4f} ± {overall_std:.4f}")
            print(f"   Range: {overall_min:.4f} - {overall_max:.4f}")
            print(f"   Median: {overall_median:.4f}")

            # Find best and worst performing combinations
            all_combinations = []
            for key, f1_data in self.results.items():
                if f1_data:
                    f1_scores = [f1 for f1, _, _ in f1_data]
                    mean_f1 = np.mean(f1_scores)
                    all_combinations.append((key, mean_f1, f1_data[0][1]))  # key, mean_f1, example_path

            if all_combinations:
                # Best combination
                best_combo, best_f1, best_path = max(all_combinations, key=lambda x: x[1])
                best_model, best_known, best_method, best_level = best_combo

                # Worst combination
                worst_combo, worst_f1, worst_path = min(all_combinations, key=lambda x: x[1])
                worst_model, worst_known, worst_method, worst_level = worst_combo

                print(f"\n   Best performing combination:")
                print(f"   {best_model} + {best_method} (Level {best_level}, {best_known} known classes): F1 = {best_f1:.4f}")
                print(f"   Example: {os.path.basename(best_path)}")

                print(f"\n   Worst performing combination:")
                print(f"   {worst_model} + {worst_method} (Level {worst_level}, {worst_known} known classes): F1 = {worst_f1:.4f}")
                print(f"   Example: {os.path.basename(worst_path)}")

        print("="*100)

    def _print_statistical_significance(self):
        """Print statistical significance analysis results"""
        if not hasattr(self, 'stats_results') or not self.stats_results:
            return

        print(f"\nSTATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*100)

        # 1. Method comparisons within models
        if self.stats_results['method_comparisons']:
            print(f"\nMETHOD COMPARISONS (KNN k=5 vs Logistic Regression)")
            print("-" * 90)

            # Apply multiple testing correction
            p_values = [comp['t_p_value'] for comp in self.stats_results['method_comparisons']]
            if p_values:
                _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

                for i, comp in enumerate(self.stats_results['method_comparisons']):
                    corrected_p = corrected_p_values[i]
                    sig_marker = "SIGNIFICANT" if corrected_p < 0.05 else "NOT SIGNIFICANT"

                    print(f"{sig_marker} {comp['model']} ({comp['known_classes']} classes):")
                    print(f"   KNN: {comp['knn_mean']:.4f}, LR: {comp['lr_mean']:.4f}")
                    print(f"   Mean diff: {comp['mean_diff']:+.4f}, Cohen's d: {comp['cohens_d']:+.3f}")

                    if not np.isnan(comp['t_p_value']):
                        print(f"   t-test: t={comp['t_statistic']:.3f}, p={comp['t_p_value']:.4f} (corrected: {corrected_p:.4f})")
                    else:
                        print(f"   t-test: Cannot compute (insufficient variation or sample size)")

                    if not np.isnan(comp['wilcoxon_p_value']):
                        print(f"   Wilcoxon: p={comp['wilcoxon_p_value']:.4f}")
                    else:
                        print(f"   Wilcoxon: Cannot compute")
                    print()

        # 2. Model comparisons within methods
        if self.stats_results['model_comparisons']:
            print(f"\nMODEL COMPARISONS WITHIN METHODS")
            print("-" * 90)

            for comp in self.stats_results['model_comparisons']:
                sig_marker = "SIGNIFICANT" if comp['significant_anova'] else "NOT SIGNIFICANT"

                print(f"{sig_marker} {comp['method'].upper().replace('_', ' ')} ({comp['known_classes']} classes):")
                print(f"   Models: {', '.join(comp['models_compared'])}")
                print(f"   ANOVA: F={comp['f_statistic']:.3f}, p={comp['anova_p_value']:.4f}")

                # Show model rankings
                sorted_models = sorted(comp['model_means'].items(), key=lambda x: x[1], reverse=True)
                print("   Rankings:")
                for rank, (model, mean_f1) in enumerate(sorted_models, 1):
                    print(f"     {rank}. {model}: {mean_f1:.4f}")
                print()

        # 3. Overall method comparison
        if self.stats_results['overall_method_comparison']:
            print(f"\nOVERALL METHOD COMPARISON")
            print("-" * 90)

            comp = self.stats_results['overall_method_comparison']
            sig_marker = "SIGNIFICANT" if comp['significant_ttest'] else "NOT SIGNIFICANT"

            print(f"{sig_marker} Across all experiments:")
            print(f"   KNN k=5: {comp['knn_mean']:.4f} (n={comp['knn_n']})")
            print(f"   Logistic Regression: {comp['lr_mean']:.4f} (n={comp['lr_n']})")
            print(f"   Mean difference: {comp['mean_diff']:+.4f}")
            print(f"   Effect size (Cohen's d): {comp['cohens_d']:+.3f}")
            print(f"   t-test: t={comp['t_statistic']:.3f}, p={comp['t_p_value']:.4f}")
            print(f"   Mann-Whitney U: p={comp['mann_whitney_p_value']:.4f}")

            # Interpret effect size
            d = abs(comp['cohens_d'])
            if d < 0.2:
                effect = "negligible"
            elif d < 0.5:
                effect = "small"
            elif d < 0.8:
                effect = "medium"
            else:
                effect = "large"

            print(f"   Effect size interpretation: {effect} ({d:.3f})")

        print("="*100)

        # Print correlation results
        self._print_correlation_results()

    def _print_correlation_results(self):
        """Print correlation analysis results"""
        if not hasattr(self, 'correlation_results') or not self.correlation_results:
            return

        print(f"\nCORRELATION ANALYSIS RESULTS")
        print("="*100)

        # Print residual statistics summary
        if self.residual_stats:
            print(f"\nRESIDUAL STATISTICS SUMMARY")
            print("-" * 80)
            print(f"{'Model':<25} {'Channels':<8} {'MAE (mean±std)':<15} {'STD (mean±std)':<15}")
            print("-" * 80)

            for model_name, stats in self.residual_stats.items():
                mae_values = stats['mae']
                std_values = stats['std']
                mae_mean = np.mean(mae_values)
                mae_std = np.std(mae_values)
                std_mean = np.mean(std_values)
                std_std = np.std(std_values)

                print(f"{model_name:<25} {stats['num_channels']:<8} {mae_mean:.4f}±{mae_std:.4f}     {std_mean:.4f}±{std_std:.4f}")
            print()

        # Print channel-performance correlations
        if 'channel_performance' in self.correlation_results and self.correlation_results['channel_performance']:
            print(f"\nCHANNEL-PERFORMANCE CORRELATIONS")
            print("-" * 120)
            print(f"Correlations between channel MAE/STD and F1 performance metrics across models (with 95% bootstrap CI)")
            print(f"Significance indicators: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05, † CI excludes zero")
            print("-" * 120)

            for channel_key, channel_correlations in self.correlation_results['channel_performance'].items():
                channel_name = channel_key.upper().replace('_', ' ')
                print(f"\n{channel_name}:")
                print("-" * 150)
                print(f"  {'Comparison':<30} {'Pearson':<12} {'Pearson 95% CI':<18} {'p-val':<6} {'Sig':<3} {'CI Sig':<6} {'Spearman':<12} {'Spearman 95% CI':<18} {'p-val':<6} {'Sig':<3} {'CI Sig':<6} {'Kendall':<12} {'Kendall 95% CI':<18} {'p-val':<6} {'Sig':<3} {'CI Sig':<6}")
                print(f"  {'-'*30} {'-'*12} {'-'*18} {'-'*6} {'-'*3} {'-'*6} {'-'*12} {'-'*18} {'-'*6} {'-'*3} {'-'*6} {'-'*12} {'-'*18} {'-'*6} {'-'*3} {'-'*6}")

                metrics = [
                    ('mae_lr_f1', 'MAE vs LR F1'),
                    ('mae_lr_bootstrap_mean', 'MAE vs LR Bootstrap Mean'),
                    ('mae_lr_ci_lower', 'MAE vs LR CI Lower'),
                    ('mae_lr_ci_upper', 'MAE vs LR CI Upper'),
                    ('mae_knn_f1', 'MAE vs KNN F1'),
                    ('mae_knn_bootstrap_mean', 'MAE vs KNN Bootstrap Mean'),
                    ('mae_knn_ci_lower', 'MAE vs KNN CI Lower'),
                    ('mae_knn_ci_upper', 'MAE vs KNN CI Upper'),
                    ('std_lr_f1', 'STD vs LR F1'),
                    ('std_lr_bootstrap_mean', 'STD vs LR Bootstrap Mean'),
                    ('std_lr_ci_lower', 'STD vs LR CI Lower'),
                    ('std_lr_ci_upper', 'STD vs LR CI Upper'),
                    ('std_knn_f1', 'STD vs KNN F1'),
                    ('std_knn_bootstrap_mean', 'STD vs KNN Bootstrap Mean'),
                    ('std_knn_ci_lower', 'STD vs KNN CI Lower'),
                    ('std_knn_ci_upper', 'STD vs KNN CI Upper')
                ]

                for metric_key, metric_name in metrics:
                    # Get all three correlation types
                    pearson_key = f"{metric_key}_pearson"
                    spearman_key = f"{metric_key}_spearman"
                    kendall_key = f"{metric_key}_kendall"

                    # Helper function to format correlation data
                    def format_corr_data(corr_key):
                        if corr_key in channel_correlations:
                            corr_data = channel_correlations[corr_key]
                            corr_str = f"{corr_data['correlation']:.3f}" if not np.isnan(corr_data['correlation']) else "N/A"
                            ci_str = f"[{corr_data['ci_lower']:.3f}, {corr_data['ci_upper']:.3f}]" if not np.isnan(corr_data['ci_lower']) else "[N/A,N/A]"
                            p_val_str = f"{corr_data.get('p_value', np.nan):.3f}" if not np.isnan(corr_data.get('p_value', np.nan)) else "N/A"

                            # Add significance indicators
                            sig_str = ""
                            if not np.isnan(corr_data.get('p_value', np.nan)):
                                p_val = corr_data['p_value']
                                if p_val < 0.001:
                                    sig_str = "***"
                                elif p_val < 0.01:
                                    sig_str = "**"
                                elif p_val < 0.05:
                                    sig_str = "*"
                                else:
                                    sig_str = "ns"

                            # Check if CI excludes zero (bootstrap significance)
                            ci_sig = ""
                            if (not np.isnan(corr_data.get('ci_lower', np.nan)) and
                                not np.isnan(corr_data.get('ci_upper', np.nan))):
                                if corr_data['ci_lower'] > 0 or corr_data['ci_upper'] < 0:
                                    ci_sig = "†"
                                else:
                                    ci_sig = "ns"

                            return corr_str, ci_str, p_val_str, sig_str, ci_sig
                        else:
                            return "N/A", "[N/A,N/A]", "N/A", "N/A", "N/A"

                    p_corr_str, p_ci_str, p_pval_str, p_sig_str, p_ci_sig = format_corr_data(pearson_key)
                    s_corr_str, s_ci_str, s_pval_str, s_sig_str, s_ci_sig = format_corr_data(spearman_key)
                    k_corr_str, k_ci_str, k_pval_str, k_sig_str, k_ci_sig = format_corr_data(kendall_key)

                    print(f"  {metric_name:<30} {p_corr_str:<12} {p_ci_str:<18} {p_pval_str:<6} {p_sig_str:<3} {p_ci_sig:<6} {s_corr_str:<12} {s_ci_str:<18} {s_pval_str:<6} {s_sig_str:<3} {s_ci_sig:<6} {k_corr_str:<12} {k_ci_str:<18} {k_pval_str:<6} {k_sig_str:<3} {k_ci_sig:<6}")

        # Print method performance correlations
        if 'method_performance' in self.correlation_results and self.correlation_results['method_performance']:
            print(f"\n\nMETHOD PERFORMANCE CORRELATIONS")
            print("-" * 190)
            print(f"Correlations between Logistic Regression and KNN k=5 performance metrics")
            print(f"Significance indicators: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05, † CI excludes zero")
            print("-" * 190)

            correlations = self.correlation_results['method_performance']

            print(f"{'Metric':<20} {'Pearson':<12} {'Pearson 95% CI':<18} {'p-val':<6} {'Sig':<3} {'CI Sig':<6} {'Spearman':<12} {'Spearman 95% CI':<18} {'p-val':<6} {'Sig':<3} {'CI Sig':<6} {'Kendall':<12} {'Kendall 95% CI':<18} {'p-val':<6} {'Sig':<3} {'CI Sig':<6}")
            print("-" * 190)

            metrics = ['f1', 'bootstrap_mean', 'ci_lower', 'ci_upper']
            for metric in metrics:
                pearson_key = f"{metric}_pearson"
                spearman_key = f"{metric}_spearman"
                kendall_key = f"{metric}_kendall"

                # Helper function to format correlation data
                def format_method_corr_data(corr_key):
                    if corr_key in correlations:
                        corr_data = correlations[corr_key]
                        corr_str = f"{corr_data['correlation']:.3f}" if not np.isnan(corr_data['correlation']) else "N/A"
                        ci_str = f"[{corr_data['ci_lower']:.3f}, {corr_data['ci_upper']:.3f}]" if not np.isnan(corr_data['ci_lower']) else "[N/A, N/A]"
                        p_val_str = f"{corr_data.get('p_value', np.nan):.3f}" if not np.isnan(corr_data.get('p_value', np.nan)) else "N/A"

                        # Significance indicators
                        sig_str = ""
                        if not np.isnan(corr_data.get('p_value', np.nan)):
                            p_val = corr_data['p_value']
                            if p_val < 0.001:
                                sig_str = "***"
                            elif p_val < 0.01:
                                sig_str = "**"
                            elif p_val < 0.05:
                                sig_str = "*"
                            else:
                                sig_str = "ns"

                        ci_sig = ""
                        if (not np.isnan(corr_data.get('ci_lower', np.nan)) and
                            not np.isnan(corr_data.get('ci_upper', np.nan))):
                            if corr_data['ci_lower'] > 0 or corr_data['ci_upper'] < 0:
                                ci_sig = "†"
                            else:
                                ci_sig = "ns"

                        return corr_str, ci_str, p_val_str, sig_str, ci_sig
                    else:
                        return "N/A", "[N/A, N/A]", "N/A", "N/A", "N/A"

                p_corr_str, p_ci_str, p_pval_str, p_sig_str, p_ci_sig = format_method_corr_data(pearson_key)
                s_corr_str, s_ci_str, s_pval_str, s_sig_str, s_ci_sig = format_method_corr_data(spearman_key)
                k_corr_str, k_ci_str, k_pval_str, k_sig_str, k_ci_sig = format_method_corr_data(kendall_key)

                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:<20} {p_corr_str:<12} {p_ci_str:<18} {p_pval_str:<6} {p_sig_str:<3} {p_ci_sig:<6} {s_corr_str:<12} {s_ci_str:<18} {s_pval_str:<6} {s_sig_str:<3} {s_ci_sig:<6} {k_corr_str:<12} {k_ci_str:<18} {k_pval_str:<6} {k_sig_str:<3} {k_ci_sig:<6}")

        print("="*100)


def main():
    """Main function"""
    print("Starting Comprehensive Siamese Results Analysis...")

    analyzer = SiameseResultsAnalyzer()
    analyzer.analyze_all_directories()
    analyzer.print_results()

    print("Analysis complete!")

if __name__ == "__main__":
    main()

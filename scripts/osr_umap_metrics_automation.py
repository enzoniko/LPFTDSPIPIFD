#!/usr/bin/env python3
"""
OSR UMAP Metrics Automation Script

This script automates the process of running pinn_direct_umap_visualization.py
in embeddings mode for all experiments in final_overnight_osr_experiments/,
collecting raw features metrics and averaged UMAP features metrics for level 1 labels,
and performing statistical analysis including significance tests.

Usage:
    python osr_umap_metrics_automation.py --experiments-dir final_overnight_osr_experiments/ --output-dir osr_umap_metrics_results/
"""

import os
import sys
import json
import subprocess
import re
import argparse
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Container for metrics from a single experiment"""
    model_name: str
    experiment_name: str

    # Raw features metrics (level 1)
    raw_silhouette: float
    raw_calinski: float
    raw_davies: float

    # UMAP averaged metrics (level 1)
    umap_silhouette_mean: float
    umap_silhouette_std: float
    umap_silhouette_count: int

    umap_calinski_mean: float
    umap_calinski_std: float
    umap_calinski_count: int

    umap_davies_mean: float
    umap_davies_std: float
    umap_davies_count: int


@dataclass
class ModelStatistics:
    """Statistics for a model across all its experiments"""
    model_name: str
    n_experiments: int

    # Raw features statistics
    raw_silhouette_mean: float
    raw_silhouette_std: float
    raw_calinski_mean: float
    raw_calinski_std: float
    raw_davies_mean: float
    raw_davies_std: float

    # UMAP features statistics (averages of means and stds)
    umap_silhouette_mean_avg: float
    umap_silhouette_std_avg: float
    umap_calinski_mean_avg: float
    umap_calinski_std_avg: float
    umap_davies_mean_avg: float
    umap_davies_std_avg: float


def extract_model_name_from_experiment(experiment_path: str) -> str:
    """
    Extract model name from experiment path.
    Examples:
    - adaptive_lbpin_3classes_horizont_overhang_1114_1500 -> adaptive_lbpin
    - alpinn_4classes_overhang_overhang_underhan_1115_0045 -> alpinn
    """
    basename = os.path.basename(experiment_path)

    # Handle special cases
    if basename.startswith('adaptive_lbpin'):
        return 'adaptive_lbpin'
    elif basename.startswith('residuals_data_driven_standard'):
        return 'data_driven_standard'
    elif basename.startswith('residuals_data_driven_reg'):
        return 'data_driven_reg'
    else:
        # Extract prefix until first underscore after method name
        parts = basename.split('_')
        if len(parts) >= 2:
            # Check for specific multi-word model names first
            if '_'.join(parts[:2]) in ['constant_weight']:
                return 'constant_weight'
            # Then check for common single-word model names
            elif parts[0] in ['alpinn', 'brdr', 'constant', 'dwpinn', 'gradnorm', 'pecann', 'relobralo']:
                return parts[0]
        return parts[0] if parts else basename


def run_umap_visualization_for_experiment(experiment_path: str, output_base_dir: str) -> bool:
    """
    Run pinn_direct_umap_visualization.py for a single experiment in embeddings mode

    Args:
        experiment_path: Path to experiment directory containing embeddings.pkl
        output_base_dir: Base directory for outputs

    Returns:
        True if successful, False otherwise
    """
    experiment_name = os.path.basename(experiment_path)
    embeddings_path = os.path.join(experiment_path, 'embeddings.pkl')
    output_dir = os.path.join(output_base_dir, experiment_name)

    # Check if embeddings file exists
    if not os.path.exists(embeddings_path):
        logger.warning(f"Embeddings file not found: {embeddings_path}")
        return False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, 'pinn_direct_umap_visualization.py',
        '--embeddings', embeddings_path,
        '--output-dir', output_dir,
        '--disable-multiprocessing'  # Avoid multiprocessing issues
    ]

    logger.info(f"Running UMAP visualization for {experiment_name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"Successfully processed {experiment_name}")
            return True
        else:
            logger.error(f"Failed to process {experiment_name}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout processing {experiment_name}")
        return False
    except Exception as e:
        logger.error(f"Error processing {experiment_name}: {e}")
        return False


def extract_raw_features_metrics(metrics_file_path: str) -> Optional[Dict[str, float]]:
    """
    Extract level 1 metrics from cluster_metrics_embeddings.txt

    Args:
        metrics_file_path: Path to the metrics file

    Returns:
        Dictionary with silhouette, calinski, davies scores for level 1, or None if not found
    """
    if not os.path.exists(metrics_file_path):
        return None

    try:
        with open(metrics_file_path, 'r') as f:
            content = f.read()

        # Look for the table row with "Level 1 (Groups)"
        level1_match = re.search(r'Level 1 \(Groups\)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)', content)
        if not level1_match:
            logger.warning(f"Level 1 metrics not found in {metrics_file_path}")
            return None

        # Extract metrics from the regex groups
        metrics = {
            'silhouette': float(level1_match.group(1)),
            'calinski': float(level1_match.group(2)),
            'davies': float(level1_match.group(3))
        }

        return metrics

    except Exception as e:
        logger.error(f"Error parsing {metrics_file_path}: {e}")
        return None


def extract_umap_summary_metrics(umap_file_path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Extract level 1 averaged UMAP metrics from umap_summary_latex_embeddings.txt

    Args:
        umap_file_path: Path to the UMAP summary file

    Returns:
        Dictionary with metrics containing mean, std, count for each metric type
    """
    if not os.path.exists(umap_file_path):
        return None

    try:
        with open(umap_file_path, 'r') as f:
            content = f.read()

        # Look for level 1 section
        level1_match = re.search(r'% Level 1 \(Groups\)(.*?)(?=% Level 2|\Z)', content, re.DOTALL)
        if not level1_match:
            logger.warning(f"Level 1 section not found in {umap_file_path}")
            return None

        level1_content = level1_match.group(1)

        metrics = {}

        # Extract each metric type
        metric_patterns = {
            'silhouette': r'Silhouette Score & ([0-9.\-]+) & ([0-9.\-]+) & ([0-9]+)',
            'calinski': r'Calinski-Harabasz Score & ([0-9.\-]+) & ([0-9.\-]+) & ([0-9]+)',
            'davies': r'Davies-Bouldin Score & ([0-9.\-]+) & ([0-9.\-]+) & ([0-9]+)'
        }

        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, level1_content)
            if match:
                mean_val = float(match.group(1))
                std_val = float(match.group(2))
                count = int(match.group(3))
                metrics[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'count': count
                }

        if len(metrics) == 3:
            return metrics
        else:
            logger.warning(f"Incomplete UMAP metrics in {umap_file_path}: {metrics}")
            return None

    except Exception as e:
        logger.error(f"Error parsing {umap_file_path}: {e}")
        return None


def collect_residuals_metrics(residuals_base_dir: str) -> Dict[str, ModelStatistics]:
    """
    Collect metrics from residuals_umap_visualization_outputs directory (pre-SNN metrics)

    Args:
        residuals_base_dir: Path to residuals_umap_visualization_outputs directory

    Returns:
        Dictionary mapping model names to their pre-SNN ModelStatistics
    """
    logger.info("Collecting pre-SNN metrics from residuals...")

    residuals_stats = {}

    if not os.path.exists(residuals_base_dir):
        logger.warning(f"Residuals directory not found: {residuals_base_dir}")
        return residuals_stats

    # Get all model directories
    for item in os.listdir(residuals_base_dir):
        model_dir = os.path.join(residuals_base_dir, item)
        if os.path.isdir(model_dir):
            model_name = item

            # Extract raw features metrics
            cluster_file = os.path.join(model_dir, f'cluster_metrics_features_{model_name}.txt')
            raw_metrics = extract_raw_features_metrics(cluster_file)

            # Extract UMAP features metrics
            umap_file = os.path.join(model_dir, f'umap_summary_latex_features_{model_name}.txt')
            umap_metrics = extract_umap_summary_metrics(umap_file)

            if raw_metrics is not None and umap_metrics is not None:
                # Create a single experiment's worth of metrics for this model
                # Since residuals are per-model (not per-experiment), we treat each model as having 1 "experiment"
                exp_metrics = ExperimentMetrics(
                    model_name=model_name,
                    experiment_name=f"{model_name}_residuals",
                    raw_silhouette=raw_metrics['silhouette'],
                    raw_calinski=raw_metrics['calinski'],
                    raw_davies=raw_metrics['davies'],
                    umap_silhouette_mean=umap_metrics['silhouette']['mean'],
                    umap_silhouette_std=umap_metrics['silhouette']['std'],
                    umap_silhouette_count=umap_metrics['silhouette']['count'],
                    umap_calinski_mean=umap_metrics['calinski']['mean'],
                    umap_calinski_std=umap_metrics['calinski']['std'],
                    umap_calinski_count=umap_metrics['calinski']['count'],
                    umap_davies_mean=umap_metrics['davies']['mean'],
                    umap_davies_std=umap_metrics['davies']['std'],
                    umap_davies_count=umap_metrics['davies']['count']
                )

                # Calculate statistics for this "single experiment" model
                model_stats = calculate_model_statistics([exp_metrics])
                residuals_stats[model_name] = model_stats
                logger.info(f"Collected residuals metrics for {model_name}")
            else:
                logger.warning(f"Could not collect complete metrics for {model_name}")

    logger.info(f"Collected pre-SNN metrics for {len(residuals_stats)} models")
    return residuals_stats


def collect_experiment_metrics(output_base_dir: str, experiment_name: str, model_name: str) -> Optional[ExperimentMetrics]:
    """
    Collect metrics for a single experiment

    Args:
        output_base_dir: Base output directory
        experiment_name: Name of the experiment
        model_name: Name of the model

    Returns:
        ExperimentMetrics object or None if collection failed
    """
    output_dir = os.path.join(output_base_dir, experiment_name)

    # Find the metrics files
    cluster_metrics_file = os.path.join(output_dir, f'cluster_metrics_embeddings.txt')
    umap_summary_file = os.path.join(output_dir, f'umap_summary_latex_embeddings.txt')

    # Extract raw features metrics
    raw_metrics = extract_raw_features_metrics(cluster_metrics_file)
    if raw_metrics is None:
        logger.warning(f"Could not extract raw metrics for {experiment_name}")
        return None

    # Extract UMAP summary metrics
    umap_metrics = extract_umap_summary_metrics(umap_summary_file)
    if umap_metrics is None:
        logger.warning(f"Could not extract UMAP metrics for {experiment_name}")
        return None

    try:
        return ExperimentMetrics(
            model_name=model_name,
            experiment_name=experiment_name,
            raw_silhouette=raw_metrics['silhouette'],
            raw_calinski=raw_metrics['calinski'],
            raw_davies=raw_metrics['davies'],
            umap_silhouette_mean=umap_metrics['silhouette']['mean'],
            umap_silhouette_std=umap_metrics['silhouette']['std'],
            umap_silhouette_count=umap_metrics['silhouette']['count'],
            umap_calinski_mean=umap_metrics['calinski']['mean'],
            umap_calinski_std=umap_metrics['calinski']['std'],
            umap_calinski_count=umap_metrics['calinski']['count'],
            umap_davies_mean=umap_metrics['davies']['mean'],
            umap_davies_std=umap_metrics['davies']['std'],
            umap_davies_count=umap_metrics['davies']['count']
        )
    except KeyError as e:
        logger.error(f"Missing metric key for {experiment_name}: {e}")
        return None


def calculate_model_statistics(experiment_metrics_list: List[ExperimentMetrics]) -> ModelStatistics:
    """
    Calculate statistics for a model across all its experiments

    Args:
        experiment_metrics_list: List of ExperimentMetrics for the same model

    Returns:
        ModelStatistics object
    """
    if not experiment_metrics_list:
        raise ValueError("Empty experiment metrics list")

    model_name = experiment_metrics_list[0].model_name
    n_experiments = len(experiment_metrics_list)

    # Raw features metrics
    raw_silhouettes = [exp.raw_silhouette for exp in experiment_metrics_list]
    raw_calinskis = [exp.raw_calinski for exp in experiment_metrics_list]
    raw_davies = [exp.raw_davies for exp in experiment_metrics_list]

    # UMAP metrics (averages of the means and stds)
    umap_silhouette_means = [exp.umap_silhouette_mean for exp in experiment_metrics_list]
    umap_silhouette_stds = [exp.umap_silhouette_std for exp in experiment_metrics_list]
    umap_calinski_means = [exp.umap_calinski_mean for exp in experiment_metrics_list]
    umap_calinski_stds = [exp.umap_calinski_std for exp in experiment_metrics_list]
    umap_davies_means = [exp.umap_davies_mean for exp in experiment_metrics_list]
    umap_davies_stds = [exp.umap_davies_std for exp in experiment_metrics_list]

    return ModelStatistics(
        model_name=model_name,
        n_experiments=n_experiments,
        raw_silhouette_mean=np.mean(raw_silhouettes),
        raw_silhouette_std=np.std(raw_silhouettes),
        raw_calinski_mean=np.mean(raw_calinskis),
        raw_calinski_std=np.std(raw_calinskis),
        raw_davies_mean=np.mean(raw_davies),
        raw_davies_std=np.std(raw_davies),
        umap_silhouette_mean_avg=np.mean(umap_silhouette_means),
        umap_silhouette_std_avg=np.mean(umap_silhouette_stds),
        umap_calinski_mean_avg=np.mean(umap_calinski_means),
        umap_calinski_std_avg=np.mean(umap_calinski_stds),
        umap_davies_mean_avg=np.mean(umap_davies_means),
        umap_davies_std_avg=np.mean(umap_davies_stds)
    )


def perform_significance_tests(model_stats_dict: Dict[str, ModelStatistics],
                             baseline_model: str = 'adaptive_lbpin') -> Dict[str, Dict[str, float]]:
    """
    Perform statistical significance tests comparing each model to the baseline

    Args:
        model_stats_dict: Dictionary mapping model names to ModelStatistics
        baseline_model: Name of the baseline model for comparison

    Returns:
        Dictionary with p-values for each comparison and metric
    """
    if baseline_model not in model_stats_dict:
        logger.warning(f"Baseline model '{baseline_model}' not found in results")
        return {}

    significance_results = {}

    # For now, we'll implement a simple comparison framework
    # In a real scenario, you'd collect all experiment-level metrics and do proper statistical tests

    logger.info(f"Performing significance tests with baseline: {baseline_model}")
    logger.info("Note: This is a simplified analysis. For rigorous statistical testing,")
    logger.info("you would need all experiment-level raw data for proper t-tests or ANOVA.")

    # This is a placeholder for more sophisticated statistical analysis
    # In practice, you would collect all the raw experiment metrics and perform:
    # - t-tests between models
    # - ANOVA for multiple comparisons
    # - Effect size calculations
    # - Confidence intervals

    return significance_results


def create_summary_plots(model_stats_dict: Dict[str, ModelStatistics], output_dir: str):
    """
    Create summary plots comparing models

    Args:
        model_stats_dict: Dictionary mapping model names to ModelStatistics
        output_dir: Output directory for plots
    """
    if not model_stats_dict:
        return

    # Prepare data for plotting
    models = list(model_stats_dict.keys())
    n_models = len(models)

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OSR UMAP Metrics Comparison Across Models', fontsize=16)

    metrics_data = [
        ('Raw Silhouette', 'raw_silhouette_mean', 'raw_silhouette_std'),
        ('Raw Calinski-Harabasz', 'raw_calinski_mean', 'raw_calinski_std'),
        ('Raw Davies-Bouldin', 'raw_davies_mean', 'raw_davies_std'),
        ('UMAP Silhouette Mean', 'umap_silhouette_mean_avg', 'umap_silhouette_std_avg'),
        ('UMAP Calinski-Harabasz Mean', 'umap_calinski_mean_avg', 'umap_calinski_std_avg'),
        ('UMAP Davies-Bouldin Mean', 'umap_davies_mean_avg', 'umap_davies_std_avg')
    ]

    for i, (metric_name, mean_attr, std_attr) in enumerate(metrics_data):
        ax = axes[i // 3, i % 3]

        means = [getattr(model_stats_dict[model], mean_attr) for model in models]
        stds = [getattr(model_stats_dict[model], std_attr) for model in models]

        # Create bar plot with error bars
        bars = ax.bar(range(n_models), means, yerr=stds, capsize=5,
                     color=plt.cm.tab10(np.linspace(0, 1, n_models)))
        ax.set_title(metric_name)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right')

        # Add value labels on bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            ax.text(j, mean + std + 0.01, '.3f',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Raw vs UMAP comparison
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Silhouette comparison
    raw_sil = [model_stats_dict[m].raw_silhouette_mean for m in models]
    umap_sil = [model_stats_dict[m].umap_silhouette_mean_avg for m in models]

    ax1.scatter(raw_sil, umap_sil, s=50)
    for i, model in enumerate(models):
        ax1.annotate(model, (raw_sil[i], umap_sil[i]), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Raw Features Silhouette')
    ax1.set_ylabel('UMAP Silhouette Mean')
    ax1.set_title('Raw vs UMAP Silhouette Scores')
    ax1.grid(True, alpha=0.3)

    # Calinski comparison
    raw_cal = [model_stats_dict[m].raw_calinski_mean for m in models]
    umap_cal = [model_stats_dict[m].umap_calinski_mean_avg for m in models]

    ax2.scatter(raw_cal, umap_cal, s=50)
    for i, model in enumerate(models):
        ax2.annotate(model, (raw_cal[i], umap_cal[i]), xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Raw Features Calinski-Harabasz')
    ax2.set_ylabel('UMAP Calinski-Harabasz Mean')
    ax2.set_title('Raw vs UMAP Calinski-Harabasz Scores')
    ax2.grid(True, alpha=0.3)

    # Davies comparison
    raw_dav = [model_stats_dict[m].raw_davies_mean for m in models]
    umap_dav = [model_stats_dict[m].umap_davies_mean_avg for m in models]

    ax3.scatter(raw_dav, umap_dav, s=50)
    for i, model in enumerate(models):
        ax3.annotate(model, (raw_dav[i], umap_dav[i]), xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Raw Features Davies-Bouldin')
    ax3.set_ylabel('UMAP Davies-Bouldin Mean')
    ax3.set_title('Raw vs UMAP Davies-Bouldin Scores')
    ax3.grid(True, alpha=0.3)

    # Number of experiments per model
    n_experiments = [model_stats_dict[m].n_experiments for m in models]
    ax4.bar(range(len(models)), n_experiments, color='skyblue')
    ax4.set_title('Number of Experiments per Model')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylabel('Number of Experiments')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results_to_files(model_stats_dict: Dict[str, ModelStatistics],
                         experiment_metrics_list: List[ExperimentMetrics],
                         output_dir: str):
    """
    Save all results to files

    Args:
        model_stats_dict: Dictionary mapping model names to ModelStatistics
        experiment_metrics_list: List of all ExperimentMetrics
        output_dir: Output directory
    """
    # Save model statistics to CSV
    model_stats_data = []
    for model_name, stats in model_stats_dict.items():
        model_stats_data.append({
            'model': model_name,
            'n_experiments': stats.n_experiments,
            'raw_silhouette_mean': stats.raw_silhouette_mean,
            'raw_silhouette_std': stats.raw_silhouette_std,
            'raw_calinski_mean': stats.raw_calinski_mean,
            'raw_calinski_std': stats.raw_calinski_std,
            'raw_davies_mean': stats.raw_davies_mean,
            'raw_davies_std': stats.raw_davies_std,
            'umap_silhouette_mean_avg': stats.umap_silhouette_mean_avg,
            'umap_silhouette_std_avg': stats.umap_silhouette_std_avg,
            'umap_calinski_mean_avg': stats.umap_calinski_mean_avg,
            'umap_calinski_std_avg': stats.umap_calinski_std_avg,
            'umap_davies_mean_avg': stats.umap_davies_mean_avg,
            'umap_davies_std_avg': stats.umap_davies_std_avg
        })

    model_df = pd.DataFrame(model_stats_data)
    model_df.to_csv(os.path.join(output_dir, 'model_statistics_summary.csv'), index=False)

    # Save detailed experiment metrics
    experiment_data = []
    for exp in experiment_metrics_list:
        experiment_data.append({
            'model': exp.model_name,
            'experiment': exp.experiment_name,
            'raw_silhouette': exp.raw_silhouette,
            'raw_calinski': exp.raw_calinski,
            'raw_davies': exp.raw_davies,
            'umap_silhouette_mean': exp.umap_silhouette_mean,
            'umap_silhouette_std': exp.umap_silhouette_std,
            'umap_silhouette_count': exp.umap_silhouette_count,
            'umap_calinski_mean': exp.umap_calinski_mean,
            'umap_calinski_std': exp.umap_calinski_std,
            'umap_calinski_count': exp.umap_calinski_count,
            'umap_davies_mean': exp.umap_davies_mean,
            'umap_davies_std': exp.umap_davies_std,
            'umap_davies_count': exp.umap_davies_count
        })

    experiment_df = pd.DataFrame(experiment_data)
    experiment_df.to_csv(os.path.join(output_dir, 'experiment_metrics_detailed.csv'), index=False)

    # Save summary text report
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("OSR UMAP Metrics Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total experiments processed: {len(experiment_metrics_list)}\n")
        f.write(f"Models analyzed: {len(model_stats_dict)}\n\n")

        f.write("MODEL STATISTICS SUMMARY\n")
        f.write("-" * 30 + "\n\n")

        for model_name in sorted(model_stats_dict.keys()):
            stats = model_stats_dict[model_name]
            f.write(f"Model: {model_name} (n={stats.n_experiments} experiments)\n")
            f.write("-" * 40 + "\n")

            f.write("Raw Features Metrics (Level 1):\n")
            f.write(f"  Silhouette Score: {stats.raw_silhouette_mean:.4f} ± {stats.raw_silhouette_std:.4f}\n")
            f.write(f"  Calinski-Harabasz Score: {stats.raw_calinski_mean:.4f} ± {stats.raw_calinski_std:.4f}\n")
            f.write(f"  Davies-Bouldin Score: {stats.raw_davies_mean:.4f} ± {stats.raw_davies_std:.4f}\n")

            f.write("\nUMAP Features Metrics (Level 1, averaged across configurations):\n")
            f.write(f"  Silhouette Score: {stats.umap_silhouette_mean_avg:.4f} ± {stats.umap_silhouette_std_avg:.4f}\n")
            f.write(f"  Calinski-Harabasz Score: {stats.umap_calinski_mean_avg:.4f} ± {stats.umap_calinski_std_avg:.4f}\n")
            f.write(f"  Davies-Bouldin Score: {stats.umap_davies_mean_avg:.4f} ± {stats.umap_davies_std_avg:.4f}\n")
            f.write("\n\n")

        f.write("NOTE: Higher values are better for Silhouette and Calinski-Harabasz scores.\n")
        f.write("Lower values are better for Davies-Bouldin scores.\n\n")

        f.write("Statistical significance testing would require raw experiment data\n")
        f.write("for proper t-tests, ANOVA, and effect size calculations.\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="OSR UMAP Metrics Automation")
    parser.add_argument('--experiments-dir', type=str, default='final_overnight_osr_experiments',
                       help='Directory containing experiment folders')
    parser.add_argument('--residuals-dir', type=str, default='residuals_umap_visualization_outputs',
                       help='Directory containing residuals UMAP visualization outputs')
    parser.add_argument('--output-dir', type=str, default='osr_umap_metrics_results',
                       help='Output directory for results')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Maximum number of experiments to process (for testing)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip UMAP processing and only analyze existing results')
    parser.add_argument('--baseline-model', type=str, default='adaptive_lbpin',
                       help='Baseline model for significance testing')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting OSR UMAP Metrics Automation")
    logger.info(f"Experiments directory: {args.experiments_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Step 1: Discover experiment folders
    if not os.path.exists(args.experiments_dir):
        logger.error(f"Experiments directory not found: {args.experiments_dir}")
        return

    experiment_paths = []
    for item in os.listdir(args.experiments_dir):
        item_path = os.path.join(args.experiments_dir, item)
        if os.path.isdir(item_path) and not item.startswith('old'):
            # Check if it has embeddings.pkl
            if os.path.exists(os.path.join(item_path, 'embeddings.pkl')):
                experiment_paths.append(item_path)

    experiment_paths.sort()
    logger.info(f"Found {len(experiment_paths)} experiment folders with embeddings")

    if args.max_experiments:
        experiment_paths = experiment_paths[:args.max_experiments]
        logger.info(f"Limited to {len(experiment_paths)} experiments for testing")

    # Step 2: Process experiments (unless skipped)
    if not args.skip_processing:
        logger.info("Starting UMAP visualization processing...")

        successful_experiments = 0
        with tqdm(total=len(experiment_paths), desc="Processing experiments") as pbar:
            for experiment_path in experiment_paths:
                experiment_name = os.path.basename(experiment_path)

                if run_umap_visualization_for_experiment(experiment_path, args.output_dir):
                    successful_experiments += 1

                pbar.set_postfix({
                    'successful': successful_experiments,
                    'total': pbar.n
                })
                pbar.update(1)

        logger.info(f"Successfully processed {successful_experiments}/{len(experiment_paths)} experiments")
    else:
        logger.info("Skipping UMAP processing, analyzing existing results...")

    # Step 3: Collect pre-SNN metrics from residuals
    pre_snn_stats = collect_residuals_metrics(args.residuals_dir)

    # Step 4: Collect post-SNN metrics from all experiments
    logger.info("Collecting post-SNN metrics from processed experiments...")

    experiment_metrics_list = []
    processed_experiments = 0

    with tqdm(total=len(experiment_paths), desc="Collecting metrics") as pbar:
        for experiment_path in experiment_paths:
            experiment_name = os.path.basename(experiment_path)
            model_name = extract_model_name_from_experiment(experiment_path)

            metrics = collect_experiment_metrics(args.output_dir, experiment_name, model_name)
            if metrics is not None:
                experiment_metrics_list.append(metrics)
                processed_experiments += 1

            pbar.set_postfix({
                'collected': processed_experiments,
                'total': pbar.n
            })
            pbar.update(1)

    logger.info(f"Collected metrics from {processed_experiments} experiments")

    if not experiment_metrics_list:
        logger.error("No metrics collected. Exiting.")
        return

    # Step 5: Group by model and calculate post-SNN statistics
    logger.info("Calculating post-SNN model statistics...")

    model_groups = {}
    for exp_metrics in experiment_metrics_list:
        if exp_metrics.model_name not in model_groups:
            model_groups[exp_metrics.model_name] = []
        model_groups[exp_metrics.model_name].append(exp_metrics)

    post_snn_stats = {}
    for model_name, exp_list in model_groups.items():
        try:
            post_snn_stats[model_name] = calculate_model_statistics(exp_list)
            logger.info(f"Calculated post-SNN statistics for {model_name} ({len(exp_list)} experiments)")
        except Exception as e:
            logger.error(f"Error calculating post-SNN statistics for {model_name}: {e}")

    # Step 6: Calculate deltas (post-SNN - pre-SNN)
    logger.info("Calculating deltas (post-SNN - pre-SNN)...")

    deltas_stats = {}
    common_models = set(pre_snn_stats.keys()) & set(post_snn_stats.keys())

    for model_name in common_models:
        pre_stats = pre_snn_stats[model_name]
        post_stats = post_snn_stats[model_name]

        # Calculate deltas for each metric
        # Standard deviations for differences use: SD_diff = √(SD₁² + SD₂²) for independent samples
        deltas_stats[model_name] = ModelStatistics(
            model_name=model_name,
            n_experiments=post_stats.n_experiments,  # Use post-SNN experiment count
            raw_silhouette_mean=post_stats.raw_silhouette_mean - pre_stats.raw_silhouette_mean,
            raw_silhouette_std=math.sqrt(post_stats.raw_silhouette_std**2 + pre_stats.raw_silhouette_std**2),
            raw_calinski_mean=post_stats.raw_calinski_mean - pre_stats.raw_calinski_mean,
            raw_calinski_std=math.sqrt(post_stats.raw_calinski_std**2 + pre_stats.raw_calinski_std**2),
            raw_davies_mean=post_stats.raw_davies_mean - pre_stats.raw_davies_mean,
            raw_davies_std=math.sqrt(post_stats.raw_davies_std**2 + pre_stats.raw_davies_std**2),
            umap_silhouette_mean_avg=post_stats.umap_silhouette_mean_avg - pre_stats.umap_silhouette_mean_avg,
            umap_silhouette_std_avg=math.sqrt(post_stats.umap_silhouette_std_avg**2 + pre_stats.umap_silhouette_std_avg**2),
            umap_calinski_mean_avg=post_stats.umap_calinski_mean_avg - pre_stats.umap_calinski_mean_avg,
            umap_calinski_std_avg=math.sqrt(post_stats.umap_calinski_std_avg**2 + pre_stats.umap_calinski_std_avg**2),
            umap_davies_mean_avg=post_stats.umap_davies_mean_avg - pre_stats.umap_davies_mean_avg,
            umap_davies_std_avg=math.sqrt(post_stats.umap_davies_std_avg**2 + pre_stats.umap_davies_std_avg**2)
        )

    logger.info(f"Calculated deltas for {len(deltas_stats)} common models")

    # Step 7: Perform significance tests
    logger.info("Performing statistical significance tests...")
    significance_results = perform_significance_tests(post_snn_stats, args.baseline_model)

    # Step 8: Create visualizations
    logger.info("Creating summary plots...")
    create_summary_plots(post_snn_stats, args.output_dir)

    # Step 9: Save results
    logger.info("Saving results to files...")
    save_results_to_files(post_snn_stats, experiment_metrics_list, args.output_dir)

    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {args.output_dir}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("OSR UMAP METRICS ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nTotal experiments processed: {len(experiment_metrics_list)}")
    print(f"Pre-SNN models: {len(pre_snn_stats)}")
    print(f"Post-SNN models: {len(post_snn_stats)}")
    print(f"Models with deltas: {len(deltas_stats)}")

    # Pre-SNN Metrics Section
    print("\n" + "=" * 80)
    print("PRE-SNN METRICS (from residuals)")
    print("=" * 80)
    print("Clustering quality in residual feature space before SNN processing")

    for model_name in sorted(pre_snn_stats.keys()):
        stats = pre_snn_stats[model_name]
        print(f"\n{model_name.upper()}:")

        print("  Raw Features Metrics (Level 1):")
        print(f"    Silhouette Score: {stats.raw_silhouette_mean:.4f} ± {stats.raw_silhouette_std:.4f}")
        print(f"    Calinski-Harabasz Score: {stats.raw_calinski_mean:.4f} ± {stats.raw_calinski_std:.4f}")
        print(f"    Davies-Bouldin Score: {stats.raw_davies_mean:.4f} ± {stats.raw_davies_std:.4f}")

        print("  UMAP Features Metrics (Level 1, averaged across configurations):")
        print(f"    Silhouette Score: {stats.umap_silhouette_mean_avg:.4f} ± {stats.umap_silhouette_std_avg:.4f}")
        print(f"    Calinski-Harabasz Score: {stats.umap_calinski_mean_avg:.4f} ± {stats.umap_calinski_std_avg:.4f}")
        print(f"    Davies-Bouldin Score: {stats.umap_davies_mean_avg:.4f} ± {stats.umap_davies_std_avg:.4f}")

    # Post-SNN Metrics Section
    print("\n" + "=" * 80)
    print("POST-SNN METRICS (from embeddings)")
    print("=" * 80)
    print("Clustering quality in embedding feature space after SNN processing")

    for model_name in sorted(post_snn_stats.keys()):
        stats = post_snn_stats[model_name]
        print(f"\n{model_name.upper()} (n={stats.n_experiments}):")

        print("  Raw Features Metrics (Level 1):")
        print(f"    Silhouette Score: {stats.raw_silhouette_mean:.4f} ± {stats.raw_silhouette_std:.4f}")
        print(f"    Calinski-Harabasz Score: {stats.raw_calinski_mean:.4f} ± {stats.raw_calinski_std:.4f}")
        print(f"    Davies-Bouldin Score: {stats.raw_davies_mean:.4f} ± {stats.raw_davies_std:.4f}")

        print("  UMAP Features Metrics (Level 1, averaged across configurations):")
        print(f"    Silhouette Score: {stats.umap_silhouette_mean_avg:.4f} ± {stats.umap_silhouette_std_avg:.4f}")
        print(f"    Calinski-Harabasz Score: {stats.umap_calinski_mean_avg:.4f} ± {stats.umap_calinski_std_avg:.4f}")
        print(f"    Davies-Bouldin Score: {stats.umap_davies_mean_avg:.4f} ± {stats.umap_davies_std_avg:.4f}")

    # Deltas Section
    print("\n" + "=" * 80)
    print("DELTAS (Post-SNN - Pre-SNN)")
    print("=" * 80)
    print("Improvement in clustering quality after SNN processing")
    print("Positive values = improvement, Negative values = degradation")

    for model_name in sorted(deltas_stats.keys()):
        stats = deltas_stats[model_name]
        print(f"\n{model_name.upper()} (n={stats.n_experiments}):")

        print("  Raw Features Deltas (Level 1):")
        print(f"    Silhouette Score: {stats.raw_silhouette_mean:+.4f} ± {stats.raw_silhouette_std:+.4f}")
        print(f"    Calinski-Harabasz Score: {stats.raw_calinski_mean:+.4f} ± {stats.raw_calinski_std:+.4f}")
        print(f"    Davies-Bouldin Score: {stats.raw_davies_mean:+.4f} ± {stats.raw_davies_std:+.4f}")

        print("  UMAP Features Deltas (Level 1, averaged across configurations):")
        print(f"    Silhouette Score: {stats.umap_silhouette_mean_avg:+.4f} ± {stats.umap_silhouette_std_avg:+.4f}")
        print(f"    Calinski-Harabasz Score: {stats.umap_calinski_mean_avg:+.4f} ± {stats.umap_calinski_std_avg:+.4f}")
        print(f"    Davies-Bouldin Score: {stats.umap_davies_mean_avg:+.4f} ± {stats.umap_davies_std_avg:+.4f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION NOTES")
    print("=" * 80)
    print("Higher values are better for Silhouette and Calinski-Harabasz.")
    print("Lower values are better for Davies-Bouldin.")
    print("Raw metrics: Clustering quality in original feature space.")
    print("UMAP metrics: Clustering quality in 2D projections (averaged across parameter combinations).")
    print("Deltas: Positive values indicate SNN improved clustering, negative values indicate degradation.")
    print("=" * 80)


if __name__ == "__main__":
    main()

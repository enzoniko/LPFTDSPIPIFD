#!/usr/bin/env python3
"""
Extract Raw Metrics from Bayesian Optimization Results for Fair Cross-Method Comparison

This script extracts standardized raw metrics from Bayesian optimization results
to enable fair comparison between different adaptive weighting methods.

Raw metrics extracted (same across all methods):
1. Raw data loss (MAE/RMSE) - calculated consistently
2. Raw physics residuals (absolute values) - calculated consistently  
3. Learned physical parameters (M1, M2, M3, D1, D2, D3, K1, K2, E1)
4. Final validation performance on raw metrics

Usage:
    python extract_bayesian_raw_metrics.py [--results-dir test_bayesian_results] [--output-file bayesian_raw_metrics.csv]
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_training_history(file_path: str) -> Dict[str, np.ndarray]:
    """Load training history from .npz file."""
    try:
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def extract_raw_metrics(history: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract raw metrics that are calculated consistently across all methods."""
    metrics = {}
    
    # Raw data residuals (MAE) for each acceleration component - FINAL VALUES
    data_residual_keys = [
        'raw_x2_ddot_mae_val', 'raw_y2_ddot_mae_val', 
        'raw_x3_ddot_mae_val', 'raw_y3_ddot_mae_val',
        'raw_total_mae_val'
    ]
    
    for key in data_residual_keys:
        if key in history and len(history[key]) > 0:
            metrics[key] = float(history[key][-1])  # Final value
    
    # Raw physical residuals (mean and std) - FINAL VALUES
    phys_residual_keys = [
        'raw_res1_mean_val', 'raw_res2_mean_val', 'raw_res3_mean_val',
        'raw_res4_mean_val', 'raw_res_mass1_mean_val', 'raw_res_mass2_mean_val',
        'raw_res1_std_val', 'raw_res2_std_val', 'raw_res3_std_val',
        'raw_res4_std_val', 'raw_res_mass1_std_val', 'raw_res_mass2_std_val'
    ]
    
    for key in phys_residual_keys:
        if key in history and len(history[key]) > 0:
            metrics[key] = float(history[key][-1])  # Final value
    
    # Model parameters (final values)
    param_keys = ['param_M1', 'param_M2', 'param_M3', 'param_D1', 'param_D2', 
                 'param_D3', 'param_K1', 'param_K2', 'param_E1']
    
    for key in param_keys:
        if key in history and len(history[key]) > 0:
            param_name = key.replace('param_', '')
            metrics[f"final_{param_name}"] = float(history[key][-1])
    
    return metrics


def process_bayesian_results(results_dir: str) -> List[Dict[str, Any]]:
    """Process all Bayesian optimization results to extract raw metrics."""
    results_path = Path(results_dir)
    all_results = []
    
    # Methods to process (matching actual directory names)
    methods = [
        'adaptive_lbpin', 'alpinn', 'constant_weight', 'brdr',
        'relobralo', 'pecann', 'gradnorm', 'dwpinn'
    ]
    
    for method_name in methods:
        method_dir = results_path / method_name
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_name}")
            continue
        
        print(f"\nProcessing {method_name}...")
        
        # Process both starting points
        for starting_point in [1, 2]:
            starting_point_dir = method_dir / f"starting_point_{starting_point}"
            if not starting_point_dir.exists():
                continue
            
            # Find all trial directories (they're under results/ subdirectory)
            trial_results_dir = starting_point_dir / 'results'
            if not trial_results_dir.exists():
                print(f"  Warning: No results directory found in {starting_point_dir}")
                continue
                
            trial_dirs = [d for d in trial_results_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
            
            for trial_dir in trial_dirs:
                # Look for history files
                history_files = list(trial_dir.glob("*_history.npz"))
                
                if not history_files:
                    print(f"  Warning: No history file found in {trial_dir}")
                    continue
                
                history_file = history_files[0]
                history = load_training_history(str(history_file))
                
                if not history:
                    continue
                
                # Extract raw metrics
                raw_metrics = extract_raw_metrics(history)
                
                if not raw_metrics:
                    print(f"  Warning: No raw metrics found in {history_file}")
                    continue
                
                # Create result entry
                result = {
                    'method': method_name,
                    'starting_point': starting_point,
                    'trial': trial_dir.name,
                    'history_file': str(history_file)
                }
                
                # Add raw metrics
                result.update(raw_metrics)
                
                # Add trial metadata if available
                trial_log_dir = trial_dir / 'logs'
                if trial_log_dir.exists():
                    log_files = list(trial_log_dir.glob("*.log"))
                    if log_files:
                        result['log_file'] = str(log_files[0])
                
                all_results.append(result)
                print(f"  ✓ {trial_dir.name}: {len(raw_metrics)} raw metrics extracted")
    
    return all_results


def create_comparison_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary DataFrame for comparison."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['method', 'starting_point', 'trial', 'history_file']
    
    # Add log file if available
    if 'log_file' in df.columns:
        column_order.append('log_file')
    
    # Add raw data residual metrics (MAE)
    data_residual_columns = [col for col in df.columns if 'mae_val' in col]
    column_order.extend(sorted(data_residual_columns))
    
    # Add raw physical residual metrics
    phys_residual_columns = [col for col in df.columns if 'raw_res' in col and 'val' in col]
    column_order.extend(sorted(phys_residual_columns))
    
    # Add model parameters
    param_columns = [col for col in df.columns if col.startswith('final_') and col not in phys_residual_columns]
    column_order.extend(sorted(param_columns))
    
    # Add remaining columns
    remaining_columns = [col for col in df.columns if col not in column_order]
    column_order.extend(sorted(remaining_columns))
    
    # Reorder DataFrame
    df = df[column_order]
    
    return df


def analyze_best_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the best trials for each method based on raw metrics."""
    if df.empty:
        return df
    
    # Group by method and find best trials based on raw_total_mae_val
    best_trials = []
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method].copy()
        
        if 'raw_total_mae_val' in method_df.columns:
            # Sort by MAE to get best trials
            method_df_sorted = method_df.sort_values('raw_total_mae_val')
            
            # Get best trial
            if len(method_df_sorted) > 0:
                best_trial = method_df_sorted.iloc[0].copy()
                best_trial['rank'] = 1
                best_trials.append(best_trial)
                
                # Get second best trial
                if len(method_df_sorted) > 1:
                    second_best_trial = method_df_sorted.iloc[1].copy()
                    second_best_trial['rank'] = 2
                    best_trials.append(second_best_trial)
    
    if best_trials:
        best_df = pd.DataFrame(best_trials)
        return best_df.sort_values(['rank', 'raw_total_mae_val'])
    
    return df


def main():
    """Main function to extract and compare raw metrics from Bayesian optimization results."""
    parser = argparse.ArgumentParser(description='Extract Raw Metrics from Bayesian Optimization Results')
    parser.add_argument('--results-dir', type=str, default='test_bayesian_results',
                       help='Directory containing Bayesian optimization results')
    parser.add_argument('--output-file', type=str, default='bayesian_raw_metrics.csv',
                       help='Output CSV file for raw metrics')
    parser.add_argument('--best-trials-file', type=str, default='bayesian_best_trials.csv',
                       help='Output CSV file for best trials comparison')
    args = parser.parse_args()
    
    print("Extracting raw metrics from Bayesian optimization results...")
    print(f"Results directory: {args.results_dir}")
    
    # Process all results
    all_results = process_bayesian_results(args.results_dir)
    
    if not all_results:
        print("\nNo results found!")
        return
    
    # Create comparison DataFrame
    df = create_comparison_summary(all_results)
    
    # Save all results
    output_path = Path(args.output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\nAll results saved to {output_path}")
    print(f"Total trials processed: {len(df)}")
    print(f"Methods found: {list(df['method'].unique())}")
    
    # Create best trials comparison
    best_df = analyze_best_trials(df)
    
    if not best_df.empty:
        best_output_path = Path(args.best_trials_file)
        best_df.to_csv(best_output_path, index=False)
        
        print(f"\nBest trials comparison saved to {best_output_path}")
        
        # Print summary of best trials
        print(f"\n{'='*80}")
        print("BEST TRIALS COMPARISON (Based on Raw Total MAE)")
        print(f"{'='*80}")
        
        if 'raw_total_mae_val' in best_df.columns:
            # Show top methods
            top_methods = best_df[best_df['rank'] == 1].sort_values('raw_total_mae_val')
            print(f"\n🏆 TOP METHODS (Best Trial Each):")
            for i, (_, row) in enumerate(top_methods.iterrows()):
                print(f"{i+1:2d}. {row['method']:20s} | MAE: {row['raw_total_mae_val']:8.4f} | Trial: {row['trial']}")
        
        # Show detailed metrics for best trials
        print(f"\n📊 DETAILED METRICS FOR BEST TRIALS:")
        key_metrics = ['method', 'raw_total_mae_val', 'raw_x2_ddot_mae_val', 
                      'raw_y2_ddot_mae_val', 'raw_x3_ddot_mae_val', 'raw_y3_ddot_mae_val']
        available_metrics = [col for col in key_metrics if col in best_df.columns]
        if available_metrics:
            print(best_df[available_metrics].to_string(index=False))
    
    # Show summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    if 'raw_total_mae_val' in df.columns:
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                # Handle NaN values properly
                mae_series = method_df['raw_total_mae_val']
                mae_values = mae_series[~mae_series.isna()]
                if len(mae_values) > 0:
                    print(f"{method:20s} | Trials: {len(mae_values):3d} | "
                          f"Best MAE: {mae_values.min():8.4f} | "
                          f"Mean MAE: {mae_values.mean():8.4f} | "
                          f"Std MAE: {mae_values.std():8.4f}")


if __name__ == '__main__':
    main() 
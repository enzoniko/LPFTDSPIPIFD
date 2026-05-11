#!/usr/bin/env python3
"""
Creates individual comparison plots for each model.

This script generates separate metric plots for each model type
(hybrid RNN-PINN and data-driven models) showing performance metrics
across different rotation speeds.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import argparse
from tqdm import tqdm
import glob

# Set default font sizes for publication quality
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})

def filter_outliers_iqr(data, k=1.5):
    """Filter outliers using the IQR method."""
    data = np.array(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Get mask of non-outliers
    mask = (data >= lower_bound) & (data <= upper_bound)
    
    # Return filtered data and original indices
    return data[mask], np.where(mask)[0]

def process_residuals_data(residuals_dict, data_type='normal'):
    """Process residuals data to extract metrics by speed."""
    if data_type not in residuals_dict:
        return None
    
    data = residuals_dict[data_type]
    
    # Group by rotation speed
    speeds_dict = {}
    for seq_idx, seq_data in data.items():
        speed = seq_data['rot_speed']
        if speed not in speeds_dict:
            speeds_dict[speed] = []
        speeds_dict[speed].append(seq_data)
    
    # Sort speeds for consistent plotting
    sorted_speeds = sorted(speeds_dict.keys())
    
    # Calculate metrics for each speed
    metrics = {
        'speeds': sorted_speeds,
        'mean': [],
        'median': [],
        'std': [],
        'skew': [],
        'kurt': [],
        'dtw': []
    }
    
    for speed in sorted_speeds:
        speed_sequences = speeds_dict[speed]
        
        # Collect all residuals for this speed
        all_residuals = []
        dtw_distances = []
        
        for seq_data in speed_sequences:
            residuals = seq_data['data'].detach().cpu().numpy()
            all_residuals.append(residuals.flatten())
            
            if 'dtw_distances' in seq_data:
                dtw_distances.extend(seq_data['dtw_distances'])
        
        # Concatenate all residuals
        all_residuals = np.concatenate(all_residuals)
        
        # Calculate metrics
        metrics['mean'].append(np.mean(all_residuals))
        metrics['median'].append(np.median(all_residuals))
        metrics['std'].append(np.std(all_residuals))
        metrics['skew'].append(skew(all_residuals))
        metrics['kurt'].append(kurtosis(all_residuals))
        
        if dtw_distances:
            metrics['dtw'].append(np.mean(dtw_distances))
        else:
            metrics['dtw'].append(np.nan)
    
    return metrics

def create_individual_metrics_plot(model_name, metrics, output_path):
    """Create metrics plot for an individual model."""
    # Modified layout: 3 rows, 2 columns for better vertical fit in papers
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    axs = axs.flatten()
    
    # Define the metrics to plot
    plot_metrics = ['mean', 'median', 'std', 'skew', 'kurt', 'dtw']
    titles = ['Mean Residual', 'Median Residual', 'Standard Deviation', 
              'Skewness', 'Kurtosis', 'DTW Distance']
    
    speeds = metrics['speeds']
    
    # Create a color map for this model (using a consistent color)
    color = plt.cm.tab10(0)  # Use first color in the tab10 colormap
    
    # For each metric, plot the values
    for i, (metric, title) in enumerate(zip(plot_metrics, titles)):
        ax = axs[i]
        
        if metric in metrics:
            values = metrics[metric]
            
            # Filter outliers using IQR to improve scaling
            if len(values) > 4:
                filtered_values, _ = filter_outliers_iqr(values)
                if len(filtered_values) > 0:
                    y_min = min(filtered_values) * 0.9
                    y_max = max(filtered_values) * 1.1
                    ax.set_ylim(y_min, y_max)
            
            ax.plot(speeds, values, 'o-', linewidth=2, markersize=10, color=color)
            ax.set_title(title)
            ax.set_xlabel('Rotation Frequency (Hz)')
            ax.set_ylabel('Value')
            ax.grid(True)
        else:
            ax.set_title(f"{title} (Not Available)")
            ax.set_xlabel('Rotation Frequency (Hz)')
            ax.set_ylabel('Value')
            ax.grid(True)
    
    plt.suptitle(f'Performance Metrics: {model_name}', fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual metrics plot for {model_name} saved to {output_path}")
    return True

def load_residuals(residuals_path):
    """Load residuals from a file (.pth format)."""
    try:
        return torch.load(residuals_path)
    except Exception as e:
        print(f"Error loading residuals from {residuals_path}: {e}")
        return None

def load_metrics_from_npz(npz_path):
    """Load metrics data from an NPZ file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if 'improved_layout' in npz_path or any('normal' in key for key in data.keys()):
            metrics = {}
            
            # Look for speeds data
            speeds_keys = [k for k in data.keys() if 'speeds' in k.lower() or 'freq' in k.lower()]
            if speeds_keys:
                speeds_key = speeds_keys[0]
                metrics['speeds'] = data[speeds_key]
                
                # Look for metrics data for normal condition
                metric_keys = {
                    'mean': [k for k in data.keys() if 'mean' in k.lower() and 'normal' in k.lower()],
                    'median': [k for k in data.keys() if 'median' in k.lower() and 'normal' in k.lower()],
                    'std': [k for k in data.keys() if 'std' in k.lower() and 'normal' in k.lower()],
                    'skew': [k for k in data.keys() if 'skew' in k.lower() and 'normal' in k.lower()],
                    'kurt': [k for k in data.keys() if 'kurt' in k.lower() and 'normal' in k.lower()],
                    'dtw': [k for k in data.keys() if 'dtw' in k.lower() and 'normal' in k.lower()]
                }
                
                for metric, keys in metric_keys.items():
                    if keys:
                        metrics[metric] = data[keys[0]]
                
                return metrics
        
        return None
    except Exception as e:
        print(f"Error loading NPZ file {npz_path}: {e}")
        return None

def main():
    """Main function to create individual model metrics plots."""
    parser = argparse.ArgumentParser(description="Create individual model metrics plots")
    parser.add_argument("--hybrid-path", help="Path to hybrid RNN-PINN residuals file")
    parser.add_argument("--npz-path", help="Path to NPZ file with comparison data")
    parser.add_argument("--output-dir", default="metrics_comparison_plots", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store metrics by model
    all_metrics = {}
    
    # 1. Try to load hybrid RNN-PINN metrics from NPZ if provided
    if args.npz_path and os.path.exists(args.npz_path):
        npz_metrics = load_metrics_from_npz(args.npz_path)
        if npz_metrics and 'speeds' in npz_metrics:
            all_metrics['Hybrid RNN-PINN (NPZ)'] = npz_metrics
            print(f"Loaded hybrid RNN-PINN metrics from NPZ file: {args.npz_path}")
    
    # 2. Try to load hybrid RNN-PINN residuals from PTH
    hybrid_path = args.hybrid_path or 'hybrid_rnn_residuals_1742862177.pth'
    if os.path.exists(hybrid_path):
        residuals = load_residuals(hybrid_path)
        if residuals:
            metrics = process_residuals_data(residuals, 'normal')
            if metrics:
                all_metrics['Hybrid RNN-PINN'] = metrics
                print(f"Loaded and processed hybrid RNN-PINN residuals from {hybrid_path}")
    
    # 3. Try to load data-driven model residuals
    data_driven_patterns = ['data_driven_*_models/*_residuals.pth']
    
    for pattern in data_driven_patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            # Extract model type from filename
            model_type = os.path.basename(file_path).split('_residuals.pth')[0]
            
            # Load residuals
            residuals = load_residuals(file_path)
            if residuals:
                # Format model name
                if model_type == 'simple_rnn':
                    display_name = 'Simple RNN'
                elif model_type == 'cnn_lstm':
                    display_name = 'CNN-LSTM'
                else:
                    display_name = model_type.upper()
                
                # Process residuals into metrics
                metrics = process_residuals_data(residuals, 'normal')
                if metrics:
                    all_metrics[display_name] = metrics
                    print(f"Loaded and processed {display_name} residuals from {file_path}")
    
    # Create individual plots for each model
    if all_metrics:
        for model_name, metrics in all_metrics.items():
            safe_model_name = model_name.replace(' ', '_').lower()
            output_path = os.path.join(args.output_dir, f"{safe_model_name}_metrics.png")
            create_individual_metrics_plot(model_name, metrics, output_path)
        
        print(f"Created individual metrics plots for {len(all_metrics)} models")
    else:
        print("No metrics data available for plotting.")

if __name__ == "__main__":
    main() 
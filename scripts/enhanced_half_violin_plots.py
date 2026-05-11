#!/usr/bin/env python3
"""
Creates enhanced half violin plots for residual distributions comparison.

This script generates vertical layouts of half violin distribution plots for multiple models,
with residuals properly segmented by their actual rotation speeds.

Key features:
- Generates compact, vertically-oriented half violin visualizations across rotation speeds
- Uses consistent y-axes limits for all variables
- Includes publication-quality formatting suitable for academic papers
- Supports both direct PINN and hybrid RNN-PINN residual formats
- Properly segments residuals by actual rotation speeds
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import glob
import sys
import importlib.util
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_half_violin_plots(residuals_by_model, output_dir, residual_type='data'):
    """
    Create compact visualizations of residual distributions using half violin plots,
    focusing only on normal data, properly segmented by rotation speed.
    
    Parameters:
    - residuals_by_model: Dictionary with model names as keys and residuals dictionaries as values
                         in the standardized format
    - output_dir: Directory to save the output plots
    - residual_type: Type of residuals to plot ('data' or 'physical')
    
    Returns:
    - True if successful, False otherwise
    """
    # Set publication-quality font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the variables to plot based on residual type
    if residual_type == 'physical':
        variables = ['R1', 'R2', 'R3', 'R4']
        title_suffix = "Physical Residuals"
        residual_key = "physical_residuals"
    else:  # data residuals
        variables = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
        title_suffix = "Data Residuals"
        residual_key = "data_residuals"
    
    num_variables = len(variables)
    
    # Define color and linestyle patterns to cycle through
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'darkblue']
    linestyles = ['-', '--', ':', '-.']
    
    # Create all combinations
    styles = []
    for ls in linestyles:
        for color in colors:
            styles.append((color, ls))
    
    # First, collect data from all models to determine global y-axis limits
    all_data_by_variable = {var: [] for var in variables}
    
    for model_name, residuals_dict in residuals_by_model.items():
        # Skip if this model doesn't have the requested residual type
        if residual_key not in residuals_dict:
            print(f"Warning: {residual_key} not found in residuals for {model_name}")
            continue
        
        # Check if normal data exists
        if 'normal' not in residuals_dict[residual_key]:
            print(f"Warning: normal data not found in {residual_key} for {model_name}")
            continue
        
        # Process each frequency for normal data
        for freq, freq_data in residuals_dict[residual_key]['normal'].items():
            residuals_tensor = freq_data['residuals']  # Shape: [num_rotations, points_per_rotation, variables]
            
            # Convert numpy array to tensor if needed
            if isinstance(residuals_tensor, np.ndarray):
                residuals_tensor = torch.from_numpy(residuals_tensor)
            
            # Process each rotation's data for each variable
            for rotation_idx in range(residuals_tensor.shape[0]):
                rotation_data = residuals_tensor[rotation_idx]  # Shape: [points_per_rotation, variables]
                
                for var_idx, var in enumerate(variables):
                    # Skip if variable index is out of range
                    if var_idx >= rotation_data.shape[1]:
                        continue
                    
                    # Extract residuals for this variable
                    residuals = rotation_data[:, var_idx].detach().cpu().numpy()
                    all_data_by_variable[var].extend(residuals)
    
    # Calculate global y-axis limits
    y_limits = {}
    for var in variables:
        residuals = all_data_by_variable[var]
        if residuals:
            y_limits[var] = {
                'min': np.percentile(residuals, 1),  # Use 1st percentile
                'max': np.percentile(residuals, 99)  # Use 99th percentile
            }
    
    # Now process each model
    for model_name, residuals_dict in residuals_by_model.items():
        # Skip if this model doesn't have the requested residual type
        if residual_key not in residuals_dict:
            continue
            
        # Check if normal data exists
        if 'normal' not in residuals_dict[residual_key]:
            continue
        
        # Get all frequencies for normal data
        frequencies = list(residuals_dict[residual_key]['normal'].keys())
        sorted_frequencies = sorted(frequencies)
        
        # Create a vertical layout for the distributions (paper-friendly dimensions)
        fig, axs = plt.subplots(num_variables, 1, figsize=(8, 1.5*num_variables), sharex=True)
        
        if num_variables == 1:
            axs = [axs]
        
        # Process each variable
        for var_idx, (var, ax) in enumerate(zip(variables, axs)):
            # Collect data for all frequencies
            for freq_idx, freq in enumerate(sorted_frequencies):
                # Get color and linestyle for this frequency
                color, linestyle = styles[freq_idx % len(styles)]
                
                # Get data for this frequency
                freq_data = residuals_dict[residual_key]['normal'][freq]
                residuals_tensor = freq_data['residuals']
                
                # Collect residuals for this variable and frequency
                var_residuals = []
                
                # Process each rotation
                for rotation_idx in range(residuals_tensor.shape[0]):
                    rotation_data = residuals_tensor[rotation_idx]
                    
                    # Skip if variable index is out of range
                    if var_idx >= rotation_data.shape[1]:
                        continue
                    
                    # Extract residuals for this variable
                    residuals = rotation_data[:, var_idx].detach().cpu().numpy()
                    var_residuals.extend(residuals)
                
                # Skip if no data
                if not var_residuals:
                    continue
                    
                # Create half violin plot at position based on frequency
                pos = freq  # Use the actual frequency value for x-position
                
                # Calculate width - use a smaller fixed width for denser plot
                width = min(0.5, max(0.2, 0.02 * freq))
                
                # Draw a violin plot with outline only (no fill)
                parts = ax.violinplot([var_residuals], positions=[pos], 
                                      widths=width,
                                      showextrema=False, showmedians=True)
                
                # Convert to half violin plot and keep only the contour
                for pc in parts['bodies']:
                    # Keep only the right side of the violin plot
                    m = np.mean(pc.get_paths()[0].vertices[:, 0])
                    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
                    
                    # Apply varying colors and linestyles
                    pc.set_facecolor('none')  # Transparent fill
                    pc.set_edgecolor(color)   # Use the selected color
                    pc.set_linestyle(linestyle)  # Use the selected linestyle
                    pc.set_linewidth(1.5)  # Make the line a bit thicker
                
                # Add a small marker for the median with the same color
                median = np.median(var_residuals)
                ax.plot(pos, median, 'o', color=color, markersize=6)
            
            # Set y-axis limits based on global limits for comparative visualization
            if var in y_limits:
                ax.set_ylim(y_limits[var]['min'], y_limits[var]['max'])
            
            # Add grid and labels
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_ylabel(f"{var}")
        
        # Set x-axis label on the bottom subplot only
        axs[-1].set_xlabel("Rotation Frequency (Hz)")
        
        # Adjust x-axis limits for better visualization
        min_freq = min(sorted_frequencies) if sorted_frequencies else 0
        max_freq = max(sorted_frequencies) if sorted_frequencies else 100
        margin = (max_freq - min_freq) * 0.05
        for ax in axs:
            ax.set_xlim(min_freq - margin, max_freq + margin)
        
        # Adjust plot title
        plt.suptitle(f'{model_name}: {title_suffix} (Normal Data)', fontsize=14)
        plt.tight_layout()
        
        # Save plot with residual type in filename
        safe_model_name = model_name.replace(' ', '_').lower()
        output_path = os.path.join(output_dir, f"{safe_model_name}_{residual_type}_residuals.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{residual_type.capitalize()} residual plot for {model_name} (normal data) saved to {output_path}")
    
    return True

def try_import_data_module():
    """
    Try to import the Data.LoadData module to get the necessary functions.
    
    Returns:
    - Tuple of (data_paths, get_omegas) functions or (None, None) if import fails
    """
    try:
        # First, try direct import
        from Data.LoadData import data_paths, get_omegas
        return data_paths, get_omegas
    except ImportError:
        # If the import fails, try to add the parent directory to sys.path
        original_path = sys.path.copy()
        
        # Try adding various potential parent directories
        potential_paths = [
            '.',
            '..',
            '../..',
            '../../..',
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ]
        
        for path in potential_paths:
            sys.path.insert(0, path)
            try:
                from Data.LoadData import data_paths, get_omegas
                return data_paths, get_omegas
            except ImportError:
                continue
            finally:
                # Restore original path
                sys.path = original_path
        
        # If we still can't import, return None
        print("WARNING: Could not import Data.LoadData module. Using fallback segmentation approach.")
        return None, None

def segment_direct_pinn_residuals(residuals_dict, data_paths=None, get_omegas=None):
    """
    Segment Direct PINN residuals by rotation speed.
    
    Parameters:
    - residuals_dict: Dictionary of residuals from direct PINN
    - data_paths: Dictionary mapping data types to file paths (or None)
    - get_omegas: Function to get rotation speeds from file paths (or None)
    
    Returns:
    - Segmented residuals dictionary
    """
    segmented_dict = {}
    
    # If data_paths and get_omegas are available, use them for proper segmentation
    if data_paths is not None and get_omegas is not None:
        for key in residuals_dict:
            if key not in data_paths:
                print(f"Warning: {key} not found in data_paths, skipping segmentation")
                continue
                
            file_path = data_paths[key]
            try:
                # Compute rotation speeds in Hz
                omegas = get_omegas(file_path) / (2 * np.pi)
                
                # Parameters for segmentation
                num_rotations = 1
                sampling_rate = 50000  # 50kHz
                datapoints_per_rotation = sampling_rate / omegas
                datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
                
                # Extract data tensor
                tensor = residuals_dict[key]
                if isinstance(tensor, dict) and 'data' in tensor:
                    # Handle the case where the residuals are in the compute_residuals format
                    has_physical = all(k in tensor for k in ['r1', 'r2', 'r3', 'r4'])
                    data_tensor = tensor['data']
                    
                    physical_tensors = None
                    if has_physical:
                        # Combine physical residuals
                        physical_tensors = torch.cat([
                            tensor['r1'].unsqueeze(1),
                            tensor['r2'].unsqueeze(1),
                            tensor['r3'].unsqueeze(1),
                            tensor['r4'].unsqueeze(1)
                        ], dim=1)
                else:
                    # Handle the case where the residuals are direct tensors
                    data_tensor = tensor
                    has_physical = tensor.size(1) > 4
                    physical_tensors = tensor[:, 4:8] if has_physical else None
                    
                    # Extract just the data residuals (first 4 columns)
                    data_tensor = tensor[:, :4]
                
                # Segment the data
                n_blocks = len(data_tensor) // 250000
                segments = []
                physical_segments = [] if has_physical else None
                
                for i in range(n_blocks):
                    if i >= len(datapoints_needed):
                        seg_length = 200  # Default if we don't have omega info
                    else:
                        seg_length = int(datapoints_needed[i])
                    
                    # Calculate number of segments to take
                    max_segments = min(30, 250000 // seg_length) if key == 'normal' else min(10, 250000 // seg_length)
                    
                    for j in range(max_segments):
                        start_idx = i * 250000 + j * seg_length
                        end_idx = start_idx + seg_length
                        
                        if end_idx <= (i + 1) * 250000:
                            # Extract data segment
                            data_segment = data_tensor[start_idx:end_idx]
                            
                            if len(data_segment) > 0:
                                segments.append({
                                    'data': data_segment,
                                    'rot_speed': float(omegas[i])
                                })
                                
                                # Extract physical segment if available
                                if has_physical and physical_tensors is not None:
                                    phys_segment = physical_tensors[start_idx:end_idx]
                                    if len(phys_segment) > 0:
                                        physical_segments.append({
                                            'data': phys_segment,
                                            'rot_speed': float(omegas[i])
                                        })
                
                # Store the segmented data
                segmented_dict[key] = segments
                
                # Store physical residuals if available
                if has_physical and physical_segments:
                    if 'physical_residuals' not in segmented_dict:
                        segmented_dict['physical_residuals'] = {}
                    segmented_dict['physical_residuals'][key] = physical_segments
                    
                print(f"Segmented {key} into {len(segments)} segments")
                
            except Exception as e:
                print(f"Error segmenting {key}: {e}")
                # Fall back to unsegmented data with estimated rotation speed
                segmented_dict[key] = [{
                    'data': residuals_dict[key][:, :4] if isinstance(residuals_dict[key], torch.Tensor) and residuals_dict[key].size(1) > 4 else residuals_dict[key],
                    'rot_speed': 20.0 if key == 'normal' else 30.0
                }]
    else:
        # Fallback: Use simple segmentation with fixed rotation speeds and segment lengths
        print("Using fallback segmentation approach with fixed rotation speeds.")
        rotation_speeds = {
            'normal': 20.0,
            'abnormal': 30.0,
            'abnormal2': 40.0,
            'abnormal3': 50.0,
            'abnormal4': 60.0
        }
        
        for key, tensor in residuals_dict.items():
            rot_speed = rotation_speeds.get(key, 20.0)
            
            # Handle dictionary format from compute_residuals
            if isinstance(tensor, dict) and 'data' in tensor:
                has_physical = all(k in tensor for k in ['r1', 'r2', 'r3', 'r4'])
                data_tensor = tensor['data']
                
                # Segment the data
                segments = []
                physical_segments = [] if has_physical else None
                
                # Use a reasonable segment length (~1 rotation period)
                seg_length = 2500  # approximately 50ms at 50kHz
                
                for i in range(0, len(data_tensor), seg_length):
                    end_idx = min(i + seg_length, len(data_tensor))
                    segments.append({
                        'data': data_tensor[i:end_idx],
                        'rot_speed': rot_speed
                    })
                    
                    if has_physical:
                        # Combine physical residuals
                        physical_data = torch.cat([
                            tensor['r1'][i:end_idx].unsqueeze(1),
                            tensor['r2'][i:end_idx].unsqueeze(1),
                            tensor['r3'][i:end_idx].unsqueeze(1),
                            tensor['r4'][i:end_idx].unsqueeze(1)
                        ], dim=1)
                        
                        physical_segments.append({
                            'data': physical_data,
                            'rot_speed': rot_speed
                        })
                
                segmented_dict[key] = segments
                
                if has_physical and physical_segments:
                    if 'physical_residuals' not in segmented_dict:
                        segmented_dict['physical_residuals'] = {}
                    segmented_dict['physical_residuals'][key] = physical_segments
            
            # Handle direct tensor format
            elif isinstance(tensor, torch.Tensor):
                has_physical = tensor.size(1) > 4
                data_tensor = tensor[:, :4] if has_physical else tensor
                
                # Segment the data
                segments = []
                physical_segments = [] if has_physical else None
                
                # Use a reasonable segment length (~1 rotation period)
                seg_length = 2500  # approximately 50ms at 50kHz
                
                for i in range(0, len(data_tensor), seg_length):
                    end_idx = min(i + seg_length, len(data_tensor))
                    segments.append({
                        'data': data_tensor[i:end_idx],
                        'rot_speed': rot_speed
                    })
                    
                    if has_physical:
                        physical_segments.append({
                            'data': tensor[i:end_idx, 4:8],
                            'rot_speed': rot_speed
                        })
                
                segmented_dict[key] = segments
                
                if has_physical and physical_segments:
                    if 'physical_residuals' not in segmented_dict:
                        segmented_dict['physical_residuals'] = {}
                    segmented_dict['physical_residuals'][key] = physical_segments
    
    return segmented_dict

def process_direct_pinn_residuals(residuals_path):
    """
    Process Direct PINN residuals by properly segmenting by rotation speed.
    
    Parameters:
    - residuals_path: Path to the residuals file
    
    Returns:
    - Dictionary with processed residuals
    """
    print(f"Loading Direct PINN residuals from {residuals_path}")
    residuals_dict = torch.load(residuals_path)
    
    # Try to import data_paths and get_omegas
    data_paths, get_omegas = try_import_data_module()
    
    # Segment the residuals by rotation speed
    segmented_dict = segment_direct_pinn_residuals(residuals_dict, data_paths, get_omegas)
    
    return {'Direct PINN': segmented_dict}

def process_hybrid_rnn_residuals(residuals_path):
    """
    Process Hybrid RNN-PINN residuals.
    
    Parameters:
    - residuals_path: Path to the residuals file
    
    Returns:
    - Dictionary with processed residuals
    """
    print(f"Loading Hybrid RNN-PINN residuals from {residuals_path}")
    residuals_dict = torch.load(residuals_path)
    
    processed = {}
    for data_type, sequences in residuals_dict.items():
        processed[data_type] = []
        for seq_idx, seq_data in sequences.items():
            if 'data' in seq_data and 'rot_speed' in seq_data:
                processed[data_type].append({
                    'data': seq_data['data'],
                    'rot_speed': seq_data['rot_speed']
                })
                
    return {'Hybrid RNN-PINN': processed}

def try_import_utils():
    """
    Try to import the residuals utilities module for structure printing.
    """
    try:
        from utils.residuals_utils import print_residuals_structure
        return print_residuals_structure
    except ImportError:
        # If import fails, define a simple version of the function
        def print_structure(residuals_dict):
            print("\nResiduals dictionary structure:")
            for key in residuals_dict:
                print(f"- {key}")
                if isinstance(residuals_dict[key], dict):
                    for subkey in residuals_dict[key]:
                        print(f"  - {subkey}")
        return print_structure

def try_import_residuals_utils():
    """
    Try to import the standardization functions from residuals_utils.py
    """
    try:
        # First try direct import
        from utils.residuals_utils import standardize_legacy_residuals, load_and_standardize_legacy
        return standardize_legacy_residuals, load_and_standardize_legacy
    except ImportError:
        # Try to find and import from various paths
        candidates = [
            Path.cwd() / "utils" / "residuals_utils.py",
            Path.cwd().parent / "utils" / "residuals_utils.py",
            Path(__file__).parent / "utils" / "residuals_utils.py",
            Path(__file__).parent.parent / "utils" / "residuals_utils.py",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("residuals_utils", candidate)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.standardize_legacy_residuals, module.load_and_standardize_legacy
        
        print("WARNING: Could not import standardize_legacy_residuals. Legacy format conversion will not be available.")
        return None, None

def main():
    """Main function to handle command-line arguments and create plots."""
    parser = argparse.ArgumentParser(description="Create enhanced half-violin distribution plots for normal residuals by rotation speed")
    parser.add_argument("--output-dir", default="distribution_plots", help="Directory to save output plots")
    parser.add_argument("--direct-pinn-path", help="Path to Direct PINN residuals file")
    parser.add_argument("--hybrid-path", help="Path to Hybrid RNN-PINN residuals file")
    parser.add_argument("--residual-type", default="both", 
                      choices=["data", "physical", "both"],
                      help="Type of residuals to plot (data, physical, or both)")
    parser.add_argument("--print-structure", action="store_true",
                      help="Print the structure of the loaded residuals")
    
    args = parser.parse_args()
    
    residuals_by_model = {}
    print_residuals_structure = try_import_utils()
    standardize_legacy_residuals, load_and_standardize_legacy = try_import_residuals_utils()
    
    # Load residuals files
    if args.direct_pinn_path and os.path.exists(args.direct_pinn_path):
        print(f"Loading Direct PINN residuals from {args.direct_pinn_path}")
        direct_residuals = torch.load(args.direct_pinn_path)
        
        # Check if the residuals are in legacy format and convert if needed
        if standardize_legacy_residuals and isinstance(direct_residuals, dict) and \
           not any(key in ['data_residuals', 'physical_residuals'] for key in direct_residuals.keys()):
            print("Detected legacy format residuals, converting to standardized format...")
            direct_residuals = standardize_legacy_residuals(direct_residuals)
        
        residuals_by_model['Direct PINN'] = direct_residuals
        
        if args.print_structure:
            print("\nDirect PINN Residuals Structure:")
            print_residuals_structure(direct_residuals)
    
    if args.hybrid_path and os.path.exists(args.hybrid_path):
        print(f"Loading Hybrid RNN-PINN residuals from {args.hybrid_path}")
        hybrid_residuals = torch.load(args.hybrid_path)
        
        # Check if the residuals are in legacy format and convert if needed
        if standardize_legacy_residuals and isinstance(hybrid_residuals, dict) and \
           not any(key in ['data_residuals', 'physical_residuals'] for key in hybrid_residuals.keys()):
            print("Detected legacy format residuals, converting to standardized format...")
            hybrid_residuals = standardize_legacy_residuals(hybrid_residuals)
        
        residuals_by_model['Hybrid RNN-PINN'] = hybrid_residuals
        
        if args.print_structure:
            print("\nHybrid RNN-PINN Residuals Structure:")
            print_residuals_structure(hybrid_residuals)
    
    if not residuals_by_model:
        print("No valid residuals found. Please provide at least one valid residuals file.")
        return
    
    # Create the half-violin distribution plots based on residual type
    if args.residual_type in ["data", "both"]:
        success_data = create_half_violin_plots(residuals_by_model, args.output_dir, residual_type="data")
        print(f"Data residual plots created: {success_data}")
    
    if args.residual_type in ["physical", "both"]:
        # Check if any models have physical residuals
        has_physical = any('physical_residuals' in model_data 
                           for model_data in residuals_by_model.values())
        
        if has_physical:
            success_physical = create_half_violin_plots(residuals_by_model, args.output_dir, residual_type="physical")
            print(f"Physical residual plots created: {success_physical}")
        else:
            print("No physical residuals found in any model.")
    
    print(f"Enhanced half-violin distribution plots created in {args.output_dir}")

if __name__ == "__main__":
    main() 
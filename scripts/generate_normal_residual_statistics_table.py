#!/usr/bin/env python
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import os
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.LoadData import data_paths

def process_residuals(residuals_dict, seq_length=100):
    """
    Convert residual tensors (or already segmented lists) into sequences.
    If a given key already holds a list of segments, pass it through.
    Otherwise, split the tensor into fixed-length chunks.
    """
    processed = {}
    for key, value in tqdm(residuals_dict.items(), desc="Processing residuals"):
        if isinstance(value, list):
            processed[key] = value
        else:
            data = value.cpu().numpy()
            n = data.shape[0]
            n_seqs = n // seq_length
            sequences = np.split(data[:n_seqs * seq_length], n_seqs) if n_seqs > 0 else []
            processed[key] = sequences
    return processed

def adjust_segments_by_rotation(residuals_dict, original_keys_map):
    """
    Adjust segmentation based on rotation speed.
    Each key in the dictionary is assumed to correspond to a condition.
    The segmentation is modified such that each segment corresponds to one rotation.
    
    Parameters:
    -----------
    residuals_dict : dict
        Dictionary with residuals
    original_keys_map : dict
        Mapping from the composite keys to original data_paths keys
    """
    from Data.LoadData import get_omegas
    sampling_rate = 50000  # 50 kHz

    result_dict = {}
    for key, value in residuals_dict.items():
        if key not in original_keys_map:
            print(f"Warning: No mapping found for key {key}, skipping...")
            continue
            
        original_key = original_keys_map[key]
        if original_key not in data_paths:
            print(f"Warning: No data path found for original key {original_key}, skipping...")
            continue
            
        print(f"Processing {key} (original: {original_key})...")
        file_path = data_paths[original_key]
        print(f"Using file path: {file_path}")
        
        try:
            omegas = get_omegas(file_path)
            omegas = omegas / (2 * np.pi)  # Convert from rad/s to Hz
            
            print(f"Original length for {key}: {len(value)}")
            num_rotations = 1
            datapoints_per_rotation = sampling_rate / omegas
            datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
            n_blocks = len(value) // 250000
            segments = []
            for i in range(n_blocks):
                seg_length = datapoints_needed[i] if i < len(datapoints_needed) else 200
                start_idx = i * 250000
                end_idx = start_idx + seg_length
                if end_idx <= (i + 1) * 250000:
                    segments.append(value[start_idx:end_idx])
            result_dict[key] = segments
            print(f"Segmented into {len(segments)} samples for {key}")
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            # Just keep the original data if we can't process it
            result_dict[key] = value
    
    return result_dict

def compute_rotation_speed(segment, sampling_rate=50000):
    """
    Compute rotation speed in Hz assuming each segment represents one full rotation.
    """
    seg_len = len(segment)
    return sampling_rate / seg_len if seg_len > 0 else np.nan

def group_segments_by_rotation_speed(segments, sampling_rate=50000, decimals=1):
    """
    Group segments by their computed rotation speed.
    Speeds are rounded to the specified number of decimals.
    Returns a dictionary with keys as the rounded speeds.
    """
    groups = {}
    for seg in segments:
        speed = compute_rotation_speed(seg, sampling_rate)
        key = round(speed, decimals)
        groups.setdefault(key, []).append(seg)
    return groups

def calculate_residual_statistics(segments, feature_idx=None):
    """
    Calculate statistics for the residuals in the given segments.
    
    Parameters:
    -----------
    segments : list
        List of residual segments
    feature_idx : int or None
        If not None, calculate statistics for the specified feature index
        
    Returns:
    --------
    dict
        Dictionary with statistics (mean, std, skewness, kurtosis)
    """
    all_data = []
    
    for seg in segments:
        if feature_idx is not None and seg.ndim == 2 and seg.shape[1] > feature_idx:
            data = seg[:, feature_idx]
        else:
            # If feature_idx is None or segment doesn't have features, use the whole segment
            data = seg
        all_data.append(data)
    
    if not all_data:
        return {
            'mean': np.nan,
            'std': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    # Concatenate all data segments
    all_data = np.concatenate(all_data)
    
    # Calculate statistics and ensure they are scalar values
    mean = float(np.mean(all_data))
    std = float(np.std(all_data))
    skewness_val = skew(all_data)
    kurtosis_val = kurtosis(all_data)
    
    # Convert to scalar if they're arrays
    if isinstance(skewness_val, np.ndarray):
        skewness_val = float(np.mean(skewness_val))
    else:
        skewness_val = float(skewness_val)
        
    if isinstance(kurtosis_val, np.ndarray):
        kurtosis_val = float(np.mean(kurtosis_val))
    else:
        kurtosis_val = float(kurtosis_val)
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val
    }

def format_condition_name(condition):
    """
    Format condition name for display in the table.
    Replace underscores with spaces and capitalize each word.
    """
    if condition.lower() == "normal":
        return "Normal"
    
    words = condition.split('_')
    return ' '.join(word.capitalize() for word in words)

def generate_latex_table_by_condition_percent(stats_data):
    """
    Generate LaTeX table comparing statistics across all conditions.
    For the Normal condition, absolute values are displayed.
    For all other conditions, the table shows the percentage difference relative
    to the Normal condition.
    
    Parameters:
    -----------
    stats_data : dict
        Dictionary with statistics data per condition, where each condition
        maps to a dictionary of architectures; statistics are averaged across architectures.
        
    Returns:
    --------
    str
        LaTeX table code
    """
    # Aggregate statistics across architectures for each condition
    aggregated = {}
    for condition, arch_dict in stats_data.items():
        means = []
        stds = []
        skews = []
        kurts = []
        for arch, stats in arch_dict.items():
            means.append(stats["mean"])
            stds.append(stats["std"])
            skews.append(stats["skewness"])
            kurts.append(stats["kurtosis"])
        aggregated[condition] = {
            "mean": np.nanmean(means),
            "std": np.nanmean(stds),
            "skewness": np.nanmean(skews),
            "kurtosis": np.nanmean(kurts)
        }
    
    # Get baseline from the Normal condition
    baseline = aggregated.get("normal", None)
    if baseline is None:
        raise ValueError("Normal condition stats not found.")
    
    # Fixed ordering of condition groups as specified
    condition_order = [
        "normal",
        "horizontal_misalignment_fault",
        "imbalance_fault",
        "overhang_ball_fault",
        "overhang_cage_fault",
        "overhang_outer_race_fault",
        "underhang_ball_fault",
        "underhang_cage_fault",
        "underhang_outer_race_fault",
        "vertical_misalignment_fault"
    ]
    
    latex_code = r"""\begin{table}[h!]
    \centering
    \caption{Residual Distribution Metrics: Percentage Change Relative to Normal Condition}
    \label{tab:residual_distribution_comparison}
    \begin{tabular}{l|c|c|c|c}
        \hline
        Condition & $\mu$ (\%) & $\sigma$ (\%) & Skew. (\%) & Kurt. (\%) \\
        \hline
"""
    # Function to compute percentage change.
    def pct_change(val, base):
        if base == 0:
            return 0.0
        return ((val - base) / abs(base)) * 100
    
    # Loop in the fixed order and generate rows.
    for condition in condition_order:
        if condition not in aggregated:
            continue
        formatted_condition = format_condition_name(condition)
        stats = aggregated[condition]
        if condition == "normal":
            row = f"        {formatted_condition} & {stats['mean']:.1f} & {stats['std']:.1f} & {stats['skewness']:.4f} & {stats['kurtosis']:.4f} \\\\\n"
        else:
            mean_pct = pct_change(stats["mean"], baseline["mean"])
            std_pct = pct_change(stats["std"], baseline["std"])
            skew_pct = pct_change(stats["skewness"], baseline["skewness"])
            kurt_pct = pct_change(stats["kurtosis"], baseline["kurtosis"])
            row = (f"        {formatted_condition} & {'+' if mean_pct>=0 else ''}{mean_pct:.1f}\\% "
                   f"& {'+' if std_pct>=0 else ''}{std_pct:.1f}\\% "
                   f"& {'+' if skew_pct>=0 else ''}{skew_pct:.1f}\\% "
                   f"& {'+' if kurt_pct>=0 else ''}{kurt_pct:.1f}\\% \\\\\n")
        latex_code += row

    latex_code += r"""        \hline
    \end{tabular}
\end{table}
"""
    return latex_code

def extract_architecture_from_key(key):
    """
    Extract the architecture type from the residual key.
    This is a placeholder function - adjust based on your actual key format.
    """
    if "pbr" in key.lower():
        return "PBR"
    elif "hrpinn" in key.lower() or "hybrid" in key.lower():
        return "HRPINN"
    else:
        return "PBR"  # Default to PBR instead of Unknown

def extract_condition_from_key(key):
    """
    Extract the condition category from a key based on the fixed groups.
    Returns one of:
      horizontal_misalignment_fault,
      imbalance_fault,
      normal,
      overhang_ball_fault,
      overhang_cage_fault,
      overhang_outer_race_fault,
      underhang_ball_fault,
      underhang_cage_fault,
      underhang_outer_race_fault,
      vertical_misalignment_fault
    """
    key_lower = key.lower()
    mapping = {
        "horizontal_misalignment_fault": "horizontal_misalignment_fault",
        "imbalance_fault": "imbalance_fault",
        "overhang_ball_fault": "overhang_ball_fault",
        "overhang_cage_fault": "overhang_cage_fault",
        "overhang_outer_race_fault": "overhang_outer_race_fault",
        "underhang_ball_fault": "underhang_ball_fault",
        "underhang_cage_fault": "underhang_cage_fault",
        "underhang_outer_race_fault": "underhang_outer_race_fault",
        "vertical_misalignment_fault": "vertical_misalignment_fault",
        "normal": "normal"
    }
    for keyword, condition in mapping.items():
        if keyword in key_lower:
            return condition
    return "unknown_condition"

def map_composite_key_to_original(composite_key, original_keys):
    """
    Map a composite key (condition_arch_index) back to the original key in data_paths.
    """
    parts = composite_key.split('_')
    if len(parts) < 3:
        return None
    condition = parts[0]
    matching_keys = []
    for orig_key in original_keys:
        extracted_condition = extract_condition_from_key(orig_key)
        if extracted_condition == condition:
            matching_keys.append(orig_key)
    return matching_keys[0] if matching_keys else None

def average_statistics_across_speeds(stats_by_speed):
    """
    Average statistics across all rotation speeds for a condition.
    """
    if not stats_by_speed:
        return {
            'mean': np.nan,
            'std': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    all_means = []
    all_stds = []
    all_skewness = []
    all_kurtosis = []
    
    for speed, stats in stats_by_speed.items():
        all_means.append(stats['mean'])
        all_stds.append(stats['std'])
        all_skewness.append(stats['skewness'])
        all_kurtosis.append(stats['kurtosis'])
    
    return {
        'mean': float(np.nanmean(all_means)),
        'std': float(np.nanmean(all_stds)),
        'skewness': float(np.nanmean(all_skewness)),
        'kurtosis': float(np.nanmean(all_kurtosis))
    }

if __name__ == "__main__":
    # Load the residuals dictionary
    print("Loading residuals...")
    #residuals_dict = torch.load('residuals_dict.pth')
    residuals_dict = torch.load('results/random_search/best_model/best_model_residuals.pth')
    
    # Get all unique conditions from keys
    print("Extracting conditions and architectures...")
    all_keys = list(residuals_dict.keys())
    condition_keys = defaultdict(list)
    for key in all_keys:
        condition = extract_condition_from_key(key)
        condition_keys[condition].append(key)
    
    print(f"Found {len(condition_keys)} conditions: {list(condition_keys.keys())}")
    
    # Prepare a dictionary to store all residuals by condition and architecture
    processed_residuals = {}
    for condition, keys in condition_keys.items():
        processed_residuals[condition] = {}
        for key in keys:
            architecture = extract_architecture_from_key(key)
            if architecture not in processed_residuals[condition]:
                processed_residuals[condition][architecture] = []
            processed_residuals[condition][architecture].append(residuals_dict[key])
            print(f"Added {key} to {condition} / {architecture}")
    
    # Create a composite key to original key mapping
    composite_to_original = {}
    original_keys = list(data_paths.keys())
    print(f"Available data paths: {original_keys}")
    
    for condition in processed_residuals:
        for arch in processed_residuals[condition]:
            for i, _ in enumerate(processed_residuals[condition][arch]):
                composite_key = f"{condition}_{arch}_{i}"
                for orig_key in original_keys:
                    if condition == extract_condition_from_key(orig_key):
                        composite_to_original[composite_key] = orig_key
                        break
                if composite_key not in composite_to_original:
                    print(f"Warning: Could not find a matching data path for {composite_key}")
    
    # Adjust segmentation based on rotation speed for all conditions
    print("Adjusting segments by rotation...")
    all_residuals = {}
    for condition in processed_residuals:
        for arch in processed_residuals[condition]:
            for i, residual in enumerate(processed_residuals[condition][arch]):
                key = f"{condition}_{arch}_{i}"
                all_residuals[key] = residual
    
    adjusted_residuals = adjust_segments_by_rotation(all_residuals, composite_to_original)
    
    # Process all adjusted residuals
    print("Processing residuals...")
    processed = process_residuals(adjusted_residuals, seq_length=100)
    
    # Reorganize processed residuals by condition and architecture using the extracted condition
    reorganized = defaultdict(lambda: defaultdict(list))
    for key, segments in processed.items():
        condition = extract_condition_from_key(key)
        parts = key.split('_')
        arch = parts[1] if len(parts) >= 2 else "unknown"
        reorganized[condition][arch].extend(segments)
    
    # Store statistics by condition and architecture (averaging across rotation speeds)
    stats_by_condition_and_arch = defaultdict(dict)
    for condition in reorganized:
        for arch in reorganized[condition]:
            segments = reorganized[condition][arch]
            if not segments:
                print(f"No segments for {condition}/{arch}, skipping...")
                continue
            groups = group_segments_by_rotation_speed(segments)
            stats_by_speed = {}
            for speed, segs in groups.items():
                stats = calculate_residual_statistics(segs)
                stats_by_speed[speed] = stats
            avg_stats = average_statistics_across_speeds(stats_by_speed)
            stats_by_condition_and_arch[condition][arch] = avg_stats
    
    # Generate LaTeX table comparing all conditions with percentage changes
    latex_table = generate_latex_table_by_condition_percent(stats_by_condition_and_arch)
    
    print("\nGenerated LaTeX Table:")
    print(latex_table)
    
    output_file = "residual_statistics_by_condition_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to {output_file}")
    
    full_output = r"""\subsubsection{Residual Analysis Across All Conditions}
To assess the modeling capabilities of the Physics-Based Regularized (PBR) and Hybrid Regularized (HRPINN) architectures across different operating conditions, we analyze the distribution of the generated residuals. Table \ref{tab:residual_distribution_comparison} presents the metrics (mean, standard deviation, skewness, kurtosis) of the residuals for both PINN architectures, averaged across all available rotation speeds for each condition. For the Normal condition, absolute values are reported, and for other conditions the values represent the percentage change relative to Normal.

""" + latex_table
    
    full_output_file = "residual_analysis_all_conditions_section.tex"
    with open(full_output_file, "w") as f:
        f.write(full_output)
    
    print(f"Full LaTeX section with context saved to {full_output_file}")

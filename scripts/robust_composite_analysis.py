#!/usr/bin/env python3
"""
Robust Composite Analysis of Real Bayesian Optimization Results

This script implements a robust approach to select the best trial for each method:
1. Collect all trials for each method
2. Eliminate damaged/invalid trials (NaN, Inf, extreme values)
3. Scale each raw component across all valid trials within each method
4. Average the scaled components to get a composite score
5. Select the best trial based on this composite score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.preprocessing import MinMaxScaler

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def filter_damaged_trials(df):
    """Filter out trials with damaged/invalid values."""
    
    # Define reasonable bounds for different metrics
    bounds = {
        'raw_total_mae_val': (0, 100),  # MAE should be positive and reasonable
        'raw_x2_ddot_mae_val': (0, 100),
        'raw_y2_ddot_mae_val': (0, 100),
        'raw_x3_ddot_mae_val': (0, 100),
        'raw_y3_ddot_mae_val': (0, 100),
        'raw_res1_mean_val': (-10000, 10000),  # Residuals can be negative but not extreme
        'raw_res2_mean_val': (-10000, 10000),
        'raw_res3_mean_val': (-10000, 10000),
        'raw_res4_mean_val': (-10000, 10000),
        'raw_res_mass1_mean_val': (-1000, 1000),
        'raw_res_mass2_mean_val': (-1000, 1000),
        'val_total_min': (-1e10, 1e10)  # Very wide bounds for validation loss
    }
    
    # Create a mask for valid trials
    valid_mask = pd.Series([True] * len(df), index=df.index)
    
    for column, (min_val, max_val) in bounds.items():
        if column in df.columns:
            # Check for NaN, Inf, and bounds
            column_mask = (
                df[column].notna() &
                np.isfinite(df[column]) &
                (df[column] >= min_val) &
                (df[column] <= max_val)
            )
            valid_mask = valid_mask & column_mask
    
    filtered_df = df[valid_mask].copy()
    
    print(f"Original trials: {len(df)}")
    print(f"Valid trials: {len(filtered_df)}")
    print(f"Filtered out: {len(df) - len(filtered_df)} damaged trials")
    
    return filtered_df

def scale_components_within_method(method_df):
    """Scale each component within a method's trials using MinMax scaling."""
    
    # Components to scale
    data_components = [
        'raw_x2_ddot_mae_val', 'raw_y2_ddot_mae_val', 
        'raw_x3_ddot_mae_val', 'raw_y3_ddot_mae_val', 'raw_total_mae_val'
    ]
    
    physics_components = [
        'raw_res1_mean_val', 'raw_res2_mean_val', 'raw_res3_mean_val',
        'raw_res4_mean_val', 'raw_res_mass1_mean_val', 'raw_res_mass2_mean_val'
    ]
    
    scaled_df = method_df.copy()
    
    # Scale data components (lower is better, so we invert)
    for component in data_components:
        if component in method_df.columns and len(method_df[component].dropna()) > 0:
            scaler = MinMaxScaler()
            values = method_df[component].dropna().values.reshape(-1, 1)
            if len(values) > 0:
                scaled_values = scaler.fit_transform(values)
                # Invert so that lower values (better performance) get higher scores
                scaled_values = 1 - scaled_values
                
                # Create a mapping for the original indices
                valid_indices = method_df[component].dropna().index
                for i, idx in enumerate(valid_indices):
                    scaled_df.loc[idx, f'{component}_scaled'] = scaled_values[i][0]
    
    # Scale physics components (closer to 0 is better, so we use absolute values)
    for component in physics_components:
        if component in method_df.columns and len(method_df[component].dropna()) > 0:
            scaler = MinMaxScaler()
            # Use absolute values for physics residuals
            abs_values = method_df[component].dropna().abs().values.reshape(-1, 1)
            if len(abs_values) > 0:
                scaled_values = scaler.fit_transform(abs_values)
                # Invert so that lower absolute values (better performance) get higher scores
                scaled_values = 1 - scaled_values
                
                # Create a mapping for the original indices
                valid_indices = method_df[component].dropna().index
                for i, idx in enumerate(valid_indices):
                    scaled_df.loc[idx, f'{component}_scaled'] = scaled_values[i][0]
    
    return scaled_df

def calculate_composite_score(scaled_df):
    """Calculate composite score by averaging all scaled components."""
    
    # Find all scaled columns
    scaled_columns = [col for col in scaled_df.columns if col.endswith('_scaled')]
    
    if not scaled_columns:
        print("Warning: No scaled columns found!")
        return scaled_df
    
    # Calculate composite score
    scaled_df['composite_score'] = scaled_df[scaled_columns].mean(axis=1)
    
    # Fill NaN values with 0 (worst score)
    scaled_df['composite_score'] = scaled_df['composite_score'].fillna(0)
    
    return scaled_df

def load_and_analyze_results():
    """Load and analyze the Bayesian optimization results with robust composite scoring."""
    
    # Load the corrected data
    all_trials_df = pd.read_csv('real_bayesian_raw_metrics_corrected.csv')
    
    print("="*80)
    print("ROBUST COMPOSITE ANALYSIS OF REAL BAYESIAN OPTIMIZATION RESULTS")
    print("="*80)
    
    # Filter out damaged trials
    print("\n🔍 FILTERING DAMAGED TRIALS...")
    valid_all_trials = filter_damaged_trials(all_trials_df)
    
    if len(valid_all_trials) == 0:
        print("No valid trials found!")
        return None, None, None
    
    # Process each method separately
    best_trials = []
    method_stats = []
    all_scaled_trials = []  # Store all scaled trials for final dataframe
    
    for method in valid_all_trials['method'].unique():
        print(f"\n📊 Processing {method}...")
        
        # Get trials for this method
        method_df = valid_all_trials[valid_all_trials['method'] == method].copy()
        
        if len(method_df) == 0:
            continue
        
        # Scale components within this method
        scaled_method_df = scale_components_within_method(method_df)
        
        # Calculate composite score
        scaled_method_df = calculate_composite_score(scaled_method_df)
        
        # Store all scaled trials for final dataframe
        all_scaled_trials.append(scaled_method_df)
        
        # Find best trial based on composite score (within method only)
        if 'composite_score' in scaled_method_df.columns:
            best_trial_idx = scaled_method_df['composite_score'].idxmax()
            best_trial = scaled_method_df.loc[best_trial_idx].copy()
            best_trial['rank'] = 1
            best_trial['selection_criteria'] = 'composite_score_within_method'
            best_trials.append(best_trial)
        
        # Calculate method statistics
        stats = {
            'method': method,
            'total_trials': len(method_df),
            'valid_trials': len(scaled_method_df),
            'best_composite_score': scaled_method_df['composite_score'].max() if 'composite_score' in scaled_method_df.columns else np.nan,
            'mean_composite_score': scaled_method_df['composite_score'].mean() if 'composite_score' in scaled_method_df.columns else np.nan,
            'std_composite_score': scaled_method_df['composite_score'].std() if 'composite_score' in scaled_method_df.columns else np.nan
        }
        method_stats.append(stats)
    
    if not best_trials:
        print("No valid best trials found!")
        return None, None, None
    
    # Combine all scaled trials into one dataframe
    if all_scaled_trials:
        valid_all_trials_with_scores = pd.concat(all_scaled_trials, ignore_index=True)
    else:
        valid_all_trials_with_scores = valid_all_trials.copy()
    
    best_trials_df = pd.DataFrame(best_trials)
    stats_df = pd.DataFrame(method_stats)
    
    # Get only the best trial from each method (rank == 1)
    best_trials_only = best_trials_df[best_trials_df['rank'] == 1].copy()
    
    # 1. Overall Performance Ranking (Raw Metrics)
    print("\n🏆 OVERALL PERFORMANCE RANKING (Best Trials by Composite Score, Ranked by Raw Total MAE)")
    print("-" * 80)
    
    best_trials_sorted = best_trials_only.sort_values('raw_total_mae_val')
    
    for i, (_, row) in enumerate(best_trials_sorted.iterrows()):
        print(f"{i+1:2d}. {row['method']:20s} | Raw MAE: {row['raw_total_mae_val']:8.4f} | "
              f"Composite: {row['composite_score']:8.4f} | Trial: {row['trial']} | SP: {row['starting_point']}")
    
    # 2. Data Component Rankings
    print("\n📊 DATA COMPONENT RANKINGS (Best Trials by Composite Score)")
    print("-" * 60)
    
    data_components = {
        'x2_ddot': 'raw_x2_ddot_mae_val',
        'y2_ddot': 'raw_y2_ddot_mae_val', 
        'x3_ddot': 'raw_x3_ddot_mae_val',
        'y3_ddot': 'raw_y3_ddot_mae_val',
        'total': 'raw_total_mae_val'
    }
    
    for component_name, column_name in data_components.items():
        print(f"\n{component_name.upper()} Component Ranking:")
        component_sorted = best_trials_sorted.sort_values(column_name)
        
        for i, (_, row) in enumerate(component_sorted.iterrows()):
            value = row[column_name] if pd.notna(row[column_name]) else float('inf')
            print(f"  {i+1:2d}. {row['method']:20s} | MAE: {value:8.4f}")
    
    # 3. Physics Residual Rankings
    print("\n⚛️ PHYSICS RESIDUAL RANKINGS (Best Trials by Composite Score, Mean Values)")
    print("-" * 80)
    
    physics_components = {
        'res1': 'raw_res1_mean_val',
        'res2': 'raw_res2_mean_val',
        'res3': 'raw_res3_mean_val', 
        'res4': 'raw_res4_mean_val',
        'mass1': 'raw_res_mass1_mean_val',
        'mass2': 'raw_res_mass2_mean_val'
    }
    
    for component_name, column_name in physics_components.items():
        print(f"\n{component_name.upper()} Residual Ranking:")
        # Sort by absolute value of residual (closer to 0 is better)
        best_trials_sorted[f'{column_name}_abs'] = best_trials_sorted[column_name].abs()
        component_sorted = best_trials_sorted.sort_values(f'{column_name}_abs')
        
        for i, (_, row) in enumerate(component_sorted.iterrows()):
            value = row[column_name] if pd.notna(row[column_name]) else float('inf')
            print(f"  {i+1:2d}. {row['method']:20s} | Residual: {value:11.2f}")
    
    # 4. Podium Analysis
    print("\n🏅 PODIUM ANALYSIS (Best Trials by Composite Score)")
    print("-" * 60)
    
    podium_counts = {}
    for component_name, column_name in data_components.items():
        component_sorted = best_trials_sorted.sort_values(column_name)
        podium_counts[component_name] = component_sorted['method'].head(3).tolist()
    
    for component_name, column_name in physics_components.items():
        best_trials_sorted[f'{column_name}_abs'] = best_trials_sorted[column_name].abs()
        component_sorted = best_trials_sorted.sort_values(f'{column_name}_abs')
        podium_counts[component_name] = component_sorted['method'].head(3).tolist()
    
    # Count podium appearances for each method
    method_podium_counts = {}
    for method in best_trials_sorted['method'].unique():
        method_podium_counts[method] = 0
    
    for component, podium in podium_counts.items():
        print(f"\n{component.upper()} Podium:")
        for i, method in enumerate(podium):
            print(f"  {i+1}. {method}")
            method_podium_counts[method] += 1
    
    print(f"\n🏆 TOTAL PODIUM APPEARANCES:")
    sorted_methods = sorted(method_podium_counts.items(), key=lambda x: x[1], reverse=True)
    for method, count in sorted_methods:
        print(f"  {method:20s}: {count:2d} podium appearances")
    
    # 5. Success Rate Analysis
    print("\n📊 SUCCESS RATE ANALYSIS")
    print("-" * 60)
    
    print("Method Success Rates (Valid Trials / Total Trials):")
    for _, row in stats_df.iterrows():
        print(f"{row['method']:<20} | {row['valid_trials']:3d}/{row['total_trials']:3d} "
              f"| Best Composite: {row['best_composite_score']:8.4f} | "
              f"Mean Composite: {row['mean_composite_score']:8.4f}")
    
    return best_trials_df, valid_all_trials_with_scores, stats_df, podium_counts, method_podium_counts

def create_component_rankings(best_trials):
    """Create component-wise rankings for all metrics."""
    
    # Data components
    data_components = {
        'x2_ddot': 'raw_x2_ddot_mae_val',
        'y2_ddot': 'raw_y2_ddot_mae_val', 
        'x3_ddot': 'raw_x3_ddot_mae_val',
        'y3_ddot': 'raw_y3_ddot_mae_val',
        'total': 'raw_total_mae_val'
    }
    
    # Physics components
    physics_components = {
        'res1': 'raw_res1_mean_val',
        'res2': 'raw_res2_mean_val',
        'res3': 'raw_res3_mean_val', 
        'res4': 'raw_res4_mean_val',
        'mass1': 'raw_res_mass1_mean_val',
        'mass2': 'raw_res_mass2_mean_val'
    }
    
    rankings = {}
    
    # Get best trials only
    best_trials_only = best_trials[best_trials['rank'] == 1].copy()
    
    # Create rankings for data components
    for comp_name, col_name in data_components.items():
        sorted_data = best_trials_only.sort_values(col_name)
        rankings[comp_name] = sorted_data[['method', col_name]].copy()
        rankings[comp_name]['rank'] = range(1, len(sorted_data) + 1)
    
    # Create rankings for physics components (sort by absolute value)
    for comp_name, col_name in physics_components.items():
        best_trials_only[f'{col_name}_abs'] = best_trials_only[col_name].abs()
        sorted_data = best_trials_only.sort_values(f'{col_name}_abs')
        rankings[comp_name] = sorted_data[['method', col_name]].copy()
        rankings[comp_name]['rank'] = range(1, len(sorted_data) + 1)
    
    return rankings

def create_visualizations(best_trials_df, all_trials_df, stats_df, podium_counts, method_podium_counts):
    """Create comprehensive visualizations focusing on raw metrics and podium analysis."""
    
    # Get best trials only
    best_trials = best_trials_df[best_trials_df['rank'] == 1].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Robust Composite Analysis - Real Bayesian Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. Overall Performance Ranking (Raw Total MAE)
    ax1 = axes[0, 0]
    best_trials_sorted = best_trials.sort_values('raw_total_mae_val')
    methods = best_trials_sorted['method']
    raw_mae_scores = best_trials_sorted['raw_total_mae_val']
    
    bars = ax1.barh(range(len(methods)), raw_mae_scores, color='lightcoral')
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xlabel('Raw Total MAE')
    ax1.set_title('Best Performance by Method (Raw Total MAE)')
    ax1.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, raw_mae_scores)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=9)
    
    # 2. Podium Appearances
    ax2 = axes[0, 1]
    methods_podium = list(method_podium_counts.keys())
    podium_counts_values = list(method_podium_counts.values())
    
    bars = ax2.bar(range(len(methods_podium)), podium_counts_values, color='gold')
    ax2.set_xticks(range(len(methods_podium)))
    ax2.set_xticklabels(methods_podium, rotation=45, ha='right')
    ax2.set_ylabel('Podium Appearances')
    ax2.set_title('Total Podium Appearances by Method')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, podium_counts_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    # 3. Data Component Performance
    ax3 = axes[0, 2]
    components = ['raw_x2_ddot_mae_val', 'raw_y2_ddot_mae_val', 
                 'raw_x3_ddot_mae_val', 'raw_y3_ddot_mae_val']
    component_names = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
    
    x = np.arange(len(component_names))
    width = 0.8 / len(best_trials)
    
    for i, (_, row) in enumerate(best_trials.iterrows()):
        values = [row[comp] if comp in row and pd.notna(row[comp]) else 0 for comp in components]
        ax3.bar(x + i*width, values, width, label=row['method'], alpha=0.8)
    
    ax3.set_xlabel('Data Components')
    ax3.set_ylabel('MAE')
    ax3.set_title('Data Component Performance (Log Scale)')
    ax3.set_xticks(x + width * (len(best_trials) - 1) / 2)
    ax3.set_xticklabels(component_names)
    ax3.set_yscale('log')  # Use log scale for better visualization
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Physics Residual Performance
    ax4 = axes[1, 0]
    phys_components = ['raw_res1_mean_val', 'raw_res2_mean_val', 
                      'raw_res3_mean_val', 'raw_res4_mean_val']
    phys_names = ['Res1', 'Res2', 'Res3', 'Res4']
    
    x = np.arange(len(phys_names))
    width = 0.8 / len(best_trials)
    
    for i, (_, row) in enumerate(best_trials.iterrows()):
        values = [abs(row[comp]) if comp in row and pd.notna(row[comp]) else 0 for comp in phys_components]
        ax4.bar(x + i*width, values, width, label=row['method'], alpha=0.8)
    
    ax4.set_xlabel('Physics Residuals')
    ax4.set_ylabel('|Residual|')
    ax4.set_title('Physics Residual Performance (Log Scale)')
    ax4.set_xticks(x + width * (len(best_trials) - 1) / 2)
    ax4.set_xticklabels(phys_names)
    ax4.set_yscale('log')  # Use log scale for better visualization
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Success Rate Comparison
    ax5 = axes[1, 1]
    success_rates = (stats_df['valid_trials'] / stats_df['total_trials'] * 100).values
    method_names = stats_df['method']
    
    bars = ax5.bar(range(len(method_names)), success_rates, color='lightgreen')
    ax5.set_xticks(range(len(method_names)))
    ax5.set_xticklabels(method_names, rotation=45, ha='right')
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Method Success Rates')
    ax5.set_ylim(0, 100)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, success_rates)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 6. Raw MAE vs Composite Score (within method)
    ax6 = axes[1, 2]
    scatter_data = best_trials[['raw_total_mae_val', 'composite_score']].dropna()
    
    if len(scatter_data) > 0:
        ax6.scatter(scatter_data['raw_total_mae_val'], scatter_data['composite_score'], 
                   alpha=0.7, s=100)
        
        # Add method labels
        for _, row in best_trials.iterrows():
            if pd.notna(row['raw_total_mae_val']) and pd.notna(row['composite_score']):
                ax6.annotate(row['method'], 
                            (row['raw_total_mae_val'], row['composite_score']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('Raw Total MAE')
    ax6.set_ylabel('Composite Score (within method)')
    ax6.set_title('Raw MAE vs Composite Score')
    
    plt.tight_layout()
    plt.savefig('robust_composite_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualizations saved to 'robust_composite_analysis.png'")

def generate_component_rankings_report(best_trials, rankings, podium_counts, method_podium_counts):
    """Generate a comprehensive component rankings report with podium analysis."""
    
    best_trials_only = best_trials[best_trials['rank'] == 1].copy()
    
    report = {
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_performing_method": best_trials_only.sort_values('raw_total_mae_val')['method'].iloc[0],
        "best_raw_mae": best_trials_only.sort_values('raw_total_mae_val')['raw_total_mae_val'].iloc[0],
        "podium_analysis": {
            "method_podium_counts": method_podium_counts,
            "component_podiums": podium_counts
        },
        "component_rankings": {},
        "summary": {}
    }
    
    # Add rankings for each component
    for component_name, ranking_df in rankings.items():
        report["component_rankings"][component_name] = ranking_df.to_dict('records')
    
    # Create summary statistics
    summary = {}
    for component_name, ranking_df in rankings.items():
        # Find which method ranks first for each component
        top_method = ranking_df.iloc[0]['method']
        top_value = ranking_df.iloc[0][ranking_df.columns[1]]  # Second column is the metric
        summary[component_name] = {
            "best_method": top_method,
            "best_value": float(top_value)
        }
    
    report["summary"] = summary
    
    # Save report
    with open('robust_composite_rankings.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Component rankings report saved to 'robust_composite_rankings.json'")
    
    return report

def main():
    """Main analysis function."""
    
    print("Starting robust composite analysis of real Bayesian optimization results...")
    print("Using composite score for within-method selection, raw metrics for cross-method comparison...")
    
    # Load and analyze results
    result = load_and_analyze_results()
    
    if result is None:
        print("No valid results found!")
        return
    
    best_trials_df, all_trials_df, stats_df, podium_counts, method_podium_counts = result
    
    # Create component rankings
    rankings = create_component_rankings(best_trials_df)
    
    # Create visualizations
    create_visualizations(best_trials_df, all_trials_df, stats_df, podium_counts, method_podium_counts)
    
    # Generate component rankings report
    report = generate_component_rankings_report(best_trials_df, rankings, podium_counts, method_podium_counts)
    
    print("\n" + "="*80)
    print("ROBUST COMPOSITE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Total valid trials analyzed: {len(all_trials_df)}")
    print(f"🏆 Best method (raw MAE): {report['best_performing_method']}")
    print(f"📈 Best raw MAE: {report['best_raw_mae']:.4f}")
    print(f"📋 Components ranked: {len(rankings)}")
    print("="*80)
    
    # Print podium winners
    print("\n🏅 PODIUM WINNERS:")
    sorted_podium = sorted(method_podium_counts.items(), key=lambda x: x[1], reverse=True)
    for method, count in sorted_podium:
        print(f"  {method:20s}: {count:2d} podium appearances")
    
    # Print component winners
    print("\n🏅 COMPONENT WINNERS:")
    for component, data in report["summary"].items():
        print(f"  {component:10s}: {data['best_method']:15s} (value: {data['best_value']:8.4f})")

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Visualize Bayesian optimization results.

This script creates plots to compare the results across all methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_comparison_plots(csv_file: str):
    """Create comparison plots from the CSV results."""
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Filter out failed methods for plotting
    df_plot = df[df['Best Validation Loss'] != 'Failed'].copy()
    df_plot['Best Validation Loss'] = pd.to_numeric(df_plot['Best Validation Loss'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bayesian Optimization Results Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation Loss Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(df_plot['Method'], df_plot['Best Validation Loss'], 
                   color=sns.color_palette("husl", len(df_plot)))
    ax1.set_title('Best Validation Loss by Method', fontweight='bold')
    ax1.set_ylabel('Validation Loss (lower is better)')
    ax1.set_yscale('log')  # Log scale for better visualization
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, df_plot['Best Validation Loss']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Success Rate Comparison
    ax2 = axes[0, 1]
    success_rates = [float(rate.strip('%'))/100 for rate in df['Success Rate']]
    bars2 = ax2.bar(df['Method'], success_rates, 
                    color=sns.color_palette("Set2", len(df)))
    ax2.set_title('Success Rate by Method', fontweight='bold')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, rate in zip(bars2, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Architecture Distribution
    ax3 = axes[1, 0]
    arch_counts = df['Best Architecture'].value_counts()
    ax3.pie(arch_counts.values, labels=arch_counts.index, autopct='%1.1f%%')
    ax3.set_title('Distribution of Best Architectures', fontweight='bold')
    
    # Plot 4: Activation Function Distribution
    ax4 = axes[1, 1]
    activation_counts = df['Best Activation'].value_counts()
    bars4 = ax4.bar(activation_counts.index, activation_counts.values,
                    color=sns.color_palette("viridis", len(activation_counts)))
    ax4.set_title('Distribution of Best Activation Functions', fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for bar, count in zip(bars4, activation_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary table
    print("\n" + "="*80)
    print("SUMMARY OF BAYESIAN OPTIMIZATION RESULTS")
    print("="*80)
    
    # Sort by validation loss
    df_sorted = df_plot.sort_values('Best Validation Loss')
    
    print(f"\n🏆 TOP 3 METHODS:")
    for i, (_, row) in enumerate(df_sorted.head(3).iterrows(), 1):
        print(f"{i}. {row['Method']}: {row['Best Validation Loss']:.6f} (Success: {row['Success Rate']})")
    
    print(f"\n📊 METHOD RANKINGS:")
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:2d}. {row['Method']:<15} | Loss: {row['Best Validation Loss']:>10.6f} | Success: {row['Success Rate']}")
    
    # Statistics
    print(f"\n📈 STATISTICS:")
    print(f"   Average Validation Loss: {df_plot['Best Validation Loss'].mean():.6f}")
    print(f"   Median Validation Loss: {df_plot['Best Validation Loss'].median():.6f}")
    print(f"   Best Validation Loss: {df_plot['Best Validation Loss'].min():.6f} ({df_sorted.iloc[0]['Method']})")
    print(f"   Worst Validation Loss: {df_plot['Best Validation Loss'].max():.6f} ({df_sorted.iloc[-1]['Method']})")
    
    # Most common configurations
    print(f"\n🔧 MOST COMMON CONFIGURATIONS:")
    print(f"   Best Architecture: {df['Best Architecture'].mode().iloc[0] if not df['Best Architecture'].mode().empty else 'N/A'}")
    print(f"   Best Activation: {df['Best Activation'].mode().iloc[0] if not df['Best Activation'].mode().empty else 'N/A'}")
    
    return df_sorted

def main():
    """Main function."""
    csv_file = "bayesian_optimization_comparison.csv"
    
    if not Path(csv_file).exists():
        print(f"❌ CSV file '{csv_file}' not found!")
        print("Please run the comparison script first:")
        print("python compare_bayesian_results.py")
        return
    
    print("📊 Creating visualization plots...")
    try:
        df_sorted = create_comparison_plots(csv_file)
        print(f"\n✅ Visualization saved to: bayesian_optimization_comparison.png")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Make sure matplotlib and seaborn are installed:")
        print("pip install matplotlib seaborn")

if __name__ == "__main__":
    main() 
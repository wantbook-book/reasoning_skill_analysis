

#!/usr/bin/env python3
"""
Visualize SAE activation analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse


def plot_activation_distribution(reasoning_acts, normal_acts, output_path):
    """Plot activation distribution comparison."""
    plt.figure(figsize=(15, 5))
    
    # Flatten
    if len(reasoning_acts.shape) == 3:
        reasoning_acts = reasoning_acts.reshape(-1, reasoning_acts.shape[-1])
    if len(normal_acts.shape) == 3:
        normal_acts = normal_acts.reshape(-1, normal_acts.shape[-1])
    
    # Subplot 1: active feature count distribution
    plt.subplot(1, 3, 1)
    reasoning_active = (reasoning_acts != 0).sum(axis=1)
    normal_active = (normal_acts != 0).sum(axis=1)
    
    plt.hist(reasoning_active, bins=50, alpha=0.5, label='Reasoning', density=True)
    plt.hist(normal_active, bins=50, alpha=0.5, label='Normal', density=True)
    plt.xlabel('Number of Active Features')
    plt.ylabel('Density')
    plt.title('Active Features Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: sparsity comparison
    plt.subplot(1, 3, 2)
    reasoning_sparsity = 1 - (np.count_nonzero(reasoning_acts) / reasoning_acts.size)
    normal_sparsity = 1 - (np.count_nonzero(normal_acts) / normal_acts.size)
    
    categories = ['Reasoning', 'Normal']
    sparsities = [reasoning_sparsity * 100, normal_sparsity * 100]
    bars = plt.bar(categories, sparsities, color=['#FF6B6B', '#4ECDC4'])
    plt.ylabel('Sparsity (%)')
    plt.title('Sparsity Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars, sparsities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom')
    
    # Subplot 3: non-zero activation distribution
    plt.subplot(1, 3, 3)
    reasoning_nonzero = reasoning_acts[reasoning_acts != 0]
    normal_nonzero = normal_acts[normal_acts != 0]
    
    # Use log scale
    plt.hist(reasoning_nonzero, bins=50, alpha=0.5, label='Reasoning', density=True)
    plt.hist(normal_nonzero, bins=50, alpha=0.5, label='Normal', density=True)
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.title('Non-zero Activation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved activation distribution plot: {output_path}")
    plt.close()


def plot_feature_usage_comparison(reasoning_acts, normal_acts, output_path, top_n=50):
    """Plot feature usage frequency comparison."""
    # Flatten
    if len(reasoning_acts.shape) == 3:
        reasoning_acts = reasoning_acts.reshape(-1, reasoning_acts.shape[-1])
    if len(normal_acts.shape) == 3:
        normal_acts = normal_acts.reshape(-1, normal_acts.shape[-1])
    
    # Compute feature usage frequency
    reasoning_freq = (reasoning_acts != 0).sum(axis=0) / reasoning_acts.shape[0]
    normal_freq = (normal_acts != 0).sum(axis=0) / normal_acts.shape[0]
    
    feature_diff = reasoning_freq - normal_freq
    
    # Identify most discriminative features
    top_reasoning = np.argsort(feature_diff)[-top_n:][::-1]
    top_normal = np.argsort(feature_diff)[:top_n]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Subplot 1: most reasoning-specific features
    ax = axes[0]
    x = np.arange(len(top_reasoning))
    width = 0.35
    
    ax.bar(x - width/2, reasoning_freq[top_reasoning] * 100, width, 
           label='Reasoning', color='#FF6B6B', alpha=0.8)
    ax.bar(x + width/2, normal_freq[top_reasoning] * 100, width,
           label='Normal', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Usage Frequency (%)')
    ax.set_title(f'Top {top_n} Reasoning-Specific Features')
    ax.set_xticks(x)
    ax.set_xticklabels(top_reasoning, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: most normal-specific features
    ax = axes[1]
    x = np.arange(len(top_normal))
    
    ax.bar(x - width/2, reasoning_freq[top_normal] * 100, width,
           label='Reasoning', color='#FF6B6B', alpha=0.8)
    ax.bar(x + width/2, normal_freq[top_normal] * 100, width,
           label='Normal', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Usage Frequency (%)')
    ax.set_title(f'Top {top_n} Normal-Specific Features')
    ax.set_xticks(x)
    ax.set_xticklabels(top_normal, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature usage comparison plot: {output_path}")
    plt.close()


def plot_multi_token_statistics(multi_matches, output_path):
    """Plot multi-token sequence statistics."""
    if multi_matches is None:
        print("Multi-token match info not found, skipping plot")
        return
    
    match_stats = multi_matches['match_statistics']
    
    # Sort by frequency, take top 20
    sorted_stats = sorted(match_stats.items(), key=lambda x: x[1], reverse=True)[:20]
    patterns = [s[0] for s in sorted_stats]
    counts = [s[1] for s in sorted_stats]
    
    # Compute percentages
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: frequency bar chart
    ax = axes[0]
    y_pos = np.arange(len(patterns))
    bars = ax.barh(y_pos, counts, color='#95E1D3')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(patterns, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Match Count')
    ax.set_title('Top 20 Multi-Token Patterns by Frequency')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count:,}',
                ha='left', va='center', fontsize=8)
    
    # Subplot 2: percentage pie chart
    ax = axes[1]
    # Show top 10, merge the rest as "Others"
    if len(patterns) > 10:
        pie_labels = patterns[:10] + ['Others']
        pie_counts = counts[:10] + [sum(counts[10:])]
    else:
        pie_labels = patterns
        pie_counts = counts
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
    wedges, texts, autotexts = ax.pie(pie_counts, labels=None, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    
    # Add legend
    ax.legend(wedges, pie_labels, title="Patterns",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=8)
    
    ax.set_title('Distribution of Multi-Token Patterns')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multi-token statistics plot: {output_path}")
    plt.close()


def plot_feature_diff_heatmap(reasoning_acts, normal_acts, output_path, n_samples=100):
    """Plot feature difference heatmap (sampled)."""
    # Flatten
    if len(reasoning_acts.shape) == 3:
        reasoning_acts = reasoning_acts.reshape(-1, reasoning_acts.shape[-1])
    if len(normal_acts.shape) == 3:
        normal_acts = normal_acts.reshape(-1, normal_acts.shape[-1])
    
    # Random sampling (avoid huge data)
    reasoning_sample_idx = np.random.choice(reasoning_acts.shape[0], 
                                           min(n_samples, reasoning_acts.shape[0]), 
                                           replace=False)
    normal_sample_idx = np.random.choice(normal_acts.shape[0],
                                        min(n_samples, normal_acts.shape[0]),
                                        replace=False)
    
    reasoning_sample = reasoning_acts[reasoning_sample_idx]
    normal_sample = normal_acts[normal_sample_idx]
    
    # Identify most discriminative features
    reasoning_freq = (reasoning_acts != 0).sum(axis=0) / reasoning_acts.shape[0]
    normal_freq = (normal_acts != 0).sum(axis=0) / normal_acts.shape[0]
    feature_diff = np.abs(reasoning_freq - normal_freq)
    
    top_features = np.argsort(feature_diff)[-50:][::-1]  # Top 50 most discriminative features
    
    # Combine data
    combined = np.vstack([reasoning_sample[:, top_features], normal_sample[:, top_features]])
    labels = ['R'] * len(reasoning_sample) + ['N'] * len(normal_sample)
    
    # Plot heatmap
    plt.figure(figsize=(20, 10))
    
    # Create row color labels
    row_colors = ['#FF6B6B' if l == 'R' else '#4ECDC4' for l in labels]
    
    sns.clustermap(combined, cmap='YlOrRd', 
                   row_colors=row_colors,
                   col_cluster=True, row_cluster=True,
                   figsize=(20, 12),
                   cbar_kws={'label': 'Activation Value'},
                   xticklabels=top_features,
                   yticklabels=False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature difference heatmap: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE activation analysis results.")
    parser.add_argument('--output-dir', type=str, required=True, help='Analysis output directory.')
    parser.add_argument('--plot-dir', type=str, default=None, 
                       help='Plot output directory (default: output-dir/plots).')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SAE Activation Visualization")
    print("="*60)
    print(f"Data directory: {output_dir}")
    print(f"Plot directory: {plot_dir}")
    
    # Styling
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Load data
    print("\nLoading activation data...")
    
    try:
        # Load reasoning activations
        reasoning_file = output_dir / "sae_activations_reasoning.npy"
        if reasoning_file.exists():
            reasoning_acts = np.load(reasoning_file)
            print(f"✓ Reasoning activations: {reasoning_acts.shape}")
        else:
            print("✗ Reasoning activation file not found")
            reasoning_acts = None
        
        # Load normal activations
        normal_file = output_dir / "sae_activations_normal.npy"
        if normal_file.exists():
            normal_acts = np.load(normal_file)
            print(f"✓ Normal activations: {normal_acts.shape}")
        else:
            print("✗ Normal activation file not found")
            normal_acts = None
        
        # Load multi-token match info
        multi_match_file = output_dir / "multi_token_matches.json"
        if multi_match_file.exists():
            with open(multi_match_file, 'r') as f:
                multi_matches = json.load(f)
            print("✓ Multi-token match info")
        else:
            print("✗ Multi-token match info not found")
            multi_matches = None
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Generate plots
    print("\nGenerating plots...")
    
    if reasoning_acts is not None and normal_acts is not None:
        # 1. Activation distribution comparison
        plot_activation_distribution(
            reasoning_acts, normal_acts,
            plot_dir / "activation_distribution.png"
        )
        
        # 2. Feature usage comparison
        plot_feature_usage_comparison(
            reasoning_acts, normal_acts,
            plot_dir / "feature_usage_comparison.png",
            top_n=30
        )
        
        # 3. Feature difference heatmap
        plot_feature_diff_heatmap(
            reasoning_acts, normal_acts,
            plot_dir / "feature_diff_heatmap.png",
            n_samples=100
        )
    
    # 4. Multi-token statistics plot
    if multi_matches is not None:
        plot_multi_token_statistics(
            multi_matches,
            plot_dir / "multi_token_statistics.png"
        )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nPlots saved to: {plot_dir}")
    print("File list:")
    for plot_file in sorted(plot_dir.glob("*.png")):
        print(f"  - {plot_file.name}")
    
    return 0


if __name__ == "__main__":
    exit(main())

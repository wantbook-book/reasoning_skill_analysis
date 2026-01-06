#!/usr/bin/env python3
"""
Script to analyze SAE activation vectors (supports multi-token sequences).
"""

import numpy as np
import h5py
import json
import argparse
from pathlib import Path
from collections import Counter


def load_multi_token_info(output_dir):
    """Load multi-token match information."""
    match_file = Path(output_dir) / "multi_token_matches.json"
    if match_file.exists():
        with open(match_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def analyze_multi_token_patterns(output_dir):
    """Analyze multi-token sequence match patterns."""
    multi_info= load_multi_token_info(output_dir)
    
    if multi_info is None:
        print("\nNo multi-token match info found")
        return
    
    print(f"\n{'='*60}")
    print("Multi-Token Sequence Analysis")
    print(f"{'='*60}")
    
    total_matches = multi_info['total_matches']
    match_stats = multi_info['match_statistics']
    
    print(f"\nTotal matches: {total_matches:,}")
    print(f"Unique sequences: {len(match_stats)}")
    
    # Sort by frequency
    sorted_patterns = sorted(match_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost common multi-token sequences (Top 20):")
    print("-" * 60)
    for i, (pattern, count) in enumerate(sorted_patterns[:20], 1):
        percentage = count / total_matches * 100
        print(f"{i:2d}. '{pattern}': {count:,} times ({percentage:.2f}%)")
    
    # Analyze length distribution
    if 'all_matches' in multi_info and multi_info['all_matches']:
        token_lengths = Counter()
        for match in multi_info['all_matches']:
            length = len(match['token_ids'])
            token_lengths[length] += 1
        
        print("\nMulti-token sequence length distribution:")
        print("-" * 60)
        for length in sorted(token_lengths.keys()):
            count = token_lengths[length]
            percentage = count / len(multi_info['all_matches']) * 100
            print(f"  {length} tokens: {count:,} times ({percentage:.2f}%)")
    
    return match_stats


def load_activations(output_dir, file_type='all'):
    """
    Load activation vectors.
    
    Args:
        output_dir: output directory
        file_type: 'all', 'reasoning', 'normal'
    """
    output_dir = Path(output_dir)
    
    # Try HDF5
    h5_file = output_dir / f"sae_activations_{file_type}.h5"
    if h5_file.exists():
        print(f"Loading from HDF5: {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            acts = f['activations'][:]
            attrs = dict(f.attrs)
        return acts, attrs
    
    # Try NumPy
    npy_file = output_dir / f"sae_activations_{file_type}.npy"
    if npy_file.exists():
        print(f"Loading from NumPy: {npy_file}")
        acts = np.load(npy_file)
        return acts, {}
    
    raise FileNotFoundError(f"Activation file not found: {h5_file} or {npy_file}")


def analyze_activations(acts, name="Activations"):
    """Analyze activation vector statistics."""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    
    print(f"Shape: {acts.shape}")
    
    if len(acts.shape) == 3:
        # (samples, seq_len, features)
        print(f"  Samples: {acts.shape[0]:,}")
        print(f"  Sequence length: {acts.shape[1]}")
        print(f"  Features: {acts.shape[2]:,}")
        total_tokens = acts.shape[0] * acts.shape[1]
        
        # Flatten to (tokens, features)
        acts_2d = acts.reshape(-1, acts.shape[-1])
    elif len(acts.shape) == 2:
        # (tokens, features) - already flattened
        print(f"  Tokens: {acts.shape[0]:,}")
        print(f"  Features: {acts.shape[1]:,}")
        total_tokens = acts.shape[0]
        acts_2d = acts
    else:
        print("  Unsupported shape")
        return {}
    
    print("\nBasic stats:")
    print(f"  Min: {acts.min():.6f}")
    print(f"  Max: {acts.max():.6f}")
    print(f"  Mean: {acts.mean():.6f}")
    print(f"  Std: {acts.std():.6f}")
    print(f"  Median: {np.median(acts):.6f}")
    
    # Sparsity analysis
    non_zero = np.count_nonzero(acts)
    total_elements = acts.size
    sparsity = 1 - (non_zero / total_elements)
    
    print("\nSparsity analysis:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Non-zero elements: {non_zero:,}")
    print(f"  Sparsity: {sparsity*100:.2f}%")
    
    # Average active features per token
    active_features_per_token = (acts_2d != 0).sum(axis=1)
    print("\nActive features per token:")
    print(f"  Mean: {active_features_per_token.mean():.2f}")
    print(f"  Median: {np.median(active_features_per_token):.2f}")
    print(f"  Min: {active_features_per_token.min()}")
    print(f"  Max: {active_features_per_token.max()}")
    print(f"  Std: {active_features_per_token.std():.2f}")
    
    # Feature activation frequency
    feature_activation_counts = (acts_2d != 0).sum(axis=0)
    print("\nFeature activation frequency:")
    print(f"  Never active features: {(feature_activation_counts == 0).sum()}")
    print(f"  Always active features: {(feature_activation_counts == total_tokens).sum()}")
    print(f"  Mean activations: {feature_activation_counts.mean():.2f}")
    print(f"  Median activations: {np.median(feature_activation_counts):.2f}")
    
    # Activation value distribution (non-zero values)
    non_zero_acts = acts_2d[acts_2d != 0]
    if len(non_zero_acts) > 0:
        print("\nNon-zero activation distribution:")
        print(f"  Mean: {non_zero_acts.mean():.6f}")
        print(f"  Std: {non_zero_acts.std():.6f}")
        print(f"  Median: {np.median(non_zero_acts):.6f}")
        percentiles = np.percentile(non_zero_acts, [25, 50, 75, 90, 95, 99])
        print(f"  25th percentile: {percentiles[0]:.6f}")
        print(f"  50th percentile: {percentiles[1]:.6f}")
        print(f"  75th percentile: {percentiles[2]:.6f}")
        print(f"  90th percentile: {percentiles[3]:.6f}")
        print(f"  95th percentile: {percentiles[4]:.6f}")
        print(f"  99th percentile: {percentiles[5]:.6f}")
    
    return {
        'shape': acts.shape,
        'sparsity': sparsity,
        'mean': float(acts.mean()),
        'std': float(acts.std()),
        'avg_active_features': float(active_features_per_token.mean()),
        'feature_activation_counts': feature_activation_counts,
        'non_zero_acts': non_zero_acts
    }


def compare_activations(reasoning_acts, normal_acts):
    """Compare activations for reasoning vs normal tokens."""
    print(f"\n{'='*60}")
    print("Reasoning vs Normal Token Activations")
    print(f"{'='*60}")
    
    # Flatten to (tokens, features)
    if len(reasoning_acts.shape) == 3:
        reasoning_acts = reasoning_acts.reshape(-1, reasoning_acts.shape[-1])
    if len(normal_acts.shape) == 3:
        normal_acts = normal_acts.reshape(-1, normal_acts.shape[-1])
    
    print("\nSample counts:")
    print(f"  Reasoning tokens: {reasoning_acts.shape[0]:,}")
    print(f"  Normal tokens: {normal_acts.shape[0]:,}")
    print(f"  Ratio: 1:{normal_acts.shape[0]/reasoning_acts.shape[0]:.2f}")
    
    # Sparsity comparison
    reasoning_sparsity = 1 - (np.count_nonzero(reasoning_acts) / reasoning_acts.size)
    normal_sparsity = 1 - (np.count_nonzero(normal_acts) / normal_acts.size)
    
    print("\nSparsity:")
    print(f"  Reasoning: {reasoning_sparsity*100:.2f}%")
    print(f"  Normal: {normal_sparsity*100:.2f}%")
    print(f"  Difference: {(reasoning_sparsity - normal_sparsity)*100:.2f}%")
    
    # Average active features
    reasoning_active = (reasoning_acts != 0).sum(axis=1).mean()
    normal_active = (normal_acts != 0).sum(axis=1).mean()
    
    print("\nAverage active features:")
    print(f"  Reasoning: {reasoning_active:.2f}")
    print(f"  Normal: {normal_active:.2f}")
    print(f"  Ratio: {reasoning_active / normal_active:.2f}x")
    
    # Activation magnitude (non-zero values)
    reasoning_nonzero = reasoning_acts[reasoning_acts != 0]
    normal_nonzero = normal_acts[normal_acts != 0]
    
    if len(reasoning_nonzero) > 0 and len(normal_nonzero) > 0:
        reasoning_mean = reasoning_nonzero.mean()
        normal_mean = normal_nonzero.mean()
        
        print("\nMean of non-zero activations:")
        print(f"  Reasoning: {reasoning_mean:.6f}")
        print(f"  Normal: {normal_mean:.6f}")
        print(f"  Ratio: {reasoning_mean / normal_mean:.2f}x")
        
        print("\nStd of non-zero activations:")
        print(f"  Reasoning: {reasoning_nonzero.std():.6f}")
        print(f"  Normal: {normal_nonzero.std():.6f}")
    
    # Feature usage differences
    reasoning_feature_freq = (reasoning_acts != 0).sum(axis=0) / reasoning_acts.shape[0]
    normal_feature_freq = (normal_acts != 0).sum(axis=0) / normal_acts.shape[0]
    
    feature_diff = reasoning_feature_freq - normal_feature_freq
    
    print("\nFeature usage differences:")
    print(f"  Reasoning-favored features (diff > 10%): {(feature_diff > 0.1).sum()}")
    print(f"  Normal-favored features (diff < -10%): {(feature_diff < -0.1).sum()}")
    print(f"  Near-neutral features (|diff| < 5%): {(np.abs(feature_diff) < 0.05).sum()}")
    
    # Identify most discriminative features
    top_reasoning_features = np.argsort(feature_diff)[-20:][::-1]
    top_normal_features = np.argsort(feature_diff)[:20]
    
    print("\nTop 20 reasoning-favored features:")
    print(f"{'Rank':<4} {'Feature':<8} {'Reasoning%':<12} {'Normal%':<12} {'Diff%':<10}")
    print("-" * 60)
    for i, feat_idx in enumerate(top_reasoning_features, 1):
        r_freq = reasoning_feature_freq[feat_idx]
        n_freq = normal_feature_freq[feat_idx]
        diff = feature_diff[feat_idx]
        print(f"{i:<4} {feat_idx:<8} {r_freq*100:>10.2f}% {n_freq*100:>10.2f}% {diff*100:>8.2f}%")
    
    print("\nTop 20 normal-favored features:")
    print(f"{'Rank':<4} {'Feature':<8} {'Reasoning%':<12} {'Normal%':<12} {'Diff%':<10}")
    print("-" * 60)
    for i, feat_idx in enumerate(top_normal_features, 1):
        r_freq = reasoning_feature_freq[feat_idx]
        n_freq = normal_feature_freq[feat_idx]
        diff = feature_diff[feat_idx]
        print(f"{i:<4} {feat_idx:<8} {r_freq*100:>10.2f}% {n_freq*100:>10.2f}% {diff*100:>8.2f}%")
    
    return {
        'top_reasoning_features': top_reasoning_features.tolist(),
        'top_normal_features': top_normal_features.tolist(),
        'feature_diff': feature_diff
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE activations (supports multi-token sequences).")
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--compare', action='store_true', help='Compare reasoning vs normal tokens.')
    parser.add_argument('--analyze-multi-token', action='store_true', help='Analyze multi-token patterns.')
    parser.add_argument('--export-features', type=str, default=None, 
                       help='Export discriminative features to a file.')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("SAE Activation Analysis")
    print("="*60)
    
    # Load config
    config_file = output_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("\nConfig:")
        print(f"  Model: {config.get('model_name', 'N/A')}")
        print(f"  Hook point: {config.get('hook_name', 'N/A')}")
        print(f"  SAE features: {config.get('d_sae', 'N/A')}")
        print(f"  Total tokens: {config.get('total_tokens', 0):,}")
        
        if 'total_reasoning_tokens' in config:
            print(f"  Reasoning tokens: {config['total_reasoning_tokens']:,}")
            print(f"  Normal tokens: {config['total_normal_tokens']:,}")
            print(f"  Reasoning ratio: {config['reasoning_ratio']*100:.2f}%")
            
            if 'single_token_matches' in config:
                print("\nMatch stats:")
                print(f"  Single-token matches: {config['single_token_matches']:,}")
                print(f"  Multi-token sequence matches: {config['multi_token_matches']:,}")
    
    # Analyze multi-token patterns
    if args.analyze_multi_token:
        analyze_multi_token_patterns(output_dir)
    
    # Analyze all activations
    try:
        all_acts, _ = load_activations(output_dir, 'all')
        analyze_activations(all_acts, "All activations")
    except FileNotFoundError:
        print("\nAll activation file not found")
    
    # Analyze reasoning activations
    reasoning_acts = None
    try:
        reasoning_acts, _ = load_activations(output_dir, 'reasoning')
        stats_reasoning = analyze_activations(reasoning_acts, "Reasoning activations")
    except FileNotFoundError:
        print("\nReasoning activation file not found")
    
    # Analyze normal activations
    normal_acts = None
    try:
        normal_acts, _ = load_activations(output_dir, 'normal')
        stats_normal = analyze_activations(normal_acts, "Normal activations")
    except FileNotFoundError:
        print("\nNormal activation file not found")
    
    # Comparison analysis
    comparison_results = None
    if args.compare and reasoning_acts is not None and normal_acts is not None:
        comparison_results = compare_activations(reasoning_acts, normal_acts)
    
    # Export discriminative features
    if args.export_features and comparison_results is not None:
        export_path = Path(args.export_features)
        export_data = {
            'top_reasoning_features': comparison_results['top_reasoning_features'],
            'top_normal_features': comparison_results['top_normal_features'],
            'feature_diff_stats': {
                'mean': float(comparison_results['feature_diff'].mean()),
                'std': float(comparison_results['feature_diff'].std()),
                'min': float(comparison_results['feature_diff'].min()),
                'max': float(comparison_results['feature_diff'].max()),
            }
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\n✓ Exported discriminative features to: {export_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils import load_layer_means, normalize_delta, load_all_means


def parse_layers(layer_str: str) -> List[int]:
    return [int(item.strip()) for item in layer_str.split(',') if item.strip()]


def apply_filters(delta_sae: np.ndarray,
                  reasoning_mean: np.ndarray,
                  relative: np.ndarray,
                  delta_threshold: float,
                  activation_percentile: float,
                  activation_scale: float,
                  relative_percentile: float) -> Dict[str, np.ndarray]:
    stage1_idx = np.where(delta_sae > delta_threshold)[0]

    activation_threshold = np.percentile(reasoning_mean, activation_percentile) * activation_scale
    if stage1_idx.size > 0:
        stage2_mask = reasoning_mean[stage1_idx] > activation_threshold
        stage2_idx = stage1_idx[stage2_mask]
        filtered_stage2 = stage1_idx[~stage2_mask]
    else:
        stage2_idx = stage1_idx
        filtered_stage2 = stage1_idx

    positive_relative = relative[relative > 0]
    if positive_relative.size == 0 or stage2_idx.size == 0:
        relative_threshold: Optional[float] = None
        final_idx = stage2_idx
        filtered_stage3 = np.array([], dtype=int)
    else:
        relative_threshold = float(np.percentile(positive_relative, relative_percentile))
        stage3_mask = relative[stage2_idx] > relative_threshold
        final_idx = stage2_idx[stage3_mask]
        filtered_stage3 = stage2_idx[~stage3_mask]

    return {
        'stage1_candidates': stage1_idx,
        'filtered_stage2': filtered_stage2,
        'stage2_passed': stage2_idx,
        'filtered_stage3': filtered_stage3,
        'final_selected': final_idx,
        'activation_threshold': float(activation_threshold),
        'relative_threshold': relative_threshold
    }


def plot_selected_feature_distributions(layer: int,
                                        delta_values: np.ndarray,
                                        relative_values: np.ndarray,
                                        output_dir: str):
    if delta_values.size == 0:
        print(f"  No selected features to plot for layer {layer}.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(delta_values, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_title(f'Layer {layer} Selected Delta Distribution')
    axes[0].set_xlabel('Delta SAE')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(relative_values, bins=50, color='darkorange', edgecolor='black', alpha=0.8)
    axes[1].set_title(f'Layer {layer} Selected Relative Distribution')
    axes[1].set_xlabel('Relative Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = osp.join(output_dir, f'layer_{layer}_selected_feature_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved selected feature distributions to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter SAE features using multi-stage criteria.")
    parser.add_argument('--base_dir', type=str,
                        default="/SAELens-main/my_training/sae_activations_output/deepseek-llama-8b-l{layer}-math500")
    parser.add_argument('--reasoning_name', type=str, default='deep_reasoning')
    parser.add_argument('--think_only_base', type=str, default=None,
                        help="Template path for resp_think_only directories with {layer}.")
    parser.add_argument('--resp_only_base', type=str, default=None,
                        help="Template path for resp_only directories with {layer}.")
    parser.add_argument('--layers', type=str, default="0,5,10,15,20,25,27",
                        help="Comma separated list of layers.")
    parser.add_argument('--delta_threshold', type=float, default=1.0,
                        help="Threshold applied to (normalized) delta_sae.")
    parser.add_argument('--no_normalize_delta', action='store_false', dest='normalize_delta',
                        help="Disable delta normalization before thresholding.")
    parser.set_defaults(normalize_delta=True)
    parser.add_argument('--activation_percentile', type=float, default=90.0)
    parser.add_argument('--activation_scale', type=float, default=0.1,
                        help="Scaling factor applied to the percentile statistic.")
    parser.add_argument('--relative_percentile', type=float, default=95.0)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--output_dir', type=str, default='./output_analysis')
    parser.add_argument('--output_name', type=str, default='selected_sae_features.json')
    args = parser.parse_args()

    if (args.think_only_base is None) ^ (args.resp_only_base is None):
        raise ValueError("Both --think_only_base and --resp_only_base must be provided together.")

    layers = parse_layers(args.layers)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        'config': {
            'base_dir': args.base_dir,
            'reasoning_name': args.reasoning_name,
            'think_only_base': args.think_only_base,
            'resp_only_base': args.resp_only_base,
            'layers': layers,
            'delta_threshold': args.delta_threshold,
            'activation_percentile': args.activation_percentile,
            'activation_scale': args.activation_scale,
            'relative_percentile': args.relative_percentile,
            'normalize_delta': args.normalize_delta,
            'eps': args.eps
        },
        'layers': {}
    }

    for layer in layers:
        print(f"\nProcessing layer {layer}...")
        if args.think_only_base and args.resp_only_base:
            think_dir = args.think_only_base.format(layer=layer)
            resp_dir = args.resp_only_base.format(layer=layer)
            think_stats = load_all_means(think_dir)
            resp_stats = load_all_means(resp_dir)
            reasoning_mean = think_stats['all_sae_mean']
            normal_mean = resp_stats['all_sae_mean']
        else:
            layer_dir = osp.join(args.base_dir.format(layer=layer), args.reasoning_name)
            stats = load_layer_means(layer_dir)
            reasoning_mean = stats['reasoning_sae_mean']
            normal_mean = stats['normal_sae_mean']
        raw_delta_sae = reasoning_mean - normal_mean
        delta_sae = normalize_delta(raw_delta_sae) if args.normalize_delta else raw_delta_sae

        relative = (reasoning_mean - normal_mean) / (normal_mean + args.eps)
        filters = apply_filters(delta_sae, reasoning_mean, relative,
                                args.delta_threshold,
                                args.activation_percentile,
                                args.activation_scale,
                                args.relative_percentile)

        results['layers'][str(layer)] = {
            'stage1_candidates': filters['stage1_candidates'].tolist(),
            'stage2_passed': filters['stage2_passed'].tolist(),
            'stage3_selected': filters['final_selected'].tolist(),
            'filtered_stage2': filters['filtered_stage2'].tolist(),
            'filtered_stage3': filters['filtered_stage3'].tolist(),
            'counts': {
                'stage1_total': int(len(filters['stage1_candidates'])),
                'stage2_filtered': int(len(filters['filtered_stage2'])),
                'stage2_remaining': int(len(filters['stage2_passed'])),
                'stage3_filtered': int(len(filters['filtered_stage3'])),
                'final_selected': int(len(filters['final_selected'])),
            },
            'activation_threshold': filters['activation_threshold'],
            'relative_threshold': filters['relative_threshold']
        }

        selected_idx = filters['final_selected']
        selected_delta = delta_sae[selected_idx] if selected_idx.size > 0 else np.array([])
        selected_relative = relative[selected_idx] if selected_idx.size > 0 else np.array([])
        plot_selected_feature_distributions(layer, selected_delta, selected_relative, args.output_dir)

        print(f"  Stage1 candidates: {len(filters['stage1_candidates'])}")
        print(f"  Filtered at Stage2: {len(filters['filtered_stage2'])}")
        print(f"  Remaining after Stage2: {len(filters['stage2_passed'])}")
        if filters['relative_threshold'] is None:
            print("  Stage3 skipped (no positive relative values).")
        else:
            print(f"  Relative threshold: {filters['relative_threshold']:.6f}")
        print(f"  Filtered at Stage3: {len(filters['filtered_stage3'])}")
        print(f"  Final selected features: {len(filters['final_selected'])}")

    output_path = osp.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSelection results saved to {output_path}")


if __name__ == "__main__":
    main()

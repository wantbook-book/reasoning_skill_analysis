import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List

import h5py
import numpy as np

PERCENTILES = [50, 75, 90, 95, 99, 99.9]


def parse_layer_features(layer_features: List[str]) -> Dict[int, List[int]]:
    mapping = defaultdict(set)
    for lf in layer_features:
        try:
            layer_str, feat_str = lf.split(':')
            layer = int(layer_str.strip())
            feat = int(feat_str.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid layer:feature format: {lf}") from exc
        mapping[layer].add(feat)
    return {layer: sorted(feats) for layer, feats in mapping.items()}


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    positives = values[values > 0]
    stats = {
        'count': int(values.shape[0]),
        'positive_count': int(positives.shape[0]),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
    }
    if positives.size > 0:
        stats['positive_mean'] = float(np.mean(positives))
        stats['positive_min'] = float(np.min(positives))
        stats['positive_max'] = float(np.max(positives))
        percentiles = np.percentile(positives, PERCENTILES)
        stats['percentiles'] = {str(p): float(v) for p, v in zip(PERCENTILES, percentiles)}
    else:
        stats['positive_mean'] = None
        stats['positive_min'] = None
        stats['positive_max'] = None
        stats['percentiles'] = {str(p): None for p in PERCENTILES}
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute reasoning SAE feature token-wise statistics.")
    parser.add_argument('--base_dir', type=str,
                        default="/SAELens-main/my_training/sae_activations_output/deepseek-llama-8b-l{layer}-math500")
    parser.add_argument('--reasoning_name', type=str, default='deep_reasoning')
    parser.add_argument('--resp_think_base_dir', type=str, default=None,
                        help="Optional template dir for resp_think_only runs with {layer}.")
    parser.add_argument('--layer_features', type=str, nargs='+', required=True,
                        help="Layer-feature pairs, e.g. '10:42'.")
    parser.add_argument('--output_dir', type=str, default='./output_analysis')
    parser.add_argument('--output_name', type=str, default='sae_feature_stats.json')
    args = parser.parse_args()

    layer2features = parse_layer_features(args.layer_features)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for layer, features in layer2features.items():
        print(f"\nProcessing layer {layer} features {features}")
        if args.resp_think_base_dir:
            dir_path = args.resp_think_base_dir.format(layer=layer)
            h5_path = osp.join(dir_path, 'sae_activations_normal_sampled.h5')
            dataset_key = 'activations'
            if not osp.exists(h5_path):
                raise FileNotFoundError(f"Normal sampled activations not found at {h5_path}")
            print(f"  Using resp_think_only sampled activations from {h5_path}")
        else:
            dir_path = osp.join(args.base_dir.format(layer=layer), args.reasoning_name)
            h5_path = osp.join(dir_path, 'sae_activations_reasoning.h5')
            dataset_key = 'activations'
            if not osp.exists(h5_path):
                raise FileNotFoundError(f"SAE reasoning activations not found at {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            dataset = f[dataset_key] if dataset_key in f else f['hidden_states']
            feature_stats = {}
            for feat in features:
                print(f"  Feature {feat}: collecting activations...")
                values = np.array(dataset[:, feat], dtype=np.float64)
                stats = compute_stats(values)
                feature_stats[str(feat)] = stats
                print(f"    Mean={stats['mean']:.4f}, P95={stats['percentiles']['95']:.4f}, "
                      f"P99={stats['percentiles']['99']:.4f}")

        results[str(layer)] = feature_stats

    output_path = osp.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        json.dump({
            'base_dir': args.base_dir,
            'reasoning_name': args.reasoning_name,
            'percentiles': PERCENTILES,
            'stats': results
        }, f, indent=2)

    print(f"\nFeature statistics saved to {output_path}")


if __name__ == "__main__":
    main()

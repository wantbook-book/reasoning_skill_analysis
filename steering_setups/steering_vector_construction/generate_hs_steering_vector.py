import argparse
import json
import os
import os.path as osp
from typing import List

import numpy as np

from utils import load_layer_means, load_all_means
import h5py


def parse_layers(layer_str: str) -> List[int]:
    return [int(item.strip()) for item in layer_str.split(',') if item.strip()]


def compute_steering_vector(reasoning_mean: np.ndarray,
                            normal_mean: np.ndarray,
                            normalize: bool) -> np.ndarray:
    vector = reasoning_mean - normal_mean
    if normalize:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    return vector


def compute_norm_stats(h5_path: str, batch_size: int = 4096):
    """Compute mean/std of L2 norms for hidden states stored in an H5 dataset."""
    if not osp.exists(h5_path):
        return None
    with h5py.File(h5_path, 'r') as f:
        dataset = f['activations'] if 'activations' in f else f['hidden_states']
        total = dataset.shape[0]
        if total == 0:
            return {'count': 0, 'mean': 0.0, 'std': 0.0}
        sum_norm = 0.0
        sum_sq_norm = 0.0
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = dataset[start:end].astype(np.float64, copy=False)
            norms = np.linalg.norm(batch, axis=1)
            sum_norm += norms.sum()
            sum_sq_norm += np.square(norms).sum()
        mean = sum_norm / total
        variance = max(sum_sq_norm / total - mean ** 2, 0.0)
        std = float(np.sqrt(variance))
        return {'count': int(total), 'mean': float(mean), 'std': std}


def main():
    parser = argparse.ArgumentParser(description="Generate hidden-state steering vectors.")
    parser.add_argument('--base_dir', type=str,
                        default="/SAELens-main/my_training/sae_activations_output/deepseek-llama-8b-l{layer}-math500",
                        help="Template path containing layer outputs with {layer} placeholder.")
    parser.add_argument('--reasoning_name', type=str, default='deep_reasoning')
    parser.add_argument('--think_only_base', type=str, default=None,
                        help="Template dir for resp_think_only outputs with {layer}.")
    parser.add_argument('--resp_only_base', type=str, default=None,
                        help="Template dir for resp_only outputs with {layer}.")
    parser.add_argument('--layers', type=str, default="0,5,10,15,20,25,27",
                        help="Comma-separated list of layers.")
    parser.add_argument('--normalize', action='store_true', default=False,
                        help="Normalize steering vectors to unit norm.")
    parser.add_argument('--project_reasoning', action='store_true', default=False,
                        help="Project reasoning hidden states onto the steering vector to compute stats.")
    parser.add_argument('--output_dir', type=str, default='./output_analysis')
    parser.add_argument('--output_name', type=str, default='hs_steering_vectors.json')
    args = parser.parse_args()

    if (args.think_only_base is None) ^ (args.resp_only_base is None):
        raise ValueError("Both --think_only_base and --resp_only_base must be provided together.")

    os.makedirs(args.output_dir, exist_ok=True)
    layers = parse_layers(args.layers)
    summary = {'config': vars(args), 'layers': {}}

    for layer in layers:
        print(f"\nProcessing layer {layer}...")
        if args.think_only_base and args.resp_only_base:
            think_dir = args.think_only_base.format(layer=layer)
            resp_dir = args.resp_only_base.format(layer=layer)
            think_stats = load_all_means(think_dir)
            resp_stats = load_all_means(resp_dir)
            reasoning_mean = think_stats['all_hs_mean']
            normal_mean = resp_stats['all_hs_mean']
            reasoning_h5_path = osp.join(think_dir, 'hidden_states_normal_sampled.h5')
            normal_h5_path = osp.join(resp_dir, 'hidden_states_normal_sampled.h5')
        else:
            dir_path = osp.join(args.base_dir.format(layer=layer), args.reasoning_name)
            stats = load_layer_means(dir_path)
            reasoning_mean = stats['reasoning_hs_mean']
            normal_mean = stats['normal_hs_mean']
            reasoning_h5_path = osp.join(dir_path, 'hidden_states_reasoning.h5')
            normal_h5_path = osp.join(dir_path, 'hidden_states_normal_sampled.h5')

        steering_vec = compute_steering_vector(reasoning_mean, normal_mean, args.normalize)
        save_path = osp.join(args.output_dir, f'layer_{layer}_hs_steering.npy')
        np.save(save_path, steering_vec.astype(np.float32))

        layer_summary = {
            'vector_path': save_path,
            'norm': float(np.linalg.norm(steering_vec)),
            'normalize': args.normalize
        }

        reasoning_norm_stats = compute_norm_stats(reasoning_h5_path)
        if reasoning_norm_stats:
            layer_summary['reasoning_hs_norm_mean'] = reasoning_norm_stats['mean']
            layer_summary['reasoning_hs_norm_std'] = reasoning_norm_stats['std']
            layer_summary['reasoning_hs_count'] = reasoning_norm_stats['count']
            print(f"  Reasoning HS norm -> mean: {reasoning_norm_stats['mean']:.6f}, "
                  f"std: {reasoning_norm_stats['std']:.6f}, "
                  f"count: {reasoning_norm_stats['count']:,}")
        else:
            print(f"  Warning: reasoning hidden states not found at {reasoning_h5_path}")

        normal_norm_stats = compute_norm_stats(normal_h5_path)
        if normal_norm_stats:
            layer_summary['normal_hs_norm_mean'] = normal_norm_stats['mean']
            layer_summary['normal_hs_norm_std'] = normal_norm_stats['std']
            layer_summary['normal_hs_count'] = normal_norm_stats['count']
            print(f"  Normal HS norm -> mean: {normal_norm_stats['mean']:.6f}, "
                  f"std: {normal_norm_stats['std']:.6f}, "
                  f"count: {normal_norm_stats['count']:,}")
        else:
            print(f"  Warning: normal hidden states not found at {normal_h5_path}")

        if args.project_reasoning:
            if not osp.exists(reasoning_h5_path):
                raise FileNotFoundError(f"Reasoning hidden states not found: {reasoning_h5_path}")
            with h5py.File(reasoning_h5_path, 'r') as f:
                dataset = f['activations'] if 'activations' in f else f['hidden_states']
                projections = dataset @ steering_vec
                layer_summary['projection_count'] = int(dataset.shape[0])
                layer_summary['projection_mean'] = float(np.mean(projections))
                layer_summary['projection_std'] = float(np.std(projections))
                print(f"  Projection stats -> mean: {layer_summary['projection_mean']:.6f}, std: {layer_summary['projection_std']:.6f}")
            
            if not osp.exists(normal_h5_path):
                raise FileNotFoundError(f"Normal hidden states not found: {normal_h5_path}")
            with h5py.File(normal_h5_path, 'r') as f:
                dataset = f['activations'] if 'activations' in f else f['hidden_states']
                projections = dataset @ steering_vec
                layer_summary['normal_projection_count'] = int(dataset.shape[0])
                layer_summary['normal_projection_mean'] = float(np.mean(projections))
                layer_summary['normal_projection_std'] = float(np.std(projections))
                print(f"  Normal projection stats -> mean: {layer_summary['normal_projection_mean']:.6f}, std: {layer_summary['normal_projection_std']:.6f}")

        summary['layers'][str(layer)] = layer_summary
        print(f"  Saved steering vector to {save_path}")

    summary_path = osp.join(args.output_dir, args.output_name)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSteering vector summary saved to {summary_path}")


if __name__ == "__main__":
    main()

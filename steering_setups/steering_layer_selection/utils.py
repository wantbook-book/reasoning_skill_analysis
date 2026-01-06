import os.path as osp
import numpy as np
import h5py
import json
from typing import Dict, List, Tuple

import torch
from sae_lens import SAE


def _resolve_existing_file(dir_path: str, filename_options):
    """Return the first matched file path from filename_options."""
    for name in filename_options:
        candidate = osp.join(dir_path, name)
        if osp.exists(candidate):
            return candidate
    raise FileNotFoundError(f"None of {filename_options} found in {dir_path}")


def normalize_delta(delta, eps: float = 1e-12):
    """
    Normalize delta values by their standard deviation.
    """
    std = np.std(delta)
    if std < eps:
        return np.zeros_like(delta)
    return delta / std


def cal_mean_of_h5_data(h5_file: str, chunk_size: int = 1000):
    """
    Compute the mean vector of an h5 dataset named 'activations'.
    """
    with h5py.File(h5_file, "r") as f:
        dataset = f['activations'] if 'activations' in f else f['hidden_states']
        n_samples = dataset.shape[0]
        feature_dim = dataset.shape[1]

        mean = np.zeros(feature_dim, dtype=np.float64)
        n = 0

        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = dataset[i:end_idx].astype(np.float64)

            for sample in chunk:
                n += 1
                delta = sample - mean
                mean += delta / n

        return mean.astype(np.float32)


def load_layer_means(dir_path: str):
    """
    Load normal/reasoning SAE and HS mean activations for a layer directory.
    """
    normal_hs_path = osp.join(dir_path, 'hidden_states_normal_mean.npy')
    reasoning_hs_path = osp.join(dir_path, 'hidden_states_reasoning.h5')
    normal_sae_path = osp.join(dir_path, 'sae_activations_all_mean.npy')
    reasoning_sae_path = osp.join(dir_path, 'sae_activations_reasoning.h5')

    stats = {
        'normal_hs_mean': np.load(normal_hs_path),
        'normal_sae_mean': np.load(normal_sae_path),
        'reasoning_hs_mean': cal_mean_of_h5_data(reasoning_hs_path),
        'reasoning_sae_mean': cal_mean_of_h5_data(reasoning_sae_path),
    }
    return stats


def load_all_means(dir_path: str):
    """
    Load SAE/HS mean activations that were computed over all tokens.
    """
    hs_path = osp.join(dir_path, 'hidden_states_all_mean.npy')
    sae_path = _resolve_existing_file(dir_path, ['sae_activations_all_mean.npy', 'sae_activation_all_mean.npy'])
    stats = {
        'all_hs_mean': np.load(hs_path),
        'all_sae_mean': np.load(sae_path),
    }
    return stats


def load_selected_features(result_path: str) -> Dict[int, Dict[str, List[int]]]:
    """
    Load select_sae_features results JSON and return layer-wise selections.
    """
    with open(result_path, 'r') as f:
        data = json.load(f)
    layers = {}
    for layer, info in data.get('layers', {}).items():
        layers[int(layer)] = info
    return layers


def load_sae_decoder(sae_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SAE decoder weights/bias using SAE load_from_pretrained.
    """
    sae = SAE.load_from_pretrained(sae_path)
    weight = sae.W_dec.detach().cpu().numpy()
    bias = sae.b_dec.detach().cpu().numpy() if hasattr(sae, "b_dec") else np.zeros(weight.shape[1], dtype=np.float32)
    return weight, bias

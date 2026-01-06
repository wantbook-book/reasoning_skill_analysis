import argparse
import json
import os
import os.path as osp
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

from utils import load_layer_means, load_selected_features, load_sae_decoder, normalize_delta, load_all_means

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def parse_layers(layer_str: str) -> List[int]:
    return [int(item.strip()) for item in layer_str.split(',') if item.strip()]


def decode_features(weight: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    return weight[feature_indices]


def cluster_features(vectors: np.ndarray, n_clusters: int, random_state: int = 42):
    vectors_norm = normalize(vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(vectors_norm)
    return kmeans, labels, vectors_norm


def plot_cluster_projection(vectors_norm: np.ndarray, labels: np.ndarray, layer: int, output_dir: str,
                            method: str, style_cfg, plot_cfg, save_pdf: bool):
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == "tsne":
        available = max(len(vectors_norm) - 1, 1)
        perplexity = min(max(plot_cfg['tsne_perplexity'], 1.0), available)
        reducer = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
    elif method == "umap":
        if not HAS_UMAP:
            print("  UMAP not installed; skipping projection.")
            return
        n_neighbors = min(max(plot_cfg['umap_n_neighbors'], 2), max(len(vectors_norm) - 1, 2))
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=plot_cfg['umap_min_dist'],
            metric=plot_cfg['umap_metric'],
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")

    reduced = reducer.fit_transform(vectors_norm)
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = style_cfg['font_family']
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap=plot_cfg['cluster_cmap'],
        alpha=0.8,
        s=plot_cfg['cluster_point_size']
    )
    ax.set_xlabel('Dim 1', fontsize=style_cfg['label_fontsize'])
    ax.set_ylabel('Dim 2', fontsize=style_cfg['label_fontsize'])
    ax.tick_params(axis='both', labelsize=style_cfg['tick_fontsize'])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster', fontsize=plot_cfg['cluster_cbar_labelsize'])
    cbar.ax.tick_params(labelsize=plot_cfg['cluster_cbar_ticksize'])
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    save_path = osp.join(output_dir, f'layer_{layer}_sae_clusters_{method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Cluster plot ({method}) saved to {save_path}")


def plot_similarity_heatmap(vectors_norm: np.ndarray, labels: np.ndarray, layer: int, output_dir: str, style_cfg,
                            plot_cfg, save_pdf: bool):
    similarity = vectors_norm @ vectors_norm.T
    sort_idx = np.argsort(labels)
    sorted_similarity = similarity[sort_idx][:, sort_idx]
    sorted_labels = labels[sort_idx]

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = style_cfg['font_family']
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sorted_similarity, cmap=plot_cfg['heatmap_cmap'], aspect='auto',
                   vmin=plot_cfg['similarity_vmin'], vmax=plot_cfg['similarity_vmax'])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=plot_cfg['heatmap_cbar_labelsize'])
    cbar.ax.tick_params(labelsize=plot_cfg['heatmap_cbar_ticksize'])
    ax.set_xlabel('Feature Index (sorted)', fontsize=style_cfg['label_fontsize'])
    ax.set_ylabel('Feature Index (sorted)', fontsize=style_cfg['label_fontsize'])
    ax.tick_params(axis='both', labelsize=style_cfg['tick_fontsize'])

    boundaries = []
    for i in range(1, len(sorted_labels)):
        if sorted_labels[i] != sorted_labels[i-1]:
            boundaries.append(i - 0.5)
    for boundary in boundaries:
        ax.axhline(boundary, color='white', linestyle='--', linewidth=0.5)
        ax.axvline(boundary, color='white', linestyle='--', linewidth=0.5)
    sns.despine(ax=ax)

    save_path = osp.join(output_dir, f'layer_{layer}_similarity_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Similarity heatmap saved to {save_path}")


def select_top_features_per_cluster(delta_values: np.ndarray, labels: np.ndarray, feature_indices: np.ndarray):
    final_features = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_indices = feature_indices[cluster_mask]
        cluster_deltas = delta_values[cluster_mask]
        if cluster_indices.size == 0:
            continue
        top_idx = cluster_indices[np.argmax(cluster_deltas)]
        final_features.append(int(top_idx))
    return final_features


def main():
    parser = argparse.ArgumentParser(description="Cluster selected SAE features and pick representatives.")
    parser.add_argument('--selection_path', type=str,
                        default='./output_analysis/selected_sae_features.json')
    parser.add_argument('--base_dir', type=str,
                        default="/SAELens-main/my_training/sae_activations_output/deepseek-llama-8b-l{layer}-math500")
    parser.add_argument('--reasoning_name', type=str, default='deep_reasoning')
    parser.add_argument('--think_only_base', type=str, default=None,
                        help="Template dir for resp_think_only outputs with {layer}.")
    parser.add_argument('--resp_only_base', type=str, default=None,
                        help="Template dir for resp_only outputs with {layer}.")
    parser.add_argument('--layers', type=str, default="0,5,10,15,20,25,27")
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--normalize_delta', action='store_true', default=False)
    parser.add_argument('--sae_path_template', type=str,
                        default="/SAELens-main/my_training/sae_checkpoints/deepseek-llama-8b-l{layer}",
                        help="Template path to SAE checkpoints with {layer} placeholder.")
    parser.add_argument('--output_dir', type=str, default='./output_analysis')
    parser.add_argument('--output_name', type=str, default='clustered_sae_features.json')
    parser.add_argument('--font_family', type=str, default='Times New Roman')
    parser.add_argument('--label_fontsize', type=int, default=12)
    parser.add_argument('--tick_fontsize', type=int, default=11)
    parser.add_argument('--cluster_cmap', type=str, default='tab10')
    parser.add_argument('--cluster_point_size', type=float, default=30.0)
    parser.add_argument('--heatmap_cmap', type=str, default='viridis')
    parser.add_argument('--similarity_vmin', type=float, default=None)
    parser.add_argument('--similarity_vmax', type=float, default=None)
    parser.add_argument('--cluster_cbar_labelsize', type=int, default=12)
    parser.add_argument('--cluster_cbar_ticksize', type=int, default=10)
    parser.add_argument('--heatmap_cbar_labelsize', type=int, default=12)
    parser.add_argument('--heatmap_cbar_ticksize', type=int, default=10)
    parser.add_argument('--tsne_perplexity', type=float, default=10.0)
    parser.add_argument('--umap_n_neighbors', type=int, default=10)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)
    parser.add_argument('--umap_metric', type=str, default='euclidean')
    parser.add_argument('--save_pdf', action='store_true', help='Export plots as PDF in addition to PNG')
    args = parser.parse_args()

    if (args.think_only_base is None) ^ (args.resp_only_base is None):
        raise ValueError("Both --think_only_base and --resp_only_base must be provided together.")

    os.makedirs(args.output_dir, exist_ok=True)
    selected = load_selected_features(args.selection_path)
    layers = parse_layers(args.layers)

    style_cfg = {
        'font_family': args.font_family,
        'label_fontsize': args.label_fontsize,
        'tick_fontsize': args.tick_fontsize,
    }

    clustered_results: Dict[int, Dict[str, List[int]]] = {}
    final_selection = {}

    for layer in layers:
        print(f"\nProcessing layer {layer}...")
        layer_result = selected.get(layer)
        if not layer_result:
            print("  No selected features for this layer, skipping.")
            continue

        feature_indices = np.array(layer_result['stage3_selected'], dtype=int)
        if feature_indices.size == 0:
            print("  No features passed Stage 3, skipping.")
            continue

        if args.think_only_base and args.resp_only_base:
            think_dir = args.think_only_base.format(layer=layer)
            resp_dir = args.resp_only_base.format(layer=layer)
            think_stats = load_all_means(think_dir)
            resp_stats = load_all_means(resp_dir)
            raw_delta = think_stats['all_sae_mean'] - resp_stats['all_sae_mean']
        else:
            layer_dir = osp.join(args.base_dir.format(layer=layer), args.reasoning_name)
            stats = load_layer_means(layer_dir)
            raw_delta = stats['reasoning_sae_mean'] - stats['normal_sae_mean']
        delta_values = normalize_delta(raw_delta) if args.normalize_delta else raw_delta

        sae_path = args.sae_path_template.format(layer=layer)
        weight, _ = load_sae_decoder(sae_path)
        decoded_vectors = decode_features(weight, feature_indices)
        kmeans, labels, vectors_norm = cluster_features(decoded_vectors, args.n_clusters)
        for method in ["pca", "tsne", "umap"]:
            plot_cluster_projection(
                vectors_norm,
                labels,
                layer,
                args.output_dir,
                method,
                style_cfg,
                {
                    'cluster_cmap': args.cluster_cmap,
                    'cluster_point_size': args.cluster_point_size,
                    'cluster_cbar_labelsize': args.cluster_cbar_labelsize,
                    'cluster_cbar_ticksize': args.cluster_cbar_ticksize,
                    'tsne_perplexity': args.tsne_perplexity,
                    'umap_n_neighbors': args.umap_n_neighbors,
                    'umap_min_dist': args.umap_min_dist,
                    'umap_metric': args.umap_metric,
                },
                args.save_pdf
            )
        plot_similarity_heatmap(
            vectors_norm,
            labels,
            layer,
            args.output_dir,
            style_cfg,
            {
                'heatmap_cmap': args.heatmap_cmap,
                'similarity_vmin': args.similarity_vmin,
                'similarity_vmax': args.similarity_vmax,
                'heatmap_cbar_labelsize': args.heatmap_cbar_labelsize,
                'heatmap_cbar_ticksize': args.heatmap_cbar_ticksize
            },
            args.save_pdf
        )

        top_features = select_top_features_per_cluster(delta_values[feature_indices], labels, feature_indices)

        clustered_results[layer] = {
            'feature_indices': feature_indices.tolist(),
            'cluster_labels': labels.tolist(),
            'top_features': top_features
        }
        final_selection[layer] = top_features

        print(f"  Selected {len(top_features)} representative features from {feature_indices.size} candidates.")

    output_path = osp.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        json.dump({'clusters': clustered_results, 'final_selection': final_selection}, f, indent=2)

    print(f"\nClustered feature summary saved to {output_path}")


if __name__ == "__main__":
    main()

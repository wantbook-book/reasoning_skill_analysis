import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils import normalize_delta, load_layer_means, load_all_means

def plot_distribution(layer_delta_sae, layer_delta_hs, layers, save_dir, save_pdf=False, reasoning_name=""):
    """
    Plot distributions of delta_sae and delta_hs vectors for each layer.
    """
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(15, 4*n_layers))
    if reasoning_name:
        fig.suptitle(f"{reasoning_name.replace('_', ' ').title()} - Delta Distributions", fontsize=16, fontweight='bold')
    
    for idx, layer in enumerate(layers):
        sae_vals = layer_delta_sae[idx]
        sae_non_zero = sae_vals[sae_vals != 0]
        hs_vals = layer_delta_hs[idx]
        hs_non_zero = hs_vals[hs_vals != 0]
        
        # Plot delta_sae distribution
        ax_sae = axes[idx, 0] if n_layers > 1 else axes[0]
        if sae_non_zero.size > 0:
            ax_sae.hist(sae_non_zero, bins=100, alpha=0.7, edgecolor='black')
        else:
            ax_sae.text(0.5, 0.5, 'No non-zero values', ha='center', va='center',
                        transform=ax_sae.transAxes)
        ax_sae.set_title(f'Layer {layer} - SAE', fontsize=12)
        ax_sae.set_xlabel('Delta SAE Value')
        ax_sae.set_ylabel('Frequency')
        ax_sae.grid(True, alpha=0.3)
        
        # Plot delta_hs distribution
        ax_hs = axes[idx, 1] if n_layers > 1 else axes[1]
        if hs_non_zero.size > 0:
            ax_hs.hist(hs_non_zero, bins=100, alpha=0.7, color='orange', edgecolor='black')
        else:
            ax_hs.text(0.5, 0.5, 'No non-zero values', ha='center', va='center',
                       transform=ax_hs.transAxes)
        ax_hs.set_title(f'Layer {layer} - Hidden State', fontsize=12)
        ax_hs.set_xlabel('Delta HS Value')
        ax_hs.set_ylabel('Frequency')
        ax_hs.grid(True, alpha=0.3)
    
    plt.tight_layout()
    png_path = osp.join(save_dir, 'delta_distributions.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def plot_active_neurons_by_threshold(active_data, layers, save_dir, style_cfg, save_pdf=False):
    """
    Plot active neuron counts across layers for different reasoning configs (subplots).
    """

    def positive_percentile_threshold(layer_deltas, percentile=99):
        positives = []
        for delta in layer_deltas:
            positive_delta = delta[delta > 0]
            if positive_delta.size > 0:
                positives.append(positive_delta)
        if not positives:
            return None
        positives = np.concatenate(positives)
        return np.percentile(positives, percentile)

    reasoning_names = list(active_data.keys())
    n_cols = len(reasoning_names)
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = style_cfg.get('font_family', 'Times New Roman')
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6.3), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, reasoning_name in zip(axes, reasoning_names):
        layer_delta_sae = active_data[reasoning_name]['sae']
        layer_delta_hs = active_data[reasoning_name]['hs']
        sae_threshold = positive_percentile_threshold(layer_delta_sae)
        hs_threshold = positive_percentile_threshold(layer_delta_hs)

        if sae_threshold is not None:
            active_counts_sae = [np.sum(delta_sae > sae_threshold) for delta_sae in layer_delta_sae]
            ax.plot(
                layers,
                active_counts_sae,
                marker=style_cfg['sae_style'].get('marker', 'o'),
                linewidth=style_cfg['sae_style'].get('linewidth', 2),
                linestyle=style_cfg['sae_style'].get('linestyle', '-'),
                color=style_cfg['sae_style'].get('color', '#1f77b4'),
                markersize=style_cfg['sae_style'].get('markersize', 4),
                label=f"{style_cfg['sae_style'].get('label', 'SAE')} (p99={sae_threshold:.1f})",
            )
        else:
            ax.text(0.5, 0.6, 'No positive SAE deltas', ha='center', va='center',
                    transform=ax.transAxes, fontsize=style_cfg['label_fontsize'])

        if hs_threshold is not None:
            active_counts_hs = [np.sum(delta_hs > hs_threshold) for delta_hs in layer_delta_hs]
            ax.plot(
                layers,
                active_counts_hs,
                marker=style_cfg['hs_style'].get('marker', 's'),
                linewidth=style_cfg['hs_style'].get('linewidth', 2),
                linestyle=style_cfg['hs_style'].get('linestyle', '--'),
                color=style_cfg['hs_style'].get('color', '#ff7f0e'),
                markersize=style_cfg['hs_style'].get('markersize', 4),
                label=f"{style_cfg['hs_style'].get('label', 'HS')} (p99={hs_threshold:.1f})",
            )
        else:
            ax.text(0.5, 0.4, 'No positive HS deltas', ha='center', va='center',
                    transform=ax.transAxes, fontsize=style_cfg['label_fontsize'])

        ax.set_xlabel('Layer', fontsize=style_cfg['label_fontsize'])
        # ax.set_title(reasoning_name.replace('_', ' ').title(), fontsize=style_cfg['title_fontsize'], fontweight='bold')
        ax.tick_params(axis='both', labelsize=style_cfg['tick_fontsize'])
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        if sae_threshold is not None or hs_threshold is not None:
            ax.legend(fontsize=style_cfg['legend_fontsize'])
        sns.despine(ax=ax)

    axes[0].set_ylabel('Number of Active Neurons', fontsize=style_cfg['label_fontsize'])

    plt.tight_layout()
    png_path = osp.join(save_dir, 'active_neurons_by_layer.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def plot_heatmap(layer_delta_sae, layer_delta_hs, layers, save_dir, save_pdf=False, reasoning_name=""):
    """
    Plot heatmaps to show per-layer activations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    if reasoning_name:
        fig.suptitle(f"{reasoning_name.replace('_', ' ').title()} - Delta Heatmaps", fontsize=16, fontweight='bold')
    
    # Visualize the first 1000 features (if there are too many)
    max_features = 1000
    
    # SAE heatmap
    sae_data = np.array([delta[:max_features] for delta in layer_delta_sae])
    im1 = ax1.imshow(sae_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers)
    ax1.set_xlabel('Feature Index', fontsize=12)
    ax1.set_ylabel('Layer', fontsize=12)
    ax1.set_title(f'Layer Delta SAE - {reasoning_name}', fontsize=12)
    plt.colorbar(im1, ax=ax1)
    
    # Hidden States heatmap
    hs_data = np.array([delta[:max_features] for delta in layer_delta_hs])
    im2 = ax2.imshow(hs_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels(layers)
    ax2.set_xlabel('Feature Index', fontsize=12)
    ax2.set_ylabel('Layer', fontsize=12)
    ax2.set_title(f'Layer Delta Hidden States - {reasoning_name}', fontsize=12)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    png_path = osp.join(save_dir, 'delta_heatmap.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, 
                       default="/SAELens-main/my_training/sae_activations_output/deepseek-llama-8b-l{layer}-math500")
    parser.add_argument('--output_dir', type=str, default='./output_analysis')
    parser.add_argument('--reasoning_names', type=str, nargs='*',
                        default=['deep_reasoning', 'extensive_exploration', 'sensible_reflection'],
                        help='Reasoning subdirectories to process; pass empty to skip.')
    parser.add_argument('--think_only_base', type=str, default=None,
                        help='Template path for resp_think_only directories with {layer} placeholder.')
    parser.add_argument('--resp_only_base', type=str, default=None,
                        help='Template path for resp_only directories with {layer} placeholder.')
    parser.add_argument('--comparison_label', type=str, default='think_minus_resp',
                        help='Label for think-only minus resp-only comparison plots.')
    parser.add_argument('--font_family', type=str, default='Times New Roman')
    parser.add_argument('--label_fontsize', type=int, default=12)
    parser.add_argument('--title_fontsize', type=int, default=14)
    parser.add_argument('--legend_fontsize', type=int, default=11)
    parser.add_argument('--tick_fontsize', type=int, default=11)
    parser.add_argument('--sae_color', type=str, default='#1f77b4')
    parser.add_argument('--sae_linestyle', type=str, default='-')
    parser.add_argument('--sae_marker', type=str, default='o')
    parser.add_argument('--sae_markersize', type=float, default=4.0)
    parser.add_argument('--sae_linewidth', type=float, default=2.0)
    parser.add_argument('--hs_color', type=str, default='#ff7f0e')
    parser.add_argument('--hs_linestyle', type=str, default='--')
    parser.add_argument('--hs_marker', type=str, default='s')
    parser.add_argument('--hs_markersize', type=float, default=4.0)
    parser.add_argument('--hs_linewidth', type=float, default=2.0)
    parser.add_argument('--save_pdf', action='store_true', help='Export plots as PDF as well as PNG')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 5, 10, 15, 20, 25, 31],
                        help='List of layers to process')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    layers = args.layers
    reasoning_names = args.reasoning_names
    reasoning_data = {}
    
    os.makedirs(args.output_dir, exist_ok=True)

    if reasoning_names:
        for reasoning_name in reasoning_names:
            print(f"\nProcessing reasoning mode: {reasoning_name}")
            layer_delta_sae = []
            layer_delta_hs = []
            for layer in layers:
                print(f"  Processing layer {layer}...")
                dir_path = osp.join(base_dir.format(layer=layer), reasoning_name)
                stats = load_layer_means(dir_path)
                normal_hs_mean = stats['normal_hs_mean']
                reasoning_hs_mean = stats['reasoning_hs_mean']
                normal_sae_mean = stats['normal_sae_mean']
                reasoning_sae_mean = stats['reasoning_sae_mean']
                
                delta_sae = reasoning_sae_mean - normal_sae_mean
                delta_hs = reasoning_hs_mean - normal_hs_mean
                
                layer_delta_sae.append(normalize_delta(delta_sae))
                layer_delta_hs.append(normalize_delta(delta_hs))
                
                print(f"    SAE shape: {delta_sae.shape}, HS shape: {delta_hs.shape}")

            reasoning_data[reasoning_name] = {
                'sae': layer_delta_sae,
                'hs': layer_delta_hs
            }

            reason_output_dir = osp.join(args.output_dir, reasoning_name)
            os.makedirs(reason_output_dir, exist_ok=True)

            print("  Plotting distributions...")
            plot_distribution(layer_delta_sae, layer_delta_hs, layers, reason_output_dir, args.save_pdf, reasoning_name)

            print("  Plotting heatmaps...")
            plot_heatmap(layer_delta_sae, layer_delta_hs, layers, reason_output_dir, args.save_pdf, reasoning_name)
    
    if args.think_only_base and args.resp_only_base:
        diff_name = args.comparison_label
        print(f"\nProcessing comparison mode: {diff_name}")
        layer_delta_sae = []
        layer_delta_hs = []
        for layer in layers:
            print(f"  Processing layer {layer} for comparison...")
            think_dir = args.think_only_base.format(layer=layer)
            resp_dir = args.resp_only_base.format(layer=layer)
            think_stats = load_all_means(think_dir)
            resp_stats = load_all_means(resp_dir)

            delta_sae = think_stats['all_sae_mean'] - resp_stats['all_sae_mean']
            delta_hs = think_stats['all_hs_mean'] - resp_stats['all_hs_mean']

            layer_delta_sae.append(normalize_delta(delta_sae))
            layer_delta_hs.append(normalize_delta(delta_hs))

            print(f"    SAE shape: {delta_sae.shape}, HS shape: {delta_hs.shape}")

        reasoning_data[diff_name] = {
            'sae': layer_delta_sae,
            'hs': layer_delta_hs
        }

        comp_output_dir = osp.join(args.output_dir, diff_name)
        os.makedirs(comp_output_dir, exist_ok=True)

        print("  Plotting comparison distributions...")
        plot_distribution(layer_delta_sae, layer_delta_hs, layers, comp_output_dir, args.save_pdf, diff_name)

        print("  Plotting comparison heatmaps...")
        plot_heatmap(layer_delta_sae, layer_delta_hs, layers, comp_output_dir, args.save_pdf, diff_name)

    if not reasoning_data:
        print("No reasoning data available for plotting.")
        return

    print("\nPlotting combined active neuron curves...")
    style_cfg = {
        'font_family': args.font_family,
        'label_fontsize': args.label_fontsize,
        'title_fontsize': args.title_fontsize,
        'legend_fontsize': args.legend_fontsize,
        'tick_fontsize': args.tick_fontsize,
        'sae_style': {
            'color': args.sae_color,
            'linestyle': args.sae_linestyle,
            'marker': args.sae_marker,
            'markersize': args.sae_markersize,
            'linewidth': args.sae_linewidth,
            'label': 'SAE'
        },
        'hs_style': {
            'color': args.hs_color,
            'linestyle': args.hs_linestyle,
            'marker': args.hs_marker,
            'markersize': args.hs_markersize,
            'linewidth': args.hs_linewidth,
            'label': 'HS'
        }
    }
    plot_active_neurons_by_threshold(reasoning_data, layers, args.output_dir, style_cfg, args.save_pdf)
    
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()

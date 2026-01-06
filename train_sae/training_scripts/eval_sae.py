"""
Evaluate a locally trained SAE model.
Used to evaluate SAEs trained on local pretokenized datasets.
"""
import json
import argparse
from pathlib import Path
import torch
from sae_lens.saes.sae import SAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.evals import run_evals, get_eval_everything_config
from sae_lens.load_model import load_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a locally trained SAE model.")
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to the local SAE model (directory with cfg.json and sae_weights.safetensors)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset path for evaluation (local pretokenized JSONL or HF dataset name)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for evaluation results."
    )
    parser.add_argument(
        "--batch_size_prompts",
        type=int,
        default=16,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--n_eval_reconstruction_batches",
        type=int,
        default=20,
        help="Batches for reconstruction metrics (KL, cross-entropy, etc.)."
    )
    parser.add_argument(
        "--n_eval_sparsity_variance_batches",
        type=int,
        default=200,
        help="Batches for sparsity and variance metrics."
    )
    parser.add_argument(
        "--dataset_trust_remote_code",
        action="store_true",
        help="Trust remote code for datasets (may be required for HF datasets)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu); default auto-select."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed evaluation progress."
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # 1. Load local SAE
    print(f"\n{'='*60}")
    print(f"Loading SAE from {args.sae_path}...")
    print(f"{'='*60}")
    sae = SAE.load_from_pretrained(args.sae_path, device=device)
    print("✓ SAE loaded successfully")
    print(f"  - Input dim (d_in): {sae.cfg.d_in}")
    print(f"  - SAE dim (d_sae): {sae.cfg.d_sae}")
    print(f"  - Architecture: {sae.cfg.architecture}")

    # 2. Get model info from SAE config
    model_name = sae.cfg.metadata.model_name
    hook_name = sae.cfg.metadata.hook_name
    model_class_name = sae.cfg.metadata.model_class_name
    context_size = sae.cfg.metadata.context_size

    print(f"\n{'='*60}")
    print("SAE model config:")
    print(f"{'='*60}")
    print(f"  - LLM model: {model_name}")
    print(f"  - Hook point: {hook_name}")
    print(f"  - Model class: {model_class_name}")
    print(f"  - Context length: {context_size}")

    # 3. Load LLM (to generate activations from tokenized text)
    print(f"\n{'='*60}")
    print(f"Loading LLM model: {model_name}")
    print(f"{'='*60}")
    model = load_model(
        model_class_name=model_class_name,
        model_name=model_name,
        device=device,
        model_from_pretrained_kwargs=sae.cfg.metadata.model_from_pretrained_kwargs,
    )
    print("✓ LLM model loaded successfully")

    # 4. Create ActivationsStore
    # This reads tokens from the pretokenized dataset and generates activations via the LLM.
    print(f"\n{'='*60}")
    print("Creating ActivationsStore...")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Context length: {context_size}")
    print(f"{'='*60}")
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        context_size=context_size,
        dataset="json",
        data_files=[args.dataset],
        dataset_trust_remote_code=args.dataset_trust_remote_code,
        streaming=False,  # Local datasets usually don't need streaming
    )
    print("✓ ActivationsStore created successfully")

    # 5. Configure evaluation metrics (evaluate all available metrics)
    print(f"\n{'='*60}")
    print("Evaluation config:")
    print(f"{'='*60}")
    print(f"  - Batch size: {args.batch_size_prompts}")
    print(f"  - Reconstruction batches: {args.n_eval_reconstruction_batches}")
    print(f"  - Sparsity/variance batches: {args.n_eval_sparsity_variance_batches}")

    eval_config = get_eval_everything_config(
        batch_size_prompts=args.batch_size_prompts,
        n_eval_reconstruction_batches=args.n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=args.n_eval_sparsity_variance_batches,
    )

    # 6. Run evaluation
    print(f"\n{'='*60}")
    print("Running evaluation...")
    print(f"{'='*60}")
    scalar_metrics, feature_metrics = run_evals(
        sae=sae,
        activation_store=activation_store,
        model=model,
        activation_scaler=ActivationScaler(),  # No scaling by default
        eval_config=eval_config,
        verbose=args.verbose,
    )

    # 7. Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "sae_path": args.sae_path,
        "model_name": model_name,
        "hook_name": hook_name,
        "model_class_name": model_class_name,
        "context_size": context_size,
        "dataset": args.dataset,
        "eval_config": {
            "batch_size_prompts": args.batch_size_prompts,
            "n_eval_reconstruction_batches": args.n_eval_reconstruction_batches,
            "n_eval_sparsity_variance_batches": args.n_eval_sparsity_variance_batches,
        },
        "scalar_metrics": scalar_metrics,
        "feature_metrics_summary": {
            "n_features": len(feature_metrics.get("feature_density", [])),
            "avg_feature_density": sum(feature_metrics.get("feature_density", [])) / len(feature_metrics.get("feature_density", [1])),
        } if feature_metrics else {},
    }

    # Save detailed results (feature-wise metrics)
    output_file = output_path / "eval_results_detailed.json"
    full_results = results.copy()
    full_results["feature_metrics"] = feature_metrics
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    # Save summary results (scalar metrics only)
    output_file_summary = output_path / "eval_results_summary.json"
    with open(output_file_summary, "w") as f:
        json.dump(results, f, indent=2)

    # 8. Print results
    print(f"\n{'='*60}")
    print("Evaluation results:")
    print(f"{'='*60}")
    for category, metrics in scalar_metrics.items():
        print(f"\n【{category}】")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    print(f"✓ Detailed results saved to: {output_file}")
    print(f"✓ Summary results saved to: {output_file_summary}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

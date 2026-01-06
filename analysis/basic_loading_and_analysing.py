import argparse
import inspect
import itertools
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

from sae_lens.load_model import load_model
from sae_lens.saes import TrainingSAE
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


def parse_int_list(value):
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    return [int(p) for p in parts]


def normalize_jobs(sae_paths, features_list, output_files):
    if not sae_paths:
        return []
    if features_list and len(features_list) != len(sae_paths):
        raise ValueError("Number of --features entries must match --sae-path.")
    if output_files and len(output_files) != len(sae_paths):
        raise ValueError("Number of --output-file entries must match --sae-path.")
    jobs = []
    for idx, sae_path in enumerate(sae_paths):
        features = features_list[idx] if features_list else []
        output = output_files[idx] if output_files else ""
        jobs.append({"sae_path": sae_path, "features": features, "output": output})
    return jobs


def get_tokens_slice(dataset, count, device_override=None):
    try:
        tokens = dataset[:count]["tokens"]
    except Exception:
        items = list(itertools.islice(iter(dataset), count))
        tokens = [item["tokens"] for item in items]
    tensor = torch.tensor(tokens)
    if device_override is not None:
        tensor = tensor.to(device_override)
    return tensor


def infer_tl_model_name(model_name_value):
    lower = model_name_value.lower()
    if "llama-3.1-8b" in lower or "llama-8b" in lower:
        return "meta-llama/Llama-3.1-8B"
    if "qwen" in lower and "7b" in lower:
        return "Qwen/Qwen2.5-7B"
    return ""


def map_hook_name(hook_name_value):
    if hook_name_value.startswith("blocks."):
        return hook_name_value
    if hook_name_value.startswith("model.layers."):
        parts = hook_name_value.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            layer_index = int(parts[2])
            return f"blocks.{layer_index}.hook_resid_post"
    raise ValueError(
        f"sae_dashboard expects hook points like 'blocks.N.*', got '{hook_name_value}'."
    )


def build_dashboard_model(base_model, device, model_name_value, dash_tl_model_name):
    from transformer_lens import HookedTransformer

    tl_model_name = dash_tl_model_name or infer_tl_model_name(model_name_value)
    if not tl_model_name:
        raise ValueError(
            "Set --dashboard-tl-model-name to a TransformerLens official model "
            "name (e.g. 'meta-llama/Llama-3.1-8B') so sae_dashboard can build a "
            "HookedTransformer for HF models."
        )
    hf_model = getattr(base_model, "model", None)
    tokenizer = getattr(base_model, "tokenizer", None)
    if hf_model is None:
        raise ValueError(
            "base_model does not expose an HF model; cannot build HookedTransformer."
        )
    hf_model = hf_model.to(device)
    kwargs = {"hf_model": hf_model, "device": device}
    signature = inspect.signature(HookedTransformer.from_pretrained)
    if "tokenizer" in signature.parameters and tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    elif "tokenizer_name" in signature.parameters:
        kwargs["tokenizer_name"] = model_name_value
    if "fold_value_biases" in signature.parameters:
        kwargs["fold_value_biases"] = False
    if "center_writing_weights" in signature.parameters:
        kwargs["center_writing_weights"] = False
    if "fold_ln" in signature.parameters:
        kwargs["fold_ln"] = False
    return HookedTransformer.from_pretrained(tl_model_name, **kwargs)


def resolve_sae_path(path_value):
    if not path_value:
        raise ValueError("Missing SAE path.")
    path = Path(path_value)
    if path.is_file():
        return path
    if path.is_dir():
        if (path / "sae_weights.safetensors").exists() and (path / "cfg.json").exists():
            return path
        final_dirs = sorted(path.rglob("final_*"))
        if final_dirs:
            return final_dirs[-1]
        run_dirs = sorted([p for p in path.iterdir() if p.is_dir()])
        if run_dirs:
            return run_dirs[-1]
    raise FileNotFoundError(f"SAE path not found: {path_value}")


def build_output_name(output_value, sae_path_value):
    if output_value:
        return output_value
    name = Path(sae_path_value).name or "sae"
    return f"feature_dashboard_{name}.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAE feature dashboards for multiple SAE checkpoints."
    )
    parser.add_argument(
        "--model-name",
        default="DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--model-class-name", default="AutoModelForCausalLM")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--dashboard-tl-model-name", default="")
    parser.add_argument("--dataset-path", default="json")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=[
            "/dataset/lmsys-openthoughts-1B-tokenized-deepseek-qwen-7b/combined_1.00b_ratio50.jsonl"
        ],
    )
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--token-limit", type=int, default=10000)
    parser.add_argument("--minibatch-size-features", type=int, default=16)
    parser.add_argument("--minibatch-size-tokens", type=int, default=64)
    parser.add_argument("--sae-path", action="append", default=[])
    parser.add_argument("--features", action="append", default=[])
    parser.add_argument("--output-file", action="append", default=[])
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--tokenizers-parallelism", default="false")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism

    try:
        import google.colab  # type: ignore

        colab = True
    except Exception:
        colab = False

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_from_pretrained_kwargs = {"torch_dtype": args.torch_dtype}
    model = load_model(
        model_class_name=args.model_class_name,
        model_name=args.model_name,
        device=device,
        model_from_pretrained_kwargs=model_from_pretrained_kwargs,
    )

    features_list = [parse_int_list(value) for value in args.features]
    jobs = normalize_jobs(args.sae_path, features_list, args.output_file)
    if not jobs:
        raise ValueError("No SAE jobs configured. Use --sae-path.")

    first_sae_path = resolve_sae_path(jobs[0].get("sae_path"))
    sae_cache = {
        str(first_sae_path): TrainingSAE.load_from_disk(first_sae_path, device=device)
    }
    primary_sae = sae_cache[str(first_sae_path)]

    dataset = load_dataset(
        path=args.dataset_path,
        data_files=args.data_files,
        split="train",
        streaming=args.streaming,
    )

    sample = next(iter(dataset))
    if "tokens" in sample:
        token_dataset = dataset
    elif "input_ids" in sample:
        token_dataset = dataset.rename_column("input_ids", "tokens")
    else:
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,  # type: ignore
            tokenizer=model.tokenizer,  # type: ignore
            streaming=args.streaming,
            max_length=primary_sae.cfg.metadata.context_size,
            add_bos_token=primary_sae.cfg.metadata.prepend_bos,
        )

    tokens_batch = get_tokens_slice(token_dataset, args.token_limit)
    dashboard_model = None

    for job in jobs:
        sae_path = resolve_sae_path(job.get("sae_path"))
        features = job.get("features") or []
        output_file = build_output_name(job.get("output"), sae_path)

        sae = sae_cache.get(str(sae_path))
        if sae is None:
            sae = TrainingSAE.load_from_disk(sae_path, device=device)
            sae_cache[str(sae_path)] = sae

        hook_name = sae.cfg.metadata.hook_name
        hook_point = map_hook_name(hook_name)

        model_for_job = model
        if hook_name.startswith("model.layers."):
            if dashboard_model is None:
                dashboard_model = build_dashboard_model(
                    model, device, args.model_name, args.dashboard_tl_model_name
                )
            model_for_job = dashboard_model

        if not features:
            print(f"Skipping {output_file} (empty feature list).")
            continue

        feature_vis_config = SaeVisConfig(
            hook_point=hook_point,
            features=features,
            minibatch_size_features=args.minibatch_size_features,
            minibatch_size_tokens=args.minibatch_size_tokens,
            verbose=True,
            device=device,
        )

        visualization_data = SaeVisRunner(
            feature_vis_config
        ).run(
            encoder=sae,  # type: ignore
            model=model_for_job,
            tokens=tokens_batch,
        )

        save_feature_centric_vis(sae_vis_data=visualization_data, filename=output_file)

        neuronpedia_id = sae.cfg.metadata.neuronpedia_id
        if neuronpedia_id:
            neuronpedia_quick_list = get_neuronpedia_quick_list(sae, features)
            if colab:
                print(neuronpedia_quick_list)
        else:
            print(
                f"Skipping Neuronpedia quick list for {output_file} (no neuronpedia_id)."
            )


if __name__ == "__main__":
    main()

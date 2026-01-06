import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_lens.load_model import load_model
from sae_lens.saes import SAE, TrainingSAE


@torch.no_grad()
def get_feature_property_df(sae: SAE, feature_sparsity: torch.Tensor):
    """
    feature_property_df = get_feature_property_df(sae, log_feature_density.cpu())
    """

    W_dec_normalized = (
        sae.W_dec.cpu()
    )  # / sparse_autoencoder.W_dec.cpu().norm(dim=-1, keepdim=True)
    W_enc_normalized = (sae.W_enc.cpu() / sae.W_enc.cpu().norm(dim=-1, keepdim=True)).T

    d_e_projection = (W_dec_normalized * W_enc_normalized).sum(-1)
    b_dec_projection = sae.b_dec.cpu() @ W_dec_normalized.T

    return pd.DataFrame(
        {
            "log_feature_sparsity": feature_sparsity + 1e-10,
            "d_e_projection": d_e_projection,
            # "d_e_projection_normalized": d_e_projection_normalized,
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec_projection": b_dec_projection,
            "feature": list(range(sae.cfg.d_sae)),  # type: ignore
            "dead_neuron": (feature_sparsity < -9).cpu(),
        }
    )


@torch.no_grad()
def get_stats_df(projection: torch.Tensor):
    """
    Returns a dataframe with the mean, std, skewness and kurtosis of the projection
    """
    mean = projection.mean(dim=1, keepdim=True)
    diffs = projection - mean
    var = (diffs**2).mean(dim=1, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), dim=1)
    kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=1)

    return pd.DataFrame(
        {
            "feature": range(len(skews)),
            "mean": mean.numpy().squeeze(),
            "std": std.numpy().squeeze(),
            "skewness": skews.numpy(),
            "kurtosis": kurtosis.numpy(),
        }
    )


@torch.no_grad()
def get_all_stats_dfs(
    gpt2_small_sparse_autoencoders: dict[str, SAE],  # [hook_point, sae]
    gpt2_small_sae_sparsities: dict[str, torch.Tensor],  # [hook_point, sae]
    model: HookedTransformer,
    cosine_sim: bool = False,
):
    stats_dfs = []
    pbar = tqdm(gpt2_small_sparse_autoencoders.keys())
    for key in pbar:
        layer = int(key.split(".")[1])
        sparse_autoencoder = gpt2_small_sparse_autoencoders[key]
        pbar.set_description(f"Processing layer {sparse_autoencoder.cfg.hook_name}")
        W_U_stats_df_dec, _ = get_W_U_W_dec_stats_df(
            sparse_autoencoder.W_dec.cpu(), model, cosine_sim
        )
        log_feature_sparsity = gpt2_small_sae_sparsities[key].detach().cpu()
        W_U_stats_df_dec["log_feature_sparsity"] = log_feature_sparsity
        W_U_stats_df_dec["layer"] = layer + (1 if "post" in key else 0)
        stats_dfs.append(W_U_stats_df_dec)

    return pd.concat(stats_dfs, axis=0)


@torch.no_grad()
def get_W_U_W_dec_stats_df(
    W_dec: torch.Tensor, model: HookedTransformer, cosine_sim: bool = False
) -> tuple[pd.DataFrame, torch.Tensor]:
    W_U = model.W_U.detach().cpu()
    if cosine_sim:
        W_U = W_U / W_U.norm(dim=0, keepdim=True)
    dec_projection_onto_W_U = W_dec @ W_U
    W_U_stats_df = get_stats_df(dec_projection_onto_W_U)
    return W_U_stats_df, dec_projection_onto_W_U


def parse_feature_indices(value):
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    parts = [p for p in text.replace(" ", ",").split(",") if p]
    return [int(p) for p in parts]


def parse_float_list(value):
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        return [float(x) for x in json.loads(text)]
    parts = [p for p in text.replace(" ", ",").split(",") if p]
    return [float(p) for p in parts]


def load_hidden_state_vector(path, hidden_position):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hidden state file not found: {path}")
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "hidden_state" in data:
            data = data["hidden_state"]
        tensor = torch.as_tensor(data)
    elif path.suffix == ".npy":
        tensor = torch.from_numpy(np.load(path))
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        tensor = torch.as_tensor(data)
    else:
        raise ValueError(
            "Unsupported hidden state format. Use .pt, .pth, .npy, or .json."
        )

    if tensor.dim() == 2:
        tensor = tensor[hidden_position]
    elif tensor.dim() > 2:
        raise ValueError("Hidden state tensor must be 1D or 2D.")
    return tensor.float().cpu()


def load_prompts(jsonl_path, text_field, max_prompts, seed):
    rng = np.random.default_rng(seed)
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get(text_field)
            if text:
                prompts.append(text)
    if not prompts:
        raise ValueError("No prompts loaded from jsonl.")
    if max_prompts and len(prompts) > max_prompts:
        prompts = list(rng.choice(prompts, size=max_prompts, replace=False))
    return prompts


def iter_prompt_batches(prompts, batch_size):
    if batch_size is None or batch_size <= 0:
        yield prompts
        return
    for idx in range(0, len(prompts), batch_size):
        yield prompts[idx : idx + batch_size]


def get_batch_count(num_items, batch_size):
    if batch_size is None or batch_size <= 0:
        return 1
    return math.ceil(num_items / batch_size)


def load_sae(path, device):
    try:
        return TrainingSAE.load_from_disk(path, device=device)
    except Exception:
        return SAE.load_from_disk(path, device=device)


def get_unembedding_matrix(model, d_in):
    if hasattr(model, "W_U"):
        W_U = model.W_U.detach()
        if W_U.shape[0] != d_in and W_U.shape[1] == d_in:
            W_U = W_U.T
        return W_U
    hf_model = getattr(model, "model", None)
    if hf_model is None:
        raise ValueError("Model does not expose W_U or a HuggingFace model.")
    output_emb = hf_model.get_output_embeddings()
    if output_emb is None:
        if hasattr(hf_model, "lm_head"):
            output_emb = hf_model.lm_head
        else:
            raise ValueError("Cannot locate output embeddings on HF model.")
    W = output_emb.weight.detach()
    if W.shape[0] == d_in:
        return W
    if W.shape[1] == d_in:
        return W.T
    raise ValueError(f"Unexpected output embedding shape: {W.shape}")


def get_output_embeddings(hf_model):
    output_emb = hf_model.get_output_embeddings()
    if output_emb is None and hasattr(hf_model, "lm_head"):
        output_emb = hf_model.lm_head
    if output_emb is None:
        raise ValueError("Cannot locate output embeddings on HF model.")
    return output_emb


def apply_final_norm(hf_model, hidden):
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "norm"):
        return hf_model.model.norm(hidden)
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "ln_f"):
        return hf_model.transformer.ln_f(hidden)
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "final_layernorm"):
        return hf_model.model.final_layernorm(hidden)
    return hidden


def extract_layer_index(hook_name):
    parts = hook_name.split(".")
    for idx, part in enumerate(parts):
        if part.isdigit():
            return int(part)
    return None


def get_layer_module(hf_model, layer_idx):
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers[layer_idx]
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h[layer_idx]
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return hf_model.gpt_neox.layers[layer_idx]
    raise ValueError("Unsupported model type for layer hooks.")


def get_token_text(tokenizer, token_id):
    if tokenizer is None:
        return str(token_id)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    else:
        token = str(token_id)
    if token is None:
        token = str(token_id)
    try:
        decoded = tokenizer.decode([int(token_id)])
    except Exception:
        decoded = token
    if decoded is None:
        decoded = token
    decoded = decoded.replace("\n", "\\n")
    token = token.replace("\n", "\\n")
    return f"{token} | {decoded}"


def compute_projection(
    sae, model, feature_indices, cosine_sim=False, dtype=torch.float32
):
    if not feature_indices:
        raise ValueError("Empty feature index list.")
    W_dec = sae.W_dec.detach().to(dtype=dtype).cpu()
    W_dec = W_dec[feature_indices]
    W_U = get_unembedding_matrix(model, W_dec.shape[1]).to(dtype=dtype).cpu()
    if cosine_sim:
        W_U = W_U / W_U.norm(dim=0, keepdim=True)
    projection = W_dec @ W_U
    return projection


def compute_stats(values):
    values = np.asarray(values)
    mean = values.mean()
    std = values.std()
    if std == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        zscores = (values - mean) / std
        skewness = np.mean(zscores**3)
        kurtosis = np.mean(zscores**4)
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "min": values.min(),
        "max": values.max(),
        "q01": quantiles[0],
        "q05": quantiles[1],
        "q50": quantiles[2],
        "q95": quantiles[3],
        "q99": quantiles[4],
    }


def top_k_tokens(values, k):
    values = np.asarray(values)
    top_pos_idx = np.argpartition(values, -k)[-k:]
    top_neg_idx = np.argpartition(values, k)[:k]
    top_pos_idx = top_pos_idx[np.argsort(values[top_pos_idx])[::-1]]
    top_neg_idx = top_neg_idx[np.argsort(values[top_neg_idx])]
    return top_pos_idx, top_neg_idx


def make_histogram(values, feature_idx):
    fig = px.histogram(
        x=values,
        nbins=200,
        title=f"Feature {feature_idx}: Logit Weight Distribution",
        labels={"x": "Logit Weight", "y": "Count"},
        height=350,
    )
    return fig


def make_rank_plot(values, feature_idx):
    sorted_vals = np.sort(values)[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sorted_vals, mode="lines"))
    fig.update_layout(
        title=f"Feature {feature_idx}: Sorted Logit Weights",
        xaxis_title="Rank",
        yaxis_title="Logit Weight",
        height=350,
    )
    return fig


def make_bar_plot(df, title):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df["token"], y=df["logit_weight"], text=df["logit_weight"])
    )
    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="Logit Weight",
        height=350,
    )
    return fig


def make_trajectory_plot(layer_ids, layer_deltas, token_ids, tokenizer, title):
    fig = go.Figure()
    for token_id in token_ids:
        fig.add_trace(
            go.Scatter(
                x=layer_ids,
                y=layer_deltas[:, token_id],
                mode="lines+markers",
                name=get_token_text(tokenizer, token_id),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Logit Delta",
        height=420,
    )
    return fig


def write_feature_report(
    output_path,
    feature_idx,
    stats,
    top_pos_df,
    top_neg_df,
    figures,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        f"<h1>Feature {feature_idx}</h1>",
        "<h2>Summary Statistics</h2>",
        pd.DataFrame([stats]).to_html(index=False),
        "<h2>Top Positive Tokens</h2>",
        top_pos_df.to_html(index=False),
        "<h2>Top Negative Tokens</h2>",
        top_neg_df.to_html(index=False),
    ]

    include_js = True
    for fig in figures:
        html_parts.append(pio.to_html(fig, include_plotlyjs="cdn" if include_js else False))
        include_js = False

    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def analyze_features(
    sae,
    model,
    feature_indices,
    output_dir,
    top_k=20,
    cosine_sim=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = getattr(model, "tokenizer", None)

    projection = compute_projection(sae, model, feature_indices, cosine_sim=cosine_sim)
    projection = projection.cpu().numpy()

    summary_rows = []
    for local_idx, feature_idx in enumerate(feature_indices):
        values = projection[local_idx]
        stats = compute_stats(values)
        stats["feature"] = feature_idx
        summary_rows.append(stats)

        top_pos_idx, top_neg_idx = top_k_tokens(values, top_k)
        top_pos_df = pd.DataFrame(
            {
                "token_id": top_pos_idx,
                "token": [get_token_text(tokenizer, tid) for tid in top_pos_idx],
                "logit_weight": values[top_pos_idx],
            }
        )
        top_neg_df = pd.DataFrame(
            {
                "token_id": top_neg_idx,
                "token": [get_token_text(tokenizer, tid) for tid in top_neg_idx],
                "logit_weight": values[top_neg_idx],
            }
        )

        hist_fig = make_histogram(values, feature_idx)
        rank_fig = make_rank_plot(values, feature_idx)
        pos_fig = make_bar_plot(top_pos_df, f"Feature {feature_idx}: Top Positive Tokens")
        neg_fig = make_bar_plot(top_neg_df, f"Feature {feature_idx}: Top Negative Tokens")

        report_path = output_dir / f"feature_{feature_idx}.html"
        write_feature_report(
            report_path,
            feature_idx,
            stats,
            top_pos_df,
            top_neg_df,
            [hist_fig, rank_fig, pos_fig, neg_fig],
        )

        top_pos_df.to_csv(output_dir / f"feature_{feature_idx}_top_positive.csv", index=False)
        top_neg_df.to_csv(output_dir / f"feature_{feature_idx}_top_negative.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "feature_summary.csv", index=False)
    return summary_df


def analyze_hidden_vector_logits(
    hidden_vector,
    model,
    output_dir,
    top_k=20,
    label="hidden",
    apply_norm=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = getattr(model, "tokenizer", None)
    hf_model = getattr(model, "model", None)
    if hf_model is None:
        raise ValueError("Model wrapper does not expose a HuggingFace model.")

    hidden = hidden_vector.to(hf_model.device)
    if apply_norm:
        hidden = apply_final_norm(hf_model, hidden[None, None, :]).squeeze(0).squeeze(0)

    W_U = get_unembedding_matrix(model, hidden.shape[0]).to(hidden.dtype)
    logits = (hidden @ W_U).detach().float().cpu().numpy()

    stats = compute_stats(logits)
    stats["label"] = label

    top_pos_idx, top_neg_idx = top_k_tokens(logits, top_k)
    top_pos_df = pd.DataFrame(
        {
            "token_id": top_pos_idx,
            "token": [get_token_text(tokenizer, tid) for tid in top_pos_idx],
            "logit_weight": logits[top_pos_idx],
        }
    )
    top_neg_df = pd.DataFrame(
        {
            "token_id": top_neg_idx,
            "token": [get_token_text(tokenizer, tid) for tid in top_neg_idx],
            "logit_weight": logits[top_neg_idx],
        }
    )

    hist_fig = make_histogram(logits, label)
    rank_fig = make_rank_plot(logits, label)
    pos_fig = make_bar_plot(top_pos_df, f"{label}: Top Positive Tokens")
    neg_fig = make_bar_plot(top_neg_df, f"{label}: Top Negative Tokens")

    output_path = output_dir / f"{label}_logits.html"
    write_feature_report(
        output_path,
        label,
        stats,
        top_pos_df,
        top_neg_df,
        [hist_fig, rank_fig, pos_fig, neg_fig],
    )

    top_pos_df.to_csv(output_dir / f"{label}_top_positive.csv", index=False)
    top_neg_df.to_csv(output_dir / f"{label}_top_negative.csv", index=False)
    pd.DataFrame([stats]).to_csv(output_dir / f"{label}_summary.csv", index=False)


def analyze_layerwise_steering(
    sae,
    model,
    feature_indices,
    prompts,
    output_dir,
    coeffs_by_feature,
    top_k,
    max_length,
    logit_position,
    batch_size,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = getattr(model, "tokenizer", None)
    hf_model = getattr(model, "model", None)
    if hf_model is None:
        raise ValueError("Model wrapper does not expose a HuggingFace model.")

    layer_idx = extract_layer_index(sae.cfg.metadata.hook_name)
    if layer_idx is None:
        raise ValueError(f"Could not parse layer from hook name {sae.cfg.metadata.hook_name}")
    layer_module = get_layer_module(hf_model, layer_idx)

    output_emb = get_output_embeddings(hf_model)

    for feature_idx in feature_indices:
        coeff = coeffs_by_feature.get(feature_idx, 0.0)
        steering_vector = sae.W_dec[feature_idx].detach().to(hf_model.device)

        def steering_hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            steer = steering_vector.to(dtype=hidden.dtype)
            hidden = hidden + steer * hidden.new_tensor(coeff)
            if rest is None:
                return hidden
            return (hidden, *rest)

        rows = []
        layer_sums = None
        layer_weights = None
        layer_ids = None
        html_parts = [
            "<html><head><meta charset='utf-8'></head><body>",
            f"<h1>Layerwise Logit Delta (feature {feature_idx}, coeff={coeff})</h1>",
        ]
        total_batches = get_batch_count(len(prompts), batch_size)
        batch_iter = iter_prompt_batches(prompts, batch_size)
        for prompt_batch in tqdm(
            batch_iter,
            total=total_batches,
            desc=f"Feature {feature_idx} batches",
            leave=False,
        ):
            inputs = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                baseline_outputs = hf_model(**inputs, output_hidden_states=True)
            hidden_states_base = baseline_outputs.hidden_states
            if hidden_states_base is None:
                raise ValueError("Model did not return hidden states.")
            n_layers = len(hidden_states_base) - 1
            start_layer = max(0, layer_idx)

            handle = layer_module.register_forward_hook(steering_hook)
            with torch.no_grad():
                steered_outputs = hf_model(**inputs, output_hidden_states=True)
            handle.remove()

            hidden_states_steered = steered_outputs.hidden_states
            if hidden_states_steered is None:
                raise ValueError("Steered run did not return hidden states.")

            if layer_sums is None:
                layer_sums = [None] * (n_layers - start_layer)
                layer_weights = [0.0] * (n_layers - start_layer)
                layer_ids = list(range(start_layer, n_layers))
            elif len(hidden_states_steered) - 1 != n_layers:
                raise ValueError("Inconsistent layer count across batches.")

            for idx, layer in enumerate(range(start_layer, n_layers)):
                base_h = hidden_states_base[layer + 1]
                steer_h = hidden_states_steered[layer + 1]

                base_norm = apply_final_norm(hf_model, base_h)
                steer_norm = apply_final_norm(hf_model, steer_h)

                base_logits = output_emb(base_norm)
                steer_logits = output_emb(steer_norm)
                delta = steer_logits - base_logits

                if logit_position == "last":
                    delta = delta[:, -1, :].mean(dim=0)
                    weight = float(delta.shape[0])
                else:
                    delta = delta.mean(dim=(0, 1))
                    weight = float(base_logits.shape[0] * base_logits.shape[1])

                delta = delta.detach().float().cpu().numpy()
                if layer_sums[idx] is None:
                    layer_sums[idx] = delta * weight
                else:
                    layer_sums[idx] += delta * weight
                layer_weights[idx] += weight

        if layer_sums is not None:
            layer_deltas_arr = np.stack(
                [layer_sums[i] / layer_weights[i] for i in range(len(layer_sums))],
                axis=0,
            )
            mean_delta = layer_deltas_arr.mean(axis=0)

            for layer, delta in zip(layer_ids, layer_deltas_arr):
                pos_idx, neg_idx = top_k_tokens(delta, top_k)
                pos_df = pd.DataFrame(
                    {
                        "token_id": pos_idx,
                        "token": [get_token_text(tokenizer, tid) for tid in pos_idx],
                        "logit_delta": delta[pos_idx],
                    }
                )
                neg_df = pd.DataFrame(
                    {
                        "token_id": neg_idx,
                        "token": [get_token_text(tokenizer, tid) for tid in neg_idx],
                        "logit_delta": delta[neg_idx],
                    }
                )

                pos_df.to_csv(
                    output_dir / f"feature_{feature_idx}_layer_{layer}_top_positive.csv",
                    index=False,
                )
                neg_df.to_csv(
                    output_dir / f"feature_{feature_idx}_layer_{layer}_top_negative.csv",
                    index=False,
                )

                rows.append(
                    {
                        "feature": feature_idx,
                        "layer": layer,
                        "top_positive": pos_df.iloc[0]["token"],
                        "top_negative": neg_df.iloc[0]["token"],
                    }
                )

                html_parts.append(f"<h2>Layer {layer}</h2>")
                html_parts.append("<h3>Top Positive Tokens</h3>")
                html_parts.append(pos_df.to_html(index=False))
                html_parts.append("<h3>Top Negative Tokens</h3>")
                html_parts.append(neg_df.to_html(index=False))

            mean_pos_idx, mean_neg_idx = top_k_tokens(mean_delta, top_k)
            mean_pos_df = pd.DataFrame(
                {
                    "token_id": mean_pos_idx,
                    "token": [get_token_text(tokenizer, tid) for tid in mean_pos_idx],
                    "mean_logit_delta": mean_delta[mean_pos_idx],
                }
            )
            mean_neg_df = pd.DataFrame(
                {
                    "token_id": mean_neg_idx,
                    "token": [get_token_text(tokenizer, tid) for tid in mean_neg_idx],
                    "mean_logit_delta": mean_delta[mean_neg_idx],
                }
            )

            mean_pos_df.to_csv(
                output_dir
                / f"feature_{feature_idx}_layerwise_mean_top_positive.csv",
                index=False,
            )
            mean_neg_df.to_csv(
                output_dir
                / f"feature_{feature_idx}_layerwise_mean_top_negative.csv",
                index=False,
            )

            html_parts.insert(
                2,
                "<h2>Top Tokens by Mean Logit Delta Across Layers</h2>",
            )
            html_parts.insert(3, "<h3>Top Positive (Mean)</h3>")
            html_parts.insert(4, mean_pos_df.to_html(index=False))
            html_parts.insert(5, "<h3>Top Negative (Mean)</h3>")
            html_parts.insert(6, mean_neg_df.to_html(index=False))

            pos_traj_fig = make_trajectory_plot(
                layer_ids,
                layer_deltas_arr,
                mean_pos_idx,
                tokenizer,
                "Top Positive Tokens: Logit Delta by Layer (Mean Ranking)",
            )
            neg_traj_fig = make_trajectory_plot(
                layer_ids,
                layer_deltas_arr,
                mean_neg_idx,
                tokenizer,
                "Top Negative Tokens: Logit Delta by Layer (Mean Ranking)",
            )
            html_parts.insert(
                7, pio.to_html(pos_traj_fig, include_plotlyjs="cdn")
            )
            html_parts.insert(
                8, pio.to_html(neg_traj_fig, include_plotlyjs=False)
            )

            traj_rows = []
            token_series_lines = []
            for token_id in np.concatenate([mean_pos_idx, mean_neg_idx]):
                token_text = get_token_text(tokenizer, token_id)
                series = layer_deltas_arr[:, token_id]
                series_text = ",".join(f"{value:.6f}" for value in series)
                token_series_lines.append(f"{token_text}: {series_text}")
                for layer, value in zip(layer_ids, layer_deltas_arr[:, token_id]):
                    traj_rows.append(
                        {
                            "feature": feature_idx,
                            "token_id": int(token_id),
                            "token": token_text,
                            "layer": layer,
                            "logit_delta": float(value),
                        }
                    )
            pd.DataFrame(traj_rows).to_csv(
                output_dir / f"feature_{feature_idx}_layerwise_trajectories.csv",
                index=False,
            )
            (output_dir / f"feature_{feature_idx}_layerwise_token_series.txt").write_text(
                "\n".join(token_series_lines), encoding="utf-8"
            )

        html_parts.append("</body></html>")
        (output_dir / f"feature_{feature_idx}_layerwise.html").write_text(
            "\n".join(html_parts), encoding="utf-8"
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        output_dir / f"feature_{feature_idx}_layerwise_summary.csv", index=False
    )


def analyze_layerwise_hidden_steering(
    model,
    hidden_vector,
    layer_index,
    prompts,
    output_dir,
    coeff,
    top_k,
    max_length,
    logit_position,
    label,
    batch_size,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = getattr(model, "tokenizer", None)
    hf_model = getattr(model, "model", None)
    if hf_model is None:
        raise ValueError("Model wrapper does not expose a HuggingFace model.")

    hidden_vector = hidden_vector.to(hf_model.device)
    layer_module = get_layer_module(hf_model, layer_index)

    output_emb = get_output_embeddings(hf_model)

    def steering_hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        steer = hidden_vector.to(dtype=hidden.dtype)
        hidden = hidden + steer * hidden.new_tensor(coeff)
        if rest is None:
            return hidden
        return (hidden, *rest)

    rows = []
    layer_sums = None
    layer_weights = None
    layer_ids = None
    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        f"<h1>Layerwise Logit Delta (hidden={label}, coeff={coeff})</h1>",
    ]

    total_batches = get_batch_count(len(prompts), batch_size)
    batch_iter = iter_prompt_batches(prompts, batch_size)
    for prompt_batch in tqdm(
        batch_iter,
        total=total_batches,
        desc=f"Hidden {label} batches",
        leave=False,
    ):
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            baseline_outputs = hf_model(**inputs, output_hidden_states=True)
        hidden_states_base = baseline_outputs.hidden_states
        if hidden_states_base is None:
            raise ValueError("Model did not return hidden states.")
        n_layers = len(hidden_states_base) - 1
        start_layer = max(0, layer_index)

        handle = layer_module.register_forward_hook(steering_hook)
        with torch.no_grad():
            steered_outputs = hf_model(**inputs, output_hidden_states=True)
        handle.remove()

        hidden_states_steered = steered_outputs.hidden_states
        if hidden_states_steered is None:
            raise ValueError("Steered run did not return hidden states.")

        if layer_sums is None:
            layer_sums = [None] * (n_layers - start_layer)
            layer_weights = [0.0] * (n_layers - start_layer)
            layer_ids = list(range(start_layer, n_layers))
        elif len(hidden_states_steered) - 1 != n_layers:
            raise ValueError("Inconsistent layer count across batches.")

        for idx, layer in enumerate(range(start_layer, n_layers)):
            base_h = hidden_states_base[layer + 1]
            steer_h = hidden_states_steered[layer + 1]

            base_norm = apply_final_norm(hf_model, base_h)
            steer_norm = apply_final_norm(hf_model, steer_h)

            base_logits = output_emb(base_norm)
            steer_logits = output_emb(steer_norm)
            delta = steer_logits - base_logits

            if logit_position == "last":
                delta = delta[:, -1, :].mean(dim=0)
                weight = float(delta.shape[0])
            else:
                delta = delta.mean(dim=(0, 1))
                weight = float(base_logits.shape[0] * base_logits.shape[1])
            delta = delta.detach().float().cpu().numpy()

            if layer_sums[idx] is None:
                layer_sums[idx] = delta * weight
            else:
                layer_sums[idx] += delta * weight
            layer_weights[idx] += weight

    if layer_sums is not None:
        layer_deltas_arr = np.stack(
            [layer_sums[i] / layer_weights[i] for i in range(len(layer_sums))],
            axis=0,
        )
        mean_delta = layer_deltas_arr.mean(axis=0)

        for layer, delta in zip(layer_ids, layer_deltas_arr):
            pos_idx, neg_idx = top_k_tokens(delta, top_k)
            pos_df = pd.DataFrame(
                {
                    "token_id": pos_idx,
                    "token": [get_token_text(tokenizer, tid) for tid in pos_idx],
                    "logit_delta": delta[pos_idx],
                }
            )
            neg_df = pd.DataFrame(
                {
                    "token_id": neg_idx,
                    "token": [get_token_text(tokenizer, tid) for tid in neg_idx],
                    "logit_delta": delta[neg_idx],
                }
            )

            pos_df.to_csv(
                output_dir / f"hidden_{label}_layer_{layer}_top_positive.csv",
                index=False,
            )
            neg_df.to_csv(
                output_dir / f"hidden_{label}_layer_{layer}_top_negative.csv",
                index=False,
            )

            rows.append(
                {
                    "label": label,
                    "layer": layer,
                    "top_positive": pos_df.iloc[0]["token"],
                    "top_negative": neg_df.iloc[0]["token"],
                }
            )

            html_parts.append(f"<h2>Layer {layer}</h2>")
            html_parts.append("<h3>Top Positive Tokens</h3>")
            html_parts.append(pos_df.to_html(index=False))
            html_parts.append("<h3>Top Negative Tokens</h3>")
            html_parts.append(neg_df.to_html(index=False))

        mean_pos_idx, mean_neg_idx = top_k_tokens(mean_delta, top_k)
        mean_pos_df = pd.DataFrame(
            {
                "token_id": mean_pos_idx,
                "token": [get_token_text(tokenizer, tid) for tid in mean_pos_idx],
                "mean_logit_delta": mean_delta[mean_pos_idx],
            }
        )
        mean_neg_df = pd.DataFrame(
            {
                "token_id": mean_neg_idx,
                "token": [get_token_text(tokenizer, tid) for tid in mean_neg_idx],
                "mean_logit_delta": mean_delta[mean_neg_idx],
            }
        )

        mean_pos_df.to_csv(
            output_dir / f"hidden_{label}_layerwise_mean_top_positive.csv",
            index=False,
        )
        mean_neg_df.to_csv(
            output_dir / f"hidden_{label}_layerwise_mean_top_negative.csv",
            index=False,
        )

        html_parts.insert(
            2,
            "<h2>Top Tokens by Mean Logit Delta Across Layers</h2>",
        )
        html_parts.insert(3, "<h3>Top Positive (Mean)</h3>")
        html_parts.insert(4, mean_pos_df.to_html(index=False))
        html_parts.insert(5, "<h3>Top Negative (Mean)</h3>")
        html_parts.insert(6, mean_neg_df.to_html(index=False))

        pos_traj_fig = make_trajectory_plot(
            layer_ids,
            layer_deltas_arr,
            mean_pos_idx,
            tokenizer,
            "Top Positive Tokens: Logit Delta by Layer (Mean Ranking)",
        )
        neg_traj_fig = make_trajectory_plot(
            layer_ids,
            layer_deltas_arr,
            mean_neg_idx,
            tokenizer,
            "Top Negative Tokens: Logit Delta by Layer (Mean Ranking)",
        )
        html_parts.insert(7, pio.to_html(pos_traj_fig, include_plotlyjs="cdn"))
        html_parts.insert(8, pio.to_html(neg_traj_fig, include_plotlyjs=False))

        token_series_lines = []
        for token_id in np.concatenate([mean_pos_idx, mean_neg_idx]):
            token_text = get_token_text(tokenizer, token_id)
            series = layer_deltas_arr[:, token_id]
            series_text = ",".join(f"{value:.6f}" for value in series)
            token_series_lines.append(f"{token_text}: {series_text}")

        (output_dir / f"hidden_{label}_layerwise_token_series.txt").write_text(
            "\n".join(token_series_lines), encoding="utf-8"
        )

        traj_rows = []
        for token_id in np.concatenate([mean_pos_idx, mean_neg_idx]):
            token_text = get_token_text(tokenizer, token_id)
            for layer, value in zip(layer_ids, layer_deltas_arr[:, token_id]):
                traj_rows.append(
                    {
                        "label": label,
                        "token_id": int(token_id),
                        "token": token_text,
                        "layer": layer,
                        "logit_delta": float(value),
                    }
                )
        pd.DataFrame(traj_rows).to_csv(
            output_dir / f"hidden_{label}_layerwise_trajectories.csv",
            index=False,
        )

    html_parts.append("</body></html>")
    (output_dir / f"hidden_{label}_layerwise.html").write_text(
        "\n".join(html_parts), encoding="utf-8"
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        output_dir / f"hidden_{label}_layerwise_summary.csv", index=False
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SAE feature logit lens projections."
    )
    parser.add_argument("--sae-path", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-class-name", default="AutoModelForCausalLM")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--feature-indices", default="")
    parser.add_argument("--output-dir", default="logits_lens_outputs")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cosine-sim", action="store_true")
    parser.add_argument("--prompts-jsonl", default="")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--steer-coeff", type=float, default=1.0)
    parser.add_argument("--steer-coeffs", default="")
    parser.add_argument("--layerwise-top-k", type=int, default=20)
    parser.add_argument(
        "--logit-position",
        choices=["last", "mean"],
        default="last",
    )
    parser.add_argument("--hidden-state-path", default="")
    parser.add_argument("--hidden-position", type=int, default=-1)
    parser.add_argument("--hidden-layer-index", type=int, default=-1)
    parser.add_argument("--hidden-steer-coeff", type=float, default=1.0)
    parser.add_argument("--hidden-label", default="custom")
    parser.add_argument("--hidden-apply-norm", action="store_true")
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--tokenizers-parallelism", default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism

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
    sae = None
    feature_indices = []
    if args.sae_path:
        sae = load_sae(args.sae_path, device=device)
        feature_indices = parse_feature_indices(args.feature_indices)
        if not feature_indices:
            raise ValueError("No feature indices provided.")
        analyze_features(
            sae,
            model,
            feature_indices,
            args.output_dir,
            top_k=args.top_k,
            cosine_sim=args.cosine_sim,
        )
    elif args.feature_indices:
        raise ValueError("--feature-indices requires --sae-path.")

    if not args.sae_path and not args.hidden_state_path:
        raise ValueError("Provide --sae-path or --hidden-state-path to run analysis.")

    hidden_vector = None
    if args.hidden_state_path:
        hidden_vector = load_hidden_state_vector(
            args.hidden_state_path, args.hidden_position
        )
        analyze_hidden_vector_logits(
            hidden_vector,
            model,
            Path(args.output_dir) / "hidden_logits",
            top_k=args.top_k,
            label=args.hidden_label,
            apply_norm=args.hidden_apply_norm,
        )

    if args.prompts_jsonl:
        prompts = load_prompts(
            args.prompts_jsonl,
            args.text_field,
            args.max_prompts,
            args.sample_seed,
        )
        if sae is not None:
            coeff_list = parse_float_list(args.steer_coeffs)
            if coeff_list:
                if len(coeff_list) == 1 and len(feature_indices) > 1:
                    coeff_list = coeff_list * len(feature_indices)
                if len(coeff_list) != len(feature_indices):
                    raise ValueError(
                        "--steer-coeffs must match the number of feature indices."
                    )
                coeffs_by_feature = dict(zip(feature_indices, coeff_list))
            else:
                coeffs_by_feature = {idx: args.steer_coeff for idx in feature_indices}
            analyze_layerwise_steering(
                sae,
                model,
                feature_indices,
                prompts,
                Path(args.output_dir) / "layerwise",
                coeffs_by_feature=coeffs_by_feature,
                top_k=args.layerwise_top_k,
                max_length=args.max_length,
                logit_position=args.logit_position,
                batch_size=args.batch_size,
            )
        if hidden_vector is not None:
            if args.hidden_layer_index < 0:
                raise ValueError("--hidden-layer-index is required for hidden steering.")
            analyze_layerwise_hidden_steering(
                model,
                hidden_vector,
                args.hidden_layer_index,
                prompts,
                Path(args.output_dir) / "layerwise_hidden",
                coeff=args.hidden_steer_coeff,
                top_k=args.layerwise_top_k,
                max_length=args.max_length,
                logit_position=args.logit_position,
                label=args.hidden_label,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    main()

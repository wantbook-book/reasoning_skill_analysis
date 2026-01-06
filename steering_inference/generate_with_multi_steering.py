import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sae_lens import SAE  # pylint: disable=wrong-import-position
from transformers import AutoTokenizer  # pylint: disable=wrong-import-position
from vllm import LLM, SamplingParams  # pylint: disable=wrong-import-position

from utils.steering_utils import (
    add_hooks,
    get_steering_hook,
    get_multi_steering_hook_from_lists,
)  # pylint: disable=wrong-import-position
from utils.utils import load_jsonl, save_jsonl, set_seed  # pylint: disable=wrong-import-position
import torch

def create_steering_hooks(lm_model, hook_configs: List[Dict[str, Any]]):
    hooks: List[Tuple[torch.nn.Module, Any]] = []
    total_layers = len(lm_model.model.layers)
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for cfg in hook_configs:
        hook_layer = cfg["hook_layer"]
        if hook_layer < 0 or hook_layer >= total_layers:
            raise ValueError(f"Invalid hook_layer {hook_layer}; model only has {total_layers} layers.")
        grouped.setdefault(hook_layer, []).append(cfg)

    for hook_layer, cfgs in grouped.items():
        if len(cfgs) == 1:
            cfg = cfgs[0]
            steering_hook = get_steering_hook(
                sae=cfg.get("sae"),
                feature_idx=cfg.get("feature_idx"),
                alpha=cfg.get("alpha", 1.0),
                c_m=cfg.get("c_m"),
                hs_vec=cfg.get("hs_vec"),
            )
        else:
            feature_idxs = [cfg.get("feature_idx") for cfg in cfgs]
            c_ms = [cfg.get("c_m") for cfg in cfgs]
            hs_vecs = [cfg.get("hs_vec") for cfg in cfgs]
            saes = [cfg.get("sae") for cfg in cfgs]
            alphas = [cfg.get("alpha", 1.0) for cfg in cfgs]
            steering_hook = get_multi_steering_hook_from_lists(
                feature_idxs=feature_idxs,
                c_ms=c_ms,
                hs_vecs=hs_vecs,
                alphas=alphas,
                saes=saes,
            )
        hooks.append((lm_model.model.layers[hook_layer], steering_hook))
    return hooks


def _ensure_config_list(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("configs", "steering_configs"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError("Steering config JSON must be a list or dict.")


def load_steering_spec_list(args) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    if args.steering_config_json:
        config_path = os.path.expanduser(args.steering_config_json)
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        configs = _ensure_config_list(payload)
    elif args.sae_path or args.sae_release or args.hs_vec_path:
        configs = [
            {
                "hook_layer": args.hook_layer,
                "feature_idx": args.feature_idx,
                "alpha": args.alpha,
                "c_m": args.c_m,
                "sae_path": args.sae_path,
                "sae_release": args.sae_release,
                "sae_id": args.sae_id,
                "hs_vec_path": args.hs_vec_path,
            }
        ]

    normalized: List[Dict[str, Any]] = []
    for cfg in configs:
        normalized.append(
            {
                "hook_layer": cfg.get("hook_layer", args.hook_layer),
                "feature_idx": cfg.get("feature_idx", args.feature_idx),
                "alpha": cfg.get("alpha", args.alpha),
                "c_m": cfg.get("c_m", args.c_m),
                "sae_path": cfg.get("sae_path", args.sae_path),
                "sae_release": cfg.get("sae_release", args.sae_release),
                "sae_id": cfg.get("sae_id", args.sae_id),
                "hs_vec_path": cfg.get("hs_vec_path", args.hs_vec_path),
            }
        )
    return [
        cfg
        for cfg in normalized
        if cfg.get("sae_path") or cfg.get("sae_release") or cfg.get("hs_vec_path")
    ]


def materialize_steering_configs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not specs:
        return []

    sae_cache: Dict[Any, SAE] = {}
    hs_cache: Dict[str, torch.Tensor] = {}

    def _load_sae_from_spec(spec: Dict[str, Any]) -> Optional[SAE]:
        if spec.get("sae_path"):
            sae_key = ("path", os.path.abspath(os.path.expanduser(spec["sae_path"])))
        elif spec.get("sae_release"):
            sae_key = ("release", spec["sae_release"], spec.get("sae_id"))
        else:
            return None
        if sae_key not in sae_cache:
            if sae_key[0] == "path":
                sae_cache[sae_key] = SAE.load_from_pretrained(path=sae_key[1])
            else:
                release, sae_id = sae_key[1], sae_key[2]
                sae_cache[sae_key], _, _ = SAE.from_pretrained(release=release, sae_id=sae_id)
        return sae_cache[sae_key]

    def _load_hs_vec(path: str) -> torch.Tensor:
        expanded = os.path.abspath(os.path.expanduser(path))
        if expanded not in hs_cache:
            hs_vec_np = np.load(expanded)
            hs_cache[expanded] = torch.from_numpy(hs_vec_np).float()
        return hs_cache[expanded]

    hook_configs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        sae = _load_sae_from_spec(spec)
        hs_vec = None
        if spec.get("hs_vec_path"):
            hs_vec = _load_hs_vec(spec["hs_vec_path"])
        if sae is not None and spec.get("feature_idx") is None:
            raise ValueError(f"Config #{idx} uses SAE steering but no feature_idx was provided.")
        if sae is not None and hs_vec is not None:
            raise ValueError(f"Config #{idx} specifies both SAE and hs_vec steering. Choose one.")
        if sae is None and hs_vec is None:
            raise ValueError(f"Config #{idx} does not specify a steering source.")

        hook_layer = spec.get("hook_layer")
        if hook_layer is None and sae is not None and hasattr(sae, "cfg"):
            hook_layer = getattr(sae.cfg, "hook_layer", None)
        if hook_layer is None:
            raise ValueError(f"Config #{idx} is missing hook_layer.")

        hook_configs.append(
            {
                "hook_layer": hook_layer,
                "feature_idx": spec.get("feature_idx"),
                "alpha": spec.get("alpha", 1.0),
                "c_m": spec.get("c_m"),
                "sae": sae,
                "hs_vec": hs_vec,
            }
        )
    return hook_configs


def load_dataset(data_path: str):
    return load_jsonl(data_path)


def build_logit_bias_map(tokenizer, keyword_list, bias_value):
    bias = {}
    for word in keyword_list:
        tokens = tokenizer(word, add_special_tokens=False).input_ids
        if not tokens:
            continue
        bias[tokens[0]] = float(bias_value)
    return bias


def build_prompts(data_list, tokenizer, apply_chat_template: bool):
    prompts = []
    for item in data_list:
        question = item.get('problem', '')
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = question
        prompts.append(prompt)
    return prompts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    # SAE related arguments
    parser.add_argument(
        "--sae_path",
        type=str,
        default=None,
        help="Path to SAE model (local path).",
    )
    parser.add_argument(
        "--sae_release",
        type=str,
        default=None,
        help="SAE release name from HuggingFace Hub.",
    )
    parser.add_argument(
        "--sae_id",
        type=str,
        default=None,
        help="SAE ID for HuggingFace Hub release.",
    )
    parser.add_argument(
        "--c_m",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--feature_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--hs_vec_path",
        type=str,
        default=None,
        help="Path to hidden state steering vector.",
    )
    parser.add_argument(
        "--steering_config_json",
        type=str,
        default=None,
        help="Path to a JSON file describing multiple steering hooks.",
    )
    parser.add_argument("--hook_layer", type=int, default=None, help="Layer index for steering hook.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Scaling factor for steering intervention.")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG).")
    parser.add_argument("--steer_think_only", action="store_true",
                        help="Apply steering only within <think>...</think> segments.")
    parser.add_argument("--think_end_token", type=str, default="</think>",
                        help="Token/string that marks the end of the thinking block.")
    parser.add_argument("--logit_boost_json", type=str, default=None,
                        help="JSON file containing a list of words to boost.")
    parser.add_argument("--logit_boost_bias", type=float, default=2.0,
                        help="Bias value applied to all boosted tokens.")
    parser.add_argument("--debug_logit_boost", action="store_true",
                        help="Run a quick comparison of outputs with/without logit boost on the first prompt.")

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def main(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    steering_config_runtime = materialize_steering_configs(load_steering_spec_list(args))

    set_seed(args.seed)
    tensor_parallel = len([dev for dev in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if dev.strip()])
    logger.info("Loading model %s (tensor_parallel=%d, pipeline_parallel=%d)", args.model_name_or_path, tensor_parallel, args.pipeline_parallel_size)
    enforce_eager = any(cfg.get("sae") is not None for cfg in steering_config_runtime)
    steering_hooks = []
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=max(tensor_parallel, 1),
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=float(args.gpu_util),
        trust_remote_code=True,
        enforce_eager=enforce_eager,  # Required for SAE hooks to work properly
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if steering_config_runtime:
        lm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        steering_hooks = create_steering_hooks(
            lm_model=lm_model,
            hook_configs=steering_config_runtime,
        )
        layers = sorted({cfg["hook_layer"] for cfg in steering_config_runtime})
        logger.info(
            "Initialized %d steering hooks across layers %s",
            len(steering_hooks),
            layers,
        )

    logit_bias = None
    boost_token_ids = set()
    if args.logit_boost_json:
        with open(args.logit_boost_json, "r", encoding="utf-8") as f:
            keywords = json.load(f)
            if isinstance(keywords, dict):
                keywords = list(keywords.keys())
        logit_bias = build_logit_bias_map(tokenizer, keywords, args.logit_boost_bias)
        boost_token_ids = set(logit_bias.keys())

    data_specs = []
    for data_name in args.data_names.split(","):
        data_name = data_name.strip()
        if not data_name:
            continue
        data_path = os.path.join(args.data_dir, data_name, f"{args.split}.jsonl")
        data_specs.append((data_name, data_path))
    
    for data_name, data_path in data_specs:
        logger.info("Generating outputs for %s", data_path)
        data_list = list(load_dataset(data_path))
        prompts = build_prompts(data_list, tokenizer, args.apply_chat_template)
        logger.info("Loaded %d prompts", len(prompts))

        if args.debug_logit_boost and logit_bias and prompts:
            logger.info("Running logit-boost debug comparison on %s (prompt idx 0)", data_name)
            debug_prompt = prompts[0]

            def run_once(params):
                with add_hooks([], steering_hooks):
                    return llm.generate([debug_prompt], params)[0]

            logprob_params_base = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=1,
                n=1,
                logprobs=20,
            )
            logprob_params_boost = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=1,
                n=1,
                logprobs=20,
                logit_bias=logit_bias,
            )

            base_logprob_out = run_once(logprob_params_base)
            boost_logprob_out = run_once(logprob_params_boost)

            def summarize(entries):
                if isinstance(entries, dict):
                    entries = list(entries.values())
                lines = []
                for entry in entries[:10]:
                    logprob = entry.logprob
                    token_str = entry.decoded_token
                    token_id = tokenizer.convert_tokens_to_ids(token_str)
                    flag = " *boost*" if token_id in boost_token_ids else ""
                    lines.append(f"{token_str}: {logprob:.2f}{flag}")
                return "\n".join(lines)

            def extract_entries(result):
                if not result.outputs:
                    return []
                out = result.outputs[0]
                if not out.logprobs:
                    return []
                return out.logprobs[0]

            base_entries = extract_entries(base_logprob_out)
            boost_entries = extract_entries(boost_logprob_out)
            logger.info("Top logits WITHOUT bias:\n%s", summarize(base_entries))
            logger.info("Top logits WITH bias:\n%s", summarize(boost_entries))

            base_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=min(args.max_tokens_per_call, 256),
                n=1,
            )
            boosted_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=min(args.max_tokens_per_call, 256),
                n=1,
                logit_bias=logit_bias,
            )
            base_output = run_once(base_params).outputs[0].text
            boost_output = run_once(boosted_params).outputs[0].text
            logger.info("Debug prompt:\n%s", debug_prompt)
            logger.info("Without logit boost:\n%s", base_output)
            logger.info("With logit boost:\n%s", boost_output)

        output_items = []
        steer_only_think = bool(steering_hooks and args.steer_think_only)

        if steer_only_think:
            logger.info("Steering enabled only inside <think> ... </think> blocks.")
        else:
            if args.steer_think_only and not steering_hooks:
                logger.warning("--steer_think_only ignored because no steering hooks are configured.")

        sampling_params_full = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=args.n_sampling,
            logit_bias=logit_bias,
        )

        if steer_only_think:
            sampling_params_think = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=args.n_sampling,
                stop=[args.think_end_token],
                logit_bias=logit_bias,
            )

            for data_entry, prompt in zip(data_list, prompts):
                prompt_tokens = len(tokenizer(prompt).input_ids)
                with add_hooks([], steering_hooks, tokens_to_skip=prompt_tokens):
                    think_outputs = llm.generate([prompt], sampling_params_think)[0]

                think_segments = []
                think_token_lengths = []
                for completion in think_outputs.outputs:
                    text = completion.text
                    if not text.endswith(args.think_end_token):
                        text = text + args.think_end_token
                    think_segments.append(text)
                    think_token_lengths.append(len(completion.token_ids))

                completions = []
                token_lengths = []
                for seg, seg_len in zip(think_segments, think_token_lengths):
                    remaining = max(args.max_tokens_per_call - seg_len, 1)
                    sampling_params_answer = SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=remaining,
                        n=1,
                        logit_bias=logit_bias,
                    )
                    answer_prompt = prompt + seg
                    ans_output = llm.generate([answer_prompt], sampling_params_answer)[0]
                    answer_text = ans_output.outputs[0].text
                    completions.append(seg + answer_text)
                    token_lengths.append(seg_len + len(ans_output.outputs[0].token_ids))

                data_entry['resps'] = completions
                data_entry['resps_token_len'] = token_lengths
                output_items.append(data_entry)

            logger.info("Two-stage generation complete for %s", data_path)
        else:
            sampling_params_full = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=args.n_sampling,
                logit_bias=logit_bias,
            )
            with add_hooks([], steering_hooks):
                outputs = llm.generate(prompts, sampling_params_full)
            for data_entry, output in zip(data_list, outputs):
                completions = []
                token_lengths = []
                for completion in output.outputs:
                    completions.append(completion.text)
                    token_lengths.append(len(completion.token_ids))
                data_entry['resps'] = completions
                data_entry['resps_token_len'] = token_lengths
                output_items.append(data_entry)
            logger.info("Generation complete for %s", data_path)

        if args.save_outputs:
            final_dir = os.path.join(args.output_dir, data_name)
            os.makedirs(final_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            output_path = os.path.join(final_dir, f"{base_name}_outputs.jsonl")
            save_jsonl(output_items, output_path)
            logger.info("Saved %d records to %s", len(output_items), output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)

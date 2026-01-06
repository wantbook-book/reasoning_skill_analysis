import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sae_lens import SAE  # pylint: disable=wrong-import-position
from transformers import AutoTokenizer  # pylint: disable=wrong-import-position
from vllm import LLM, SamplingParams  # pylint: disable=wrong-import-position

from utils.steering_utils import add_hooks, get_steering_hook  # pylint: disable=wrong-import-position
from utils.utils import load_jsonl, save_jsonl, set_seed  # pylint: disable=wrong-import-position
import torch

def create_steering_hooks(lm_model, hook_layer: int, sae=None, feature_idx=None, alpha=1.0, c_m=None, hs_vec=None):
    steering_hook = get_steering_hook(
        sae=sae,
        feature_idx=feature_idx,
        alpha=alpha,
        c_m=c_m,
        hs_vec=hs_vec,
    )
    return [
        (
            lm_model.model.layers[hook_layer],
            steering_hook,
        )
    ]


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
        help="Path to hidden staet steering vector.",
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

    set_seed(args.seed)
    tensor_parallel = len([dev for dev in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if dev.strip()])
    logger.info("Loading model %s (tensor_parallel=%d, pipeline_parallel=%d)", args.model_name_or_path, tensor_parallel, args.pipeline_parallel_size)
    steering_hooks = []
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=max(tensor_parallel, 1),
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=float(args.gpu_util),
        trust_remote_code=True,
        enforce_eager=True if args.sae_path or args.sae_release else False  # Required for SAE hooks to work properly
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

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
    
    # load SAE if provided
    sae, hs_vec, steering_hooks = None, None, []
    if args.sae_path or args.sae_release:
        if args.sae_path:
            sae = SAE.load_from_pretrained(path=args.sae_path)
        else:
            sae, _, _ = SAE.from_pretrained(release=args.sae_release, sae_id=args.sae_id)
    elif args.hs_vec_path:
        hs_vec_np = np.load(args.hs_vec_path)
        hs_vec = torch.from_numpy(hs_vec_np).float()
    if (sae is not None or hs_vec is not None) and args.hook_layer is None:
        raise ValueError("Steering requires --hook_layer to be specified.")
    if sae is not None or hs_vec is not None:
        lm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        steering_hooks = create_steering_hooks(
            lm_model=lm_model,
            hook_layer=args.hook_layer,
            sae=sae,
            feature_idx=args.feature_idx,
            alpha=args.alpha,
            c_m=args.c_m,
            hs_vec=hs_vec
        )
        logger.info("Initialized steering hooks on layer %s", args.hook_layer)
    
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

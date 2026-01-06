#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI


# ---------------------------
# Utils
# ---------------------------

def read_lines(path: str, keep_empty: bool = False) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if (not keep_empty) and (s.strip() == ""):
                continue
            lines.append(s)
    return lines


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_jsonl(output_text: str) -> List[Dict[str, Any]]:
    """Parse strict JSONL: one JSON object per line."""
    items: List[Dict[str, Any]] = []
    for raw in output_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = safe_json_loads(raw)
        if obj is None:
            raise ValueError(f"Invalid JSONL line: {raw[:120]}")
        items.append(obj)
    return items


def extract_json_block(text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def parse_jsonl_with_extraction(output_text: str) -> Tuple[List[Dict[str, Any]], bool]:
    try:
        return parse_jsonl(output_text), False
    except Exception:
        extracted = extract_json_block(output_text)
        if extracted:
            return parse_jsonl(extracted), True
        raise


def chunk_list(xs: List[str], batch_size: int) -> List[List[str]]:
    return [xs[i:i + batch_size] for i in range(0, len(xs), batch_size)]


# ---------------------------
# OpenAI call with retry
# ---------------------------

def get_usage_counts(resp: Any) -> Tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    else:
        input_tokens = (
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", 0)
            or 0
        )
        output_tokens = (
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", 0)
            or 0
        )
    return int(input_tokens), int(output_tokens)


def call_openai_once(
    client: OpenAI,
    model: str,
    input_text: str,
    temperature: float,
    max_output_tokens: Optional[int] = None,
) -> Tuple[str, int, int]:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": input_text}],
        "temperature": temperature,
        "reasoning_effort": "low"
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens

    resp = client.chat.completions.create(**kwargs)
    input_tokens, output_tokens = get_usage_counts(resp)
    # Official SDK example uses resp.output_text for final text :contentReference[oaicite:1]{index=1}
    content = resp.choices[0].message.content
    return content or "", input_tokens, output_tokens


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    input_text: str,
    temperature: float,
    max_output_tokens: Optional[int],
    max_retries: int = 6,
) -> Tuple[str, int, int]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return call_openai_once(
                client=client,
                model=model,
                input_text=input_text,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:
            last_err = e
            # Exponential backoff + jitter
            sleep_s = min(60.0, (2.0 ** attempt) + random.random())
            print(f"[warn] API call failed (attempt {attempt+1}/{max_retries}): {e}")
            print(f"[warn] sleeping {sleep_s:.2f}s then retry...")
            time.sleep(sleep_s)
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}")


# ---------------------------
# Batch processing (with fallback splitting)
# ---------------------------

def build_request_text(prompt_template: str, placeholder: str, phrases: List[str]) -> str:
    batch_text = "\n".join(phrases)
    if placeholder in prompt_template:
        return prompt_template.replace(placeholder, batch_text)
    # If your prompt has no placeholder, append to the end.
    return prompt_template.rstrip() + "\n\nINPUT:\n" + batch_text + "\n"


def normalize_items(
    items: List[Dict[str, Any]],
    phrases: List[str],
) -> List[Dict[str, Any]]:
    """
    Align outputs to inputs:
    - Force the phrase field to match the original input (avoid model rewrites)
    - Allow extra fields, but require at least label/strength/reason (per your prompt)
    """
    norm: List[Dict[str, Any]] = []
    for i, phrase in enumerate(phrases):
        obj = items[i] if i < len(items) else {}
        if not isinstance(obj, dict):
            obj = {}
        obj["phrase"] = phrase
        # Fallback defaults
        obj.setdefault("label", "NONE")
        obj.setdefault("strength", "none" if obj["label"] == "NONE" else "scaffold")
        obj.setdefault("reason", "")
        norm.append(obj)
    return norm


def process_batch_recursive(
    client: OpenAI,
    model: str,
    prompt_template: str,
    placeholder: str,
    phrases: List[str],
    temperature: float,
    max_output_tokens: Optional[int],
    soft_repair_suffix: str,
    out_path: str,
    batch_idx: int,
    depth: int = 0,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Run a full batch; if parsing fails or counts mismatch, retry with stronger constraints.
    If it still fails, recursively split until usable (worst case: single item).
    """
    indent = "  " * depth
    req = build_request_text(prompt_template, placeholder, phrases)

    # 1) Normal attempt
    total_input_tokens = 0
    total_output_tokens = 0
    out_text, in_tok, out_tok = call_openai_with_retry(
        client=client,
        model=model,
        input_text=req,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    try:
        items, used_extract = parse_jsonl_with_extraction(out_text)
        if len(items) != len(phrases):
            print(f"Count mismatch: got {len(items)} items, expected {len(phrases)}")
        if used_extract:
            print(f"{indent}[info] extracted JSON from code block (batch {batch_idx})")
        return normalize_items(items, phrases), total_input_tokens, total_output_tokens
    except Exception as e1:
        print(f"{indent}[warn] batch parse/count failed: {e1}")

    # 2) Soft repair: append stronger format constraints and retry
    req2 = req.rstrip() + "\n\n" + soft_repair_suffix
    out_text2, in_tok, out_tok = call_openai_with_retry(
        client=client,
        model=model,
        input_text=req2,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    try:
        items2, used_extract = parse_jsonl_with_extraction(out_text2)
        if len(items2) != len(phrases):
            print(f"Count mismatch: got {len(items2)} items, expected {len(phrases)}")
        if used_extract:
            print(f"{indent}[info] extracted JSON from code block (batch {batch_idx})")
        return normalize_items(items2, phrases), total_input_tokens, total_output_tokens
    except Exception as e2:
        print(f"{indent}[warn] repair attempt failed: {e2}")

    log_path = out_path
    if log_path.endswith(".jsonl"):
        log_path = log_path[: -len(".jsonl")] + f"{batch_idx}.txt"
    else:
        log_path = log_path + f".{batch_idx}.txt"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"==== batch {batch_idx} raw_output ====\n")
        handle.write(out_text)
        if not out_text.endswith("\n"):
            handle.write("\n")
        handle.write(f"==== batch {batch_idx} repair_output ====\n")
        handle.write(out_text2)
        if not out_text2.endswith("\n"):
            handle.write("\n")

    # 3) Still fails: no more splitting, fallback directly
    fallback_items = [
        {"label": "NONE", "strength": "none", "reason": "parse_failed"}
        for _ in phrases
    ]
    return (
        normalize_items(fallback_items, phrases),
        total_input_tokens,
        total_output_tokens,
    )


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Prompt template file (txt)")
    ap.add_argument("--phrases", required=True, help="Input phrases file, one phrase per line")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument("--model", default="gpt-5-mini", help="Model name, e.g. gpt-5, gpt-5-mini")
    ap.add_argument("--batch-size", type=int, default=200, help="Number of phrases per API call")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0-0.2 recommended)")
    ap.add_argument("--max-output-tokens", type=int, default=8000, help="Cap output tokens (optional)")
    ap.add_argument("--placeholder", default="<<<PASTE YOUR PHRASES HERE, ONE PER LINE>>>",
                    help="Placeholder string in prompt template to be replaced by batch phrases")
    ap.add_argument("--resume", action="store_true", help="Resume from existing out.jsonl line count")
    args = ap.parse_args()

    prompt_template = open(args.prompt, "r", encoding="utf-8").read()
    all_phrases = read_lines(args.phrases, keep_empty=False)

    start_idx = 0
    if args.resume:
        done = count_jsonl_lines(args.out)
        start_idx = min(done, len(all_phrases))
        print(f"[info] resume enabled: already have {done} lines, start at idx={start_idx}")

    phrases = all_phrases[start_idx:]
    batches = chunk_list(phrases, args.batch_size)

    client = OpenAI()

    # Append to the request to enforce strict JSONL output.
    soft_repair_suffix = (
        "REMINDER: Output MUST be strict JSONL with exactly one JSON object per input line, "
        "in the same order, and no extra text."
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    with open(args.out, mode, encoding="utf-8") as fout:
        total = len(phrases)
        processed = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for bi, batch in enumerate(batches):
            print(f"[info] batch {bi+1}/{len(batches)}: size={len(batch)}")
            items, in_tok, out_tok = process_batch_recursive(
                client=client,
                model=args.model,
                prompt_template=prompt_template,
                placeholder=args.placeholder,
                phrases=batch,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens if args.max_output_tokens > 0 else None,
                soft_repair_suffix=soft_repair_suffix,
                out_path=args.out,
                batch_idx=bi + 1,
            )
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            for obj in items:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            processed += len(batch)
            print(
                "[info] progress: "
                f"{processed}/{total} (this run), "
                f"tokens in={total_input_tokens}, out={total_output_tokens}"
            )

    print(
        f"[done] wrote: {args.out} "
        f"(tokens in={total_input_tokens}, out={total_output_tokens})"
    )


if __name__ == "__main__":
    main()

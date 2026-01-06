import json
from typing import List, Dict, Any

import numpy as np
import os


def compute_method_stats(files: List[str], methods: List[str], dataset_names: List[str],
                         k: int, keyword_json: str):
    """
    Summarize evaluation outputs across multiple methods.
    Each file should be a JSONL-like file saved by evaluate_outputs.py, containing fields:
    problem, answer, level, idx, resps, resps_token_len, scores, extracted_answers.
    """
    with open(keyword_json, "r", encoding="utf-8") as f:
        keyword_map = json.load(f)

    print(f"{'Method':15} {'Dataset':15} {'avg@' + str(k):>15} {'TokenLen':>15} "
          f"{'DR freq':>15} {'EE freq':>15} {'SR freq':>15} "
          f"{'DR count':>15} {'EE count':>15} {'SR count':>15}")

    for file_path, method, dataset in zip(files, methods, dataset_names):
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        metrics_path = file_path.replace(".jsonl", "_metrics.json")
        file_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                file_metrics = json.load(f)
        avg_metric = file_metrics.get('avg_k')
        if avg_metric is not None:
            file_metrics['avg_k'] = float(avg_metric) * 100
        std_metric = file_metrics.get('std_k')
        if std_metric is not None:
            file_metrics['std_k'] = float(std_metric) * 100

        avg_scores = []
        avg_token_lens = []
        category_freq = {cat: [] for cat in keyword_map}
        category_count = {cat: [] for cat in keyword_map}

        for entry in data:
            resps = entry.get('resps', [])
            token_lens = entry.get('resps_token_len', [])
            scores = entry.get('scores', [])

            if not resps or not scores:
                continue

            top_k_scores = scores[:k] if len(scores) >= k else scores
            avg_scores.append(sum(top_k_scores) / len(top_k_scores) * 100)

            if token_lens:
                top_k_tokens = token_lens[:k] if len(token_lens) >= k else token_lens
                avg_token_lens.append(sum(top_k_tokens) / len(top_k_tokens))

            for resp, tok_len in zip(resps, token_lens or [len(resp) for resp in resps]):
                lowered = resp.lower()
                for cat, keywords in keyword_map.items():
                    count = sum(lowered.count(word.lower()) for word in keywords)
                    category_count[cat].append(count)
                    freq = (count / tok_len * 100) if tok_len else 0
                    category_freq[cat].append(freq)

        avg_score = np.mean(avg_scores) if avg_scores else 0
        std_score = np.std(avg_scores) if len(avg_scores) > 1 else 0
        avg_len = np.mean(avg_token_lens) if avg_token_lens else 0
        std_len = np.std(avg_token_lens) if len(avg_token_lens) > 1 else 0

        freqs = {}
        freq_stds = {}
        counts = {}
        count_stds = {}
        for cat, vals in category_freq.items():
            freqs[cat] = np.mean(vals) if vals else 0
            freq_stds[cat] = np.std(vals) if len(vals) > 1 else 0
        for cat, vals in category_count.items():
            counts[cat] = np.mean(vals) if vals else 0
            count_stds[cat] = np.std(vals) if len(vals) > 1 else 0

        dr_freq = f"{freqs.get('deep_reasoning', 0):.2f}±{freq_stds.get('deep_reasoning', 0):.2f}"
        ee_freq = f"{freqs.get('extensive_exploration', 0):.2f}±{freq_stds.get('extensive_exploration', 0):.2f}"
        sr_freq = f"{freqs.get('sensible_reflection', 0):.2f}±{freq_stds.get('sensible_reflection', 0):.2f}"
        dr_count = f"{counts.get('deep_reasoning', 0):.1f}±{count_stds.get('deep_reasoning', 0):.1f}"
        ee_count = f"{counts.get('extensive_exploration', 0):.1f}±{count_stds.get('extensive_exploration', 0):.1f}"
        sr_count = f"{counts.get('sensible_reflection', 0):.1f}±{count_stds.get('sensible_reflection', 0):.1f}"

        avg_display = f"{file_metrics.get('avg_k', avg_score):.1f}±{file_metrics.get('std_k', std_score):.1f}"
        len_display = f"{avg_len:.0f}±{std_len:.0f}"

        print(f"{method:15} {dataset:15} {avg_display:>15} {len_display:>15} "
              f"{dr_freq:>15} {ee_freq:>15} {sr_freq:>15} "
              f"{dr_count:>15} {ee_count:>15} {sr_count:>15}")

if __name__ == "__main__":
    # Example usage
    files = [
        "eval_outputs_method1.jsonl",
        "eval_outputs_method2.jsonl"
    ]
    methods = [
        "Method 1",
        "Method 2"
    ]
    dataset_names = [
        "Dataset A",
        "Dataset A"
    ]
    k = 3
    keyword_json = "keywords.json"

    compute_method_stats(files, methods, dataset_names, k, keyword_json)

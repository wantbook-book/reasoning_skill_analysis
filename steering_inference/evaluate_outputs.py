import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Tuple

sys.path.append('../')
from utils.utils import load_jsonl, save_jsonl  # pylint: disable=wrong-import-position
from math_verify.metric import math_metric  # pylint: disable=wrong-import-position
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)

_VERIFY_FUNC_CACHE: Dict[bool, object] = {}
_MULTI_CHOICE_DATASETS = {
    "gpqa_diamond",
    "mmlu_stem",
    "gpqa",
    "mmlu",
    "mmlu_pro",
    "mmlu_stem_500",
}
_CHOICE_PATTERNS = (
    re.compile(r"answer\s+is\s+\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"answer:\s*([A-J])", re.IGNORECASE),
)
_CHOICE_FALLBACK = re.compile(r"\b([A-J])\b")

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and evaluate answers using sympy')
    parser.add_argument('--input_template', type=str, required=True,
                        help='Template path with {data_name}, e.g., /path/{data_name}/file.jsonl')
    parser.add_argument('--data_names', type=str, required=True,
                        help='Comma separated list of data names to evaluate')
    parser.add_argument('--gold_is_latex', action='store_true', help='Use LaTeX normalization when parsing gold answers')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def _extract_choice_answer(response: str) -> str | None:
    """
    Extract a multiple-choice answer (A-J) from the raw response.
    """
    if not response:
        return None
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(response)
    if match:
        return match.group(1)
    last_match = None
    for match in _CHOICE_FALLBACK.finditer(response):
        last_match = match
    if last_match:
        return last_match.group(1)
    return None

def _get_verify_func(gold_is_latex: bool):
    if gold_is_latex not in _VERIFY_FUNC_CACHE:
        logger.debug("Creating math_metric verify function (gold_is_latex=%s)", gold_is_latex)
        _VERIFY_FUNC_CACHE[gold_is_latex] = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),) if gold_is_latex else (ExprExtractionConfig(), LatexExtractionConfig()),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            aggregation_function=max,
            precision=6
        )
    return _VERIFY_FUNC_CACHE[gold_is_latex]


def compute_score(data_name: str, responses: List[str], answer: str, gold_is_latex: bool) -> Tuple[List[float], List[str | None]]:
    scores = []
    extracted_answers = []
    if data_name in ['gpqa_diamond', 'mmlu_stem', 'gpqa', 'mmlu', 'mmlu_pro', 'mmlu_stem_500']:
        for resp in responses:
            extracted_answer = _extract_choice_answer(resp)
            extracted_answers.append(extracted_answer)
            if extracted_answer is not None and extracted_answer == answer:
                scores.append(1.0)
            else:
                scores.append(0.0)
    elif data_name in ['math500', 'aime24', 'amc23', 'math_500', 'aime25']:
        verify_func = _get_verify_func(gold_is_latex)
        for response in responses:
            try:
                grade, extracted_answer = verify_func([answer], [response])
                if grade != 1:
                    grade, extracted_answer = verify_func([response], [answer])
            except Exception as exc:
                grade = 0
                extracted_answer = None
            extracted_answer = extracted_answer[0] if extracted_answer else None
            extracted_answers.append(extracted_answer)
            scores.append(float(grade))
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    return scores, extracted_answers

def cal_maj_k(data_name: str, resps: List[str], answer: str, gold_is_latex: bool):
    """Calculate maj@k by finding the most frequent answer among k candidates."""
    if len(resps) <= 1:
        scores, _ = compute_score(data_name, resps, answer, gold_is_latex)
        return scores[0]
    if data_name in _MULTI_CHOICE_DATASETS:
        extracted = [_extract_choice_answer(resp) for resp in resps]
        counts = {}
        for ans in extracted:
            if ans is None:
                continue
            counts[ans] = counts.get(ans, 0) + 1
        if not counts:
            return 0.0
        max_count = max(counts.values())
        majority_answer = None
        for ans in extracted:
            if ans is not None and counts.get(ans) == max_count:
                majority_answer = ans
                break
        return 1.0 if majority_answer == answer else 0.0
    
    # Create equal matrix to compare all responses
    equal_matrix = [[False] * len(resps) for _ in range(len(resps))]
    
    for i in range(len(resps)):
        for j in range(i, len(resps)):
            if i == j:
                equal_matrix[i][j] = True
                equal_matrix[j][i] = True
            else:
                scores, _ = compute_score(data_name, [resps[i]], resps[j], gold_is_latex)
                grade = scores[0]
                is_equal = grade == 1.0
                equal_matrix[i][j] = is_equal
                equal_matrix[j][i] = is_equal
    
    # Count how many responses are equivalent to each response
    maj_count = [sum(equal_matrix[i]) for i in range(len(resps))]
    
    # Find the response with the highest count (majority)
    majority_idx = maj_count.index(max(maj_count))
    
    maj_answer = resps[majority_idx]
    scores, _ = compute_score(data_name, [maj_answer], answer, gold_is_latex)
    return scores[0]


def process_answers(input_list: List[Dict], data_name: str, gold_is_latex: bool) -> Tuple[List[Dict], Dict[str, float]]:
    """Process each answer through extraction workflow and compare with gold answers."""

    for idx, entry in enumerate(input_list):
        print(f'Processing {idx}...')
        resps = entry.get('resps', [])
        answer = entry.get('answer')
        if not resps or answer is None:
            logger.warning("Skipping entry with missing responses or answer: %s", entry.get('id', '<no-id>'))
            entry.update({'scores': [], 'extracted_answers': [], 'avg_k': 0.0, 'std_k': 0.0, 'maj_k': 0.0, 'pass_k': 0.0})
            continue

        scores, extracted_answers = compute_score(data_name, resps, answer, gold_is_latex)
        avg_k = sum(scores) / len(scores)
        std_k = (sum((x - avg_k) ** 2 for x in scores) / len(scores)) ** 0.5
        maj_k = cal_maj_k(data_name, resps, answer, gold_is_latex)
        pass_k = 1.0 if any(score > 0 for score in scores) else 0.0

        entry.update({
            'scores': scores,
            'extracted_answers': extracted_answers,
            'avg_k': avg_k,
            'std_k': std_k,
            'maj_k': maj_k,
            'pass_k': pass_k,
        })

    dataset_size = max(len(input_list), 1)
    acc_list = []
    k = len(input_list[0].get('scores', []))
    for i in range(k):
        correct_count = sum(1 for entry in input_list if entry.get('scores', []) and entry['scores'][i] > 0)
        acc = correct_count / dataset_size
        acc_list.append(acc)

    result_json = {
        "avg_k": sum(entry['avg_k'] for entry in input_list) / dataset_size,
        "std_k": sum(entry['std_k'] for entry in input_list) / dataset_size,
        "maj_k": sum(entry['maj_k'] for entry in input_list) / dataset_size,
        "pass_k": sum(entry['pass_k'] for entry in input_list) / dataset_size,
        "acc_list": acc_list,
    }
    return input_list, result_json
    
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    data_names = [name.strip() for name in args.data_names.split(',') if name.strip()]
    for data_name in data_names:
        input_path = args.input_template.format(data_name=data_name)
        if not os.path.exists(input_path):
            logger.warning("Input file not found for %s: %s", data_name, input_path)
            continue
        logger.info("Evaluating dataset: %s (input=%s)", data_name, input_path)

        input_list = list(load_jsonl(input_path))
        logger.info("Loaded %d input records", len(input_list))
        
        output_list, result_json = process_answers(input_list, data_name, args.gold_is_latex)
        logger.info("Aggregate metrics: %s", result_json)

        output_dir = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_jsonl = os.path.join(output_dir, f"{base}_scores.jsonl")
        save_jsonl(output_list, output_jsonl)
        metrics_path = output_jsonl.replace(".jsonl", "_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)
        logger.info("Per-sample output saved to %s", output_jsonl)
        logger.info("Metrics saved to %s", metrics_path)

if __name__ == "__main__":
    main()

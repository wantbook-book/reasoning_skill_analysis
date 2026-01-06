import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Union, Any
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import torch
import numpy as np

def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable formats.
    """
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def load_data(source: str, use_arrow: bool = False) -> List[Dict]:
    """
    Load data.

    Args:
        source: data source (file path or directory path)
        use_arrow: whether to load from Arrow format

    Returns:
        List of data records
    """
    if use_arrow:
        print(f"Loading dataset from Arrow format: {source}")

        # Try to fix legacy Arrow dataset metadata
        dataset_info_path = Path(source) / "dataset_info.json"
        if dataset_info_path.exists():
            try:
                # Read and repair dataset_info.json
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    info_dict = json.load(f)

                # Check and fix List types in features
                if 'features' in info_dict:
                    info_str = json.dumps(info_dict['features'])
                    if '"_type": "List"' in info_str or '"_type":"List"' in info_str:
                        print("Detected legacy List feature types; repairing...")
                        # Replace List with Sequence
                        info_str = info_str.replace('"_type": "List"', '"_type": "Sequence"')
                        info_str = info_str.replace('"_type":"List"', '"_type":"Sequence"')
                        info_dict['features'] = json.loads(info_str)

                        # Backup original file
                        backup_path = dataset_info_path.with_suffix('.json.backup')
                        if not backup_path.exists():
                            import shutil
                            shutil.copy2(dataset_info_path, backup_path)
                            print(f"Backed up original file to: {backup_path}")

                        # Write back repaired file
                        with open(dataset_info_path, 'w', encoding='utf-8') as f:
                            json.dump(info_dict, f, indent=2)
                        print("✓ Repaired dataset_info.json")
            except Exception as e:
                print(f"Warning: error while repairing dataset_info.json: {e}")
                print("Will try to load directly...")

        # Load dataset
        try:
            dataset = load_from_disk(source)
        except ValueError as e:
            if "Feature type 'List' not found" in str(e):
                print("\nError: Arrow dataset uses incompatible 'List' feature type")
                print("Solutions:")
                print("1. Edit dataset_info.json to replace all '\"_type\": \"List\"' with '\"_type\": \"Sequence\"'")
                print("2. Or use a JSONL dataset instead (set DATASET_ARROW=false)")
                raise
            else:
                raise

        # Convert to list and ensure JSON-serializable data
        data = []
        for item in tqdm(dataset, desc=f"Loading {Path(source).name}"):
            # Convert to dict and handle tensor types
            item_dict = dict(item)
            item_dict = convert_to_json_serializable(item_dict)
            data.append(item_dict)
        print(f"Loaded {len(data):,} records")
    else:
        print(f"Loading from JSONL: {source}")
        data = []
        with open(source, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                data.append(json.loads(line.strip()))
        print(f"Loaded {len(data):,} records")

    return data

def count_tokens(item: Dict) -> int:
    """Count tokens for a single record."""
    input_ids = item.get('input_ids', [])
    # Handle possible tensor types
    if isinstance(input_ids, (torch.Tensor, np.ndarray)):
        return len(input_ids)
    return len(input_ids)

def get_dataset_stats(data: List[Dict]) -> Dict:
    """Get dataset statistics."""
    if not data:
        return {
            'num_samples': 0,
            'total_tokens': 0,
            'avg_tokens': 0,
            'min_tokens': 0,
            'max_tokens': 0,
            'median_tokens': 0
        }
    
    token_counts = [count_tokens(item) for item in data]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(data)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    # Compute median
    sorted_counts = sorted(token_counts)
    n = len(sorted_counts)
    if n % 2 == 0:
        median_tokens = (sorted_counts[n//2-1] + sorted_counts[n//2]) / 2
    else:
        median_tokens = sorted_counts[n//2]
    
    return {
        'num_samples': len(data),
        'total_tokens': total_tokens,
        'avg_tokens': avg_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'median_tokens': median_tokens
    }

def format_number(num: Union[int, float]) -> str:
    """Format numbers with thousands separators."""
    if isinstance(num, float):
        return f"{num:,.2f}"
    return f"{num:,}"

def format_tokens_human(tokens: int) -> str:
    """Convert token count to a human-readable format."""
    if tokens >= 1_000_000_000:
        return f"{tokens/1_000_000_000:.2f}B"
    elif tokens >= 1_000_000:
        return f"{tokens/1_000_000:.2f}M"
    elif tokens >= 1_000:
        return f"{tokens/1_000:.2f}K"
    else:
        return str(tokens)

def write_stats_to_txt(stats_info: Dict, output_path: str):
    """Write stats info to a txt file."""
    txt_path = output_path.replace('.jsonl', '_stats.txt')
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Dataset Sampling Report\n")
        f.write("="*70 + "\n\n")
        
        # Config
        f.write("[Config]\n")
        f.write("-"*70 + "\n")
        config = stats_info['config']
        f.write(f"Total target tokens:     {format_number(config['total_target_tokens'])} ({format_tokens_human(config['total_target_tokens'])})\n")
        f.write(f"Dataset1 target tokens:  {format_number(config['dataset1_target_tokens'])} ({format_tokens_human(config['dataset1_target_tokens'])})\n")
        f.write(f"Dataset2 target tokens:  {format_number(config['dataset2_target_tokens'])} ({format_tokens_human(config['dataset2_target_tokens'])})\n")
        f.write(f"Dataset1 ratio:          {config['ratio']*100:.1f}%\n")
        f.write(f"Dataset2 ratio:          {(1-config['ratio'])*100:.1f}%\n")
        f.write(f"Random seed:             {config['seed']}\n")
        f.write("\n")
        
        # Dataset 1 stats
        f.write("[Dataset 1]\n")
        f.write("-"*70 + "\n")
        ds1 = stats_info['dataset1']
        f.write(f"Source: {ds1.get('path', 'N/A')}\n")
        f.write(f"Format: {ds1.get('format', 'N/A')}\n\n")
        
        f.write("Original dataset:\n")
        orig1 = ds1['original']
        f.write(f"  Samples:        {format_number(orig1['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(orig1['total_tokens'])} ({format_tokens_human(orig1['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(orig1['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(orig1['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(orig1['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(orig1['max_tokens'])}\n\n")
        
        f.write("Sampled dataset:\n")
        samp1 = ds1['sampled']
        f.write(f"  Samples:        {format_number(samp1['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(samp1['total_tokens'])} ({format_tokens_human(samp1['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(samp1['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(samp1['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(samp1['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(samp1['max_tokens'])}\n")
        f.write(f"  Sampling ratio: {samp1['num_samples']/orig1['num_samples']*100:.2f}%\n")
        f.write(f"  Target hit rate:{samp1['total_tokens']/config['dataset1_target_tokens']*100:.2f}%\n")
        f.write("\n")
        
        # Dataset 2 stats
        f.write("[Dataset 2]\n")
        f.write("-"*70 + "\n")
        ds2 = stats_info['dataset2']
        f.write(f"Source: {ds2.get('path', 'N/A')}\n")
        f.write(f"Format: {ds2.get('format', 'N/A')}\n\n")
        
        f.write("Original dataset:\n")
        orig2 = ds2['original']
        f.write(f"  Samples:        {format_number(orig2['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(orig2['total_tokens'])} ({format_tokens_human(orig2['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(orig2['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(orig2['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(orig2['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(orig2['max_tokens'])}\n\n")
        
        f.write("Sampled dataset:\n")
        samp2 = ds2['sampled']
        f.write(f"  Samples:        {format_number(samp2['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(samp2['total_tokens'])} ({format_tokens_human(samp2['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(samp2['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(samp2['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(samp2['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(samp2['max_tokens'])}\n")
        f.write(f"  Sampling ratio: {samp2['num_samples']/orig2['num_samples']*100:.2f}%\n")
        f.write(f"  Target hit rate:{samp2['total_tokens']/config['dataset2_target_tokens']*100:.2f}%\n")
        f.write("\n")
        
        # Combined stats
        f.write("[Combined Dataset]\n")
        f.write("-"*70 + "\n")
        combined = stats_info['combined']
        f.write(f"Total samples:  {format_number(combined['num_samples'])}\n")
        f.write(f"Total tokens:   {format_number(combined['total_tokens'])} ({format_tokens_human(combined['total_tokens'])})\n")
        f.write(f"Avg tokens:     {format_number(combined['avg_tokens'])}\n")
        f.write(f"Median tokens:  {format_number(combined['median_tokens'])}\n")
        f.write(f"Min tokens:     {format_number(combined['min_tokens'])}\n")
        f.write(f"Max tokens:     {format_number(combined['max_tokens'])}\n\n")
        
        f.write("Dataset 1 contribution:\n")
        f.write(f"  Samples:        {format_number(samp1['num_samples'])} ({samp1['num_samples']/combined['num_samples']*100:.2f}%)\n")
        f.write(f"  Tokens:         {format_number(samp1['total_tokens'])} ({samp1['total_tokens']/combined['total_tokens']*100:.2f}%)\n\n")
        
        f.write("Dataset 2 contribution:\n")
        f.write(f"  Samples:        {format_number(samp2['num_samples'])} ({samp2['num_samples']/combined['num_samples']*100:.2f}%)\n")
        f.write(f"  Tokens:         {format_number(samp2['total_tokens'])} ({samp2['total_tokens']/combined['total_tokens']*100:.2f}%)\n\n")
        
        f.write("Target completion:\n")
        f.write(f"  Overall completion: {combined['total_tokens']/config['total_target_tokens']*100:.2f}%\n")
        f.write(f"  Dataset1 completion:{samp1['total_tokens']/config['dataset1_target_tokens']*100:.2f}%\n")
        f.write(f"  Dataset2 completion:{samp2['total_tokens']/config['dataset2_target_tokens']*100:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("End of report\n")
        f.write("="*70 + "\n")
    
    print(f"Stats (TXT) saved to: {txt_path}")
    return txt_path

def sample_by_token_budget(
    data: List[Dict],
    target_tokens: int,
    shuffle: bool = True,
    seed: int = 42,
    return_remaining: bool = False
) -> Union[List[Dict], tuple[List[Dict], List[Dict]]]:
    """
    Sample data by a target token budget.

    Args:
        data: input data
        target_tokens: target token count
        shuffle: whether to shuffle data
        seed: random seed
        return_remaining: whether to return remaining data

    Returns:
        If return_remaining=False: sampled data
        If return_remaining=True: (sampled data, remaining data)
    """
    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)

    sampled_data = []
    current_tokens = 0
    sampled_indices = set()

    for idx, item in enumerate(tqdm(data, desc="Sampling")):
        item_tokens = count_tokens(item)

        # If adding this item exceeds budget, check proximity to target
        if current_tokens + item_tokens > target_tokens:
            # If we've reached at least 95% of target, stop
            if current_tokens >= target_tokens * 0.95:
                break
            # Otherwise keep going (allow slight overflow)
            if current_tokens + item_tokens > target_tokens * 1.05:
                break

        sampled_data.append(item)
        sampled_indices.add(idx)
        current_tokens += item_tokens

        # Stop if target reached
        if current_tokens >= target_tokens:
            break

    if return_remaining:
        # Return remaining data
        remaining_data = [item for idx, item in enumerate(data) if idx not in sampled_indices]
        return sampled_data, remaining_data
    else:
        return sampled_data


def combine_and_save_datasets(
    dataset1_source: str,
    dataset2_source: str,
    output_path: str,
    total_tokens: int = 1_000_000_000,  # 1B tokens
    ratio: float = 0.5,  # dataset1 ratio
    shuffle_final: bool = True,
    seed: int = 42,
    dataset1_use_arrow: bool = False,
    dataset2_use_arrow: bool = False,
    eval_tokens: int = 0,  # eval token budget
    eval_output_path: str = None  # eval output path
):
    """
    Sample from two datasets and merge.

    Args:
        dataset1_source: path to dataset 1
        dataset2_source: path to dataset 2
        output_path: output path (file or directory)
        total_tokens: total token budget (default 1B)
        ratio: dataset1 token ratio (default 0.5)
        shuffle_final: whether to shuffle the final merged dataset
        seed: random seed
        dataset1_use_arrow: whether dataset1 uses Arrow format
        dataset2_use_arrow: whether dataset2 uses Arrow format
        eval_tokens: eval token budget sampled from remaining data (default 0 = none)
        eval_output_path: eval output path
    """
    
    print("="*60)
    print("Sampling and merging by token budget from two datasets")
    print("="*60)
    
    # Compute target tokens per dataset
    dataset1_tokens = int(total_tokens * ratio)
    dataset2_tokens = int(total_tokens * (1 - ratio))
    
    print("\nTarget config:")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Dataset1 target: {dataset1_tokens:,} ({dataset1_tokens/1e9:.2f}B)")
    print(f"  Dataset2 target: {dataset2_tokens:,} ({dataset2_tokens/1e9:.2f}B)")
    print(f"  Random seed: {seed}")
    
    # Load dataset 1
    print(f"\n{'='*60}")
    print(f"Loading dataset1: {dataset1_source}")
    print(f"Format: {'Arrow' if dataset1_use_arrow else 'JSONL'}")
    data1 = load_data(dataset1_source, use_arrow=dataset1_use_arrow)
    stats1 = get_dataset_stats(data1)
    print("Dataset1 stats:")
    print(f"  Samples: {stats1['num_samples']:,}")
    print(f"  Total tokens: {stats1['total_tokens']:,} ({stats1['total_tokens']/1e9:.2f}B)")
    print(f"  Avg tokens/sample: {stats1['avg_tokens']:.2f}")
    print(f"  Median tokens/sample: {stats1['median_tokens']:.2f}")
    print(f"  Token range: [{stats1['min_tokens']}, {stats1['max_tokens']}]")
    
    # Load dataset 2
    print(f"\n{'='*60}")
    print(f"Loading dataset2: {dataset2_source}")
    print(f"Format: {'Arrow' if dataset2_use_arrow else 'JSONL'}")
    data2 = load_data(dataset2_source, use_arrow=dataset2_use_arrow)
    stats2 = get_dataset_stats(data2)
    print("Dataset2 stats:")
    print(f"  Samples: {stats2['num_samples']:,}")
    print(f"  Total tokens: {stats2['total_tokens']:,} ({stats2['total_tokens']/1e9:.2f}B)")
    print(f"  Avg tokens/sample: {stats2['avg_tokens']:.2f}")
    print(f"  Median tokens/sample: {stats2['median_tokens']:.2f}")
    print(f"  Token range: [{stats2['min_tokens']}, {stats2['max_tokens']}]")
    
    # Check dataset sufficiency
    if stats1['total_tokens'] < dataset1_tokens:
        print("\nWarning: dataset1 total tokens below target!")
        print(f"  Available: {stats1['total_tokens']:,}, needed: {dataset1_tokens:,}")
        dataset1_tokens = stats1['total_tokens']
    
    if stats2['total_tokens'] < dataset2_tokens:
        print("\nWarning: dataset2 total tokens below target!")
        print(f"  Available: {stats2['total_tokens']:,}, needed: {dataset2_tokens:,}")
        dataset2_tokens = stats2['total_tokens']
    
    # Sample from dataset 1
    print(f"\n{'='*60}")
    print("Sampling dataset1...")
    need_remaining = eval_tokens > 0 and eval_output_path is not None

    if need_remaining:
        sampled_data1, remaining_data1 = sample_by_token_budget(
            data1, dataset1_tokens, shuffle=True, seed=seed, return_remaining=True
        )
        remaining_stats1 = get_dataset_stats(remaining_data1)
        print(f"Remaining data1: {remaining_stats1['num_samples']:,} samples, {remaining_stats1['total_tokens']:,} tokens")
    else:
        sampled_data1 = sample_by_token_budget(data1, dataset1_tokens, shuffle=True, seed=seed)
        remaining_data1 = []

    sampled_stats1 = get_dataset_stats(sampled_data1)
    print("Sampling results:")
    print(f"  Samples: {sampled_stats1['num_samples']:,}")
    print(f"  Total tokens: {sampled_stats1['total_tokens']:,} ({sampled_stats1['total_tokens']/1e9:.2f}B)")
    print(f"  Completion: {sampled_stats1['total_tokens']/dataset1_tokens*100:.2f}%")

    # Free memory
    del data1

    # Sample from dataset 2
    print(f"\n{'='*60}")
    print("Sampling dataset2...")

    if need_remaining:
        sampled_data2, remaining_data2 = sample_by_token_budget(
            data2, dataset2_tokens, shuffle=True, seed=seed, return_remaining=True
        )
        remaining_stats2 = get_dataset_stats(remaining_data2)
        print(f"Remaining data2: {remaining_stats2['num_samples']:,} samples, {remaining_stats2['total_tokens']:,} tokens")
    else:
        sampled_data2 = sample_by_token_budget(data2, dataset2_tokens, shuffle=True, seed=seed)
        remaining_data2 = []

    sampled_stats2 = get_dataset_stats(sampled_data2)
    print("Sampling results:")
    print(f"  Samples: {sampled_stats2['num_samples']:,}")
    print(f"  Total tokens: {sampled_stats2['total_tokens']:,} ({sampled_stats2['total_tokens']/1e9:.2f}B)")
    print(f"  Completion: {sampled_stats2['total_tokens']/dataset2_tokens*100:.2f}%")

    # Free memory
    del data2
    
    # Merge datasets
    print(f"\n{'='*60}")
    print("Merging datasets...")
    
    # Add source tags
    for item in sampled_data1:
        item['source'] = 'dataset1'
    for item in sampled_data2:
        item['source'] = 'dataset2'
    
    combined_data = sampled_data1 + sampled_data2
    
    # Shuffle merged data
    if shuffle_final:
        print("Shuffling merged data...")
        random.seed(seed)
        random.shuffle(combined_data)
    
    # Stats for merged data
    combined_stats = get_dataset_stats(combined_data)
    print("\nMerged stats:")
    print(f"  Total samples: {combined_stats['num_samples']:,}")
    print(f"  Total tokens: {combined_stats['total_tokens']:,} ({combined_stats['total_tokens']/1e9:.2f}B)")
    print(f"  Avg tokens/sample: {combined_stats['avg_tokens']:.2f}")
    print(f"  Median tokens/sample: {combined_stats['median_tokens']:.2f}")
    print(f"  Dataset1 samples: {sampled_stats1['num_samples']:,} ({sampled_stats1['num_samples']/combined_stats['num_samples']*100:.2f}%)")
    print(f"  Dataset2 samples: {sampled_stats2['num_samples']:,} ({sampled_stats2['num_samples']/combined_stats['num_samples']*100:.2f}%)")
    print(f"  Dataset1 tokens: {sampled_stats1['total_tokens']:,} ({sampled_stats1['total_tokens']/combined_stats['total_tokens']*100:.2f}%)")
    print(f"  Dataset2 tokens: {sampled_stats2['total_tokens']:,} ({sampled_stats2['total_tokens']/combined_stats['total_tokens']*100:.2f}%)")
    
    # Handle output path
    print(f"\n{'='*60}")
    output_path_obj = Path(output_path)
    
    # If directory or no extension, create default filename
    if output_path_obj.is_dir() or not output_path_obj.suffix:
        # Ensure directory exists
        output_path_obj.mkdir(parents=True, exist_ok=True)
        # Create default filename
        base_filename = f"combined_{format_tokens_human(total_tokens).lower()}_ratio{int(ratio*100)}"
        output_file = output_path_obj / f"{base_filename}.jsonl"
    else:
        # Ensure parent directory exists
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path_obj
    
    print(f"Saving to: {output_file}")
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(combined_data, desc="Saving data"):
            # Ensure data is JSON-serializable
            serializable_item = convert_to_json_serializable(item)
            f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(combined_data):,} records")
    
    # Base output name (without .jsonl)
    output_base = str(output_file).replace('.jsonl', '')
    
    # Save one sample
    sample_path = f"{output_base}_sample.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        sample_item = convert_to_json_serializable(combined_data[0])
        json.dump(sample_item, f, ensure_ascii=False, indent=2)
    print(f"Sample data saved to: {sample_path}")
    
    # Save stats (JSON)
    stats_json_path = f"{output_base}_stats.json"
    stats_info = {
        'config': {
            'total_target_tokens': total_tokens,
            'dataset1_target_tokens': dataset1_tokens,
            'dataset2_target_tokens': dataset2_tokens,
            'ratio': ratio,
            'seed': seed
        },
        'dataset1': {
            'path': dataset1_source,
            'format': 'Arrow' if dataset1_use_arrow else 'JSONL',
            'original': stats1,
            'sampled': sampled_stats1
        },
        'dataset2': {
            'path': dataset2_source,
            'format': 'Arrow' if dataset2_use_arrow else 'JSONL',
            'original': stats2,
            'sampled': sampled_stats2
        },
        'combined': combined_stats
    }
    
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_info, f, ensure_ascii=False, indent=2)
    print(f"Stats (JSON) saved to: {stats_json_path}")
    
    # Save stats to TXT
    stats_txt_path = f"{output_base}_stats.txt"
    write_stats_to_txt_direct(stats_info, stats_txt_path)
    print(f"Stats (TXT) saved to: {stats_txt_path}")

    # Eval dataset: sample from remaining data using the same ratio
    eval_file = None
    if eval_tokens > 0 and eval_output_path is not None:
        print(f"\n{'='*60}")
        print("Sampling evaluation dataset from remaining data by ratio...")
        print(f"{'='*60}")

        # Compute eval token targets per dataset (same ratio as training)
        eval_dataset1_tokens = int(eval_tokens * ratio)
        eval_dataset2_tokens = int(eval_tokens * (1 - ratio))

        print("\nEval target config:")
        print(f"  Total eval tokens: {eval_tokens:,} ({eval_tokens/1e9:.2f}B)")
        print(f"  Dataset1 target: {eval_dataset1_tokens:,} ({eval_dataset1_tokens/1e9:.2f}B)")
        print(f"  Dataset2 target: {eval_dataset2_tokens:,} ({eval_dataset2_tokens/1e9:.2f}B)")

        # Stats for remaining data
        if need_remaining:
            remaining_stats1 = get_dataset_stats(remaining_data1)
            remaining_stats2 = get_dataset_stats(remaining_data2)

            print("\nRemaining data stats:")
            print(f"  Dataset1 remaining: {remaining_stats1['num_samples']:,} samples, {remaining_stats1['total_tokens']:,} tokens ({remaining_stats1['total_tokens']/1e9:.2f}B)")
            print(f"  Dataset2 remaining: {remaining_stats2['num_samples']:,} samples, {remaining_stats2['total_tokens']:,} tokens ({remaining_stats2['total_tokens']/1e9:.2f}B)")

            # Check remaining data sufficiency
            if remaining_stats1['total_tokens'] < eval_dataset1_tokens:
                print("\nWarning: dataset1 remaining tokens below eval target!")
                print(f"  Available: {remaining_stats1['total_tokens']:,}, needed: {eval_dataset1_tokens:,}")
                eval_dataset1_tokens = remaining_stats1['total_tokens']

            if remaining_stats2['total_tokens'] < eval_dataset2_tokens:
                print("\nWarning: dataset2 remaining tokens below eval target!")
                print(f"  Available: {remaining_stats2['total_tokens']:,}, needed: {eval_dataset2_tokens:,}")
                eval_dataset2_tokens = remaining_stats2['total_tokens']

            # Sample eval data from dataset1 remaining
            print("\nSampling eval data from dataset1 remaining...")
            eval_data1 = sample_by_token_budget(remaining_data1, eval_dataset1_tokens, shuffle=True, seed=seed)
            eval_stats1 = get_dataset_stats(eval_data1)
            print(f"  Samples: {eval_stats1['num_samples']:,}")
            print(f"  Total tokens: {eval_stats1['total_tokens']:,} ({eval_stats1['total_tokens']/1e9:.2f}B)")
            print(f"  Completion: {eval_stats1['total_tokens']/eval_dataset1_tokens*100:.2f}%")

            # Sample eval data from dataset2 remaining
            print("\nSampling eval data from dataset2 remaining...")
            eval_data2 = sample_by_token_budget(remaining_data2, eval_dataset2_tokens, shuffle=True, seed=seed)
            eval_stats2 = get_dataset_stats(eval_data2)
            print(f"  Samples: {eval_stats2['num_samples']:,}")
            print(f"  Total tokens: {eval_stats2['total_tokens']:,} ({eval_stats2['total_tokens']/1e9:.2f}B)")
            print(f"  Completion: {eval_stats2['total_tokens']/eval_dataset2_tokens*100 if eval_dataset2_tokens>0 else 0:.2f}%")

            # Add source tags
            for item in eval_data1:
                item['source'] = 'dataset1'
            for item in eval_data2:
                item['source'] = 'dataset2'

            # Merge eval data
            eval_combined = eval_data1 + eval_data2

            # Shuffle eval data
            if shuffle_final:
                print("\nShuffling eval data...")
                random.seed(seed + 1)  # Use a different seed
                random.shuffle(eval_combined)

            eval_combined_stats = get_dataset_stats(eval_combined)
            print("\nEval merged results:")
            print(f"  Total samples: {eval_combined_stats['num_samples']:,}")
            print(f"  Total tokens: {eval_combined_stats['total_tokens']:,} ({eval_combined_stats['total_tokens']/1e9:.2f}B)")
            print(f"  Avg tokens/sample: {eval_combined_stats['avg_tokens']:.2f}")
            print(f"  Dataset1 samples: {eval_stats1['num_samples']:,} ({eval_stats1['num_samples']/eval_combined_stats['num_samples']*100:.2f}%)")
            print(f"  Dataset2 samples: {eval_stats2['num_samples']:,} ({eval_stats2['num_samples']/eval_combined_stats['num_samples']*100:.2f}%)")
            print(f"  Dataset1 tokens: {eval_stats1['total_tokens']:,} ({eval_stats1['total_tokens']/eval_combined_stats['total_tokens']*100:.2f}%)")
            print(f"  Dataset2 tokens: {eval_stats2['total_tokens']:,} ({eval_stats2['total_tokens']/eval_combined_stats['total_tokens']*100:.2f}%)")

            # Handle eval output path
            eval_output_path_obj = Path(eval_output_path)
            if eval_output_path_obj.is_dir() or not eval_output_path_obj.suffix:
                eval_output_path_obj.mkdir(parents=True, exist_ok=True)
                eval_filename = f"eval_{format_tokens_human(eval_tokens).lower()}_ratio{int(ratio*100)}.jsonl"
                eval_file = eval_output_path_obj / eval_filename
            else:
                eval_output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                eval_file = eval_output_path_obj

            print(f"\nSaving eval data to: {eval_file}")

            # Save eval data
            with open(eval_file, 'w', encoding='utf-8') as f:
                for item in tqdm(eval_combined, desc="Saving eval data"):
                    serializable_item = convert_to_json_serializable(item)
                    f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')

            print(f"Saved {len(eval_combined):,} eval records")

            # Save eval stats
            eval_base = str(eval_file).replace('.jsonl', '')
            eval_stats_json = f"{eval_base}_stats.json"
            eval_stats_info = {
                'source': 'remaining_data_from_training_split',
                'training_config': {
                    'total_train_tokens': total_tokens,
                    'dataset1_train_tokens': dataset1_tokens,
                    'dataset2_train_tokens': dataset2_tokens,
                    'ratio': ratio,
                },
                'eval_config': {
                    'total_eval_tokens': eval_tokens,
                    'dataset1_eval_tokens': eval_dataset1_tokens,
                    'dataset2_eval_tokens': eval_dataset2_tokens,
                    'ratio': ratio,
                    'seed': seed,
                },
                'remaining_data': {
                    'dataset1': remaining_stats1,
                    'dataset2': remaining_stats2,
                },
                'eval_sampled': {
                    'dataset1': eval_stats1,
                    'dataset2': eval_stats2,
                },
                'eval_combined': eval_combined_stats,
            }

            with open(eval_stats_json, 'w', encoding='utf-8') as f:
                json.dump(eval_stats_info, f, ensure_ascii=False, indent=2)
            print(f"Eval stats saved to: {eval_stats_json}")

            # Save one eval sample
            eval_sample_path = f"{eval_base}_sample.json"
            with open(eval_sample_path, 'w', encoding='utf-8') as f:
                eval_sample_item = convert_to_json_serializable(eval_combined[0])
                json.dump(eval_sample_item, f, ensure_ascii=False, indent=2)
            print(f"Eval sample saved to: {eval_sample_path}")
        else:
            print("\nError: no remaining data kept, cannot create eval dataset")
            print("Make sure eval_tokens > 0 and eval_output_path is not None")

    print(f"\n{'='*60}")
    print("Processing complete!")
    print("="*60)
    print("\nGenerated files:")
    print("[Training data]")
    print(f"  - Data file: {output_file}")
    print(f"  - Sample file: {sample_path}")
    print(f"  - Stats (JSON): {stats_json_path}")
    print(f"  - Stats (TXT): {stats_txt_path}")

    if eval_file:
        print("\n[Eval data]")
        print(f"  - Data file: {eval_file}")
        print(f"  - Stats (JSON): {eval_stats_json}")
        print(f"  - Sample file: {eval_sample_path}")

    return combined_data


def write_stats_to_txt_direct(stats_info: Dict, txt_path: str):
    """Write stats directly to the specified txt path."""
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Dataset Sampling Report\n")
        f.write("="*70 + "\n\n")
        
        # Config
        f.write("[Config]\n")
        f.write("-"*70 + "\n")
        config = stats_info['config']
        f.write(f"Total target tokens:     {format_number(config['total_target_tokens'])} ({format_tokens_human(config['total_target_tokens'])})\n")
        f.write(f"Dataset1 target tokens:  {format_number(config['dataset1_target_tokens'])} ({format_tokens_human(config['dataset1_target_tokens'])})\n")
        f.write(f"Dataset2 target tokens:  {format_number(config['dataset2_target_tokens'])} ({format_tokens_human(config['dataset2_target_tokens'])})\n")
        f.write(f"Dataset1 ratio:          {config['ratio']*100:.1f}%\n")
        f.write(f"Dataset2 ratio:          {(1-config['ratio'])*100:.1f}%\n")
        f.write(f"Random seed:             {config['seed']}\n")
        f.write("\n")
        
        # Dataset 1 stats
        f.write("[Dataset 1]\n")
        f.write("-"*70 + "\n")
        ds1 = stats_info['dataset1']
        f.write(f"Source: {ds1.get('path', 'N/A')}\n")
        f.write(f"Format: {ds1.get('format', 'N/A')}\n\n")
        
        f.write("Original dataset:\n")
        orig1 = ds1['original']
        f.write(f"  Samples:        {format_number(orig1['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(orig1['total_tokens'])} ({format_tokens_human(orig1['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(orig1['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(orig1['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(orig1['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(orig1['max_tokens'])}\n\n")
        
        f.write("Sampled dataset:\n")
        samp1 = ds1['sampled']
        f.write(f"  Samples:        {format_number(samp1['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(samp1['total_tokens'])} ({format_tokens_human(samp1['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(samp1['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(samp1['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(samp1['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(samp1['max_tokens'])}\n")
        f.write(f"  Sampling ratio: {samp1['num_samples']/orig1['num_samples']*100:.2f}%\n")
        f.write(f"  Target hit rate:{samp1['total_tokens']/config['dataset1_target_tokens']*100:.2f}%\n")
        f.write("\n")
        
        # Dataset 2 stats
        f.write("[Dataset 2]\n")
        f.write("-"*70 + "\n")
        ds2 = stats_info['dataset2']
        f.write(f"Source: {ds2.get('path', 'N/A')}\n")
        f.write(f"Format: {ds2.get('format', 'N/A')}\n\n")
        
        f.write("Original dataset:\n")
        orig2 = ds2['original']
        f.write(f"  Samples:        {format_number(orig2['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(orig2['total_tokens'])} ({format_tokens_human(orig2['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(orig2['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(orig2['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(orig2['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(orig2['max_tokens'])}\n\n")
        
        f.write("Sampled dataset:\n")
        samp2 = ds2['sampled']
        f.write(f"  Samples:        {format_number(samp2['num_samples'])}\n")
        f.write(f"  Total tokens:   {format_number(samp2['total_tokens'])} ({format_tokens_human(samp2['total_tokens'])})\n")
        f.write(f"  Avg tokens:     {format_number(samp2['avg_tokens'])}\n")
        f.write(f"  Median tokens:  {format_number(samp2['median_tokens'])}\n")
        f.write(f"  Min tokens:     {format_number(samp2['min_tokens'])}\n")
        f.write(f"  Max tokens:     {format_number(samp2['max_tokens'])}\n")
        f.write(f"  Sampling ratio: {samp2['num_samples']/orig2['num_samples']*100:.2f}%\n")
        f.write(f"  Target hit rate:{samp2['total_tokens']/config['dataset2_target_tokens']*100:.2f}%\n")
        f.write("\n")
        
        # Combined stats
        f.write("[Combined Dataset]\n")
        f.write("-"*70 + "\n")
        combined = stats_info['combined']
        f.write(f"Total samples:  {format_number(combined['num_samples'])}\n")
        f.write(f"Total tokens:   {format_number(combined['total_tokens'])} ({format_tokens_human(combined['total_tokens'])})\n")
        f.write(f"Avg tokens:     {format_number(combined['avg_tokens'])}\n")
        f.write(f"Median tokens:  {format_number(combined['median_tokens'])}\n")
        f.write(f"Min tokens:     {format_number(combined['min_tokens'])}\n")
        f.write(f"Max tokens:     {format_number(combined['max_tokens'])}\n\n")
        
        f.write("Dataset 1 contribution:\n")
        f.write(f"  Samples:        {format_number(samp1['num_samples'])} ({samp1['num_samples']/combined['num_samples']*100:.2f}%)\n")
        f.write(f"  Tokens:         {format_number(samp1['total_tokens'])} ({samp1['total_tokens']/combined['total_tokens']*100:.2f}%)\n\n")
        
        f.write("Dataset 2 contribution:\n")
        f.write(f"  Samples:        {format_number(samp2['num_samples'])} ({samp2['num_samples']/combined['num_samples']*100:.2f}%)\n")
        f.write(f"  Tokens:         {format_number(samp2['total_tokens'])} ({samp2['total_tokens']/combined['total_tokens']*100:.2f}%)\n\n")
        
        f.write("Target completion:\n")
        f.write(f"  Overall completion: {combined['total_tokens']/config['total_target_tokens']*100:.2f}%\n")
        f.write(f"  Dataset1 completion:{samp1['total_tokens']/config['dataset1_target_tokens']*100:.2f}%\n")
        f.write(f"  Dataset2 completion:{samp2['total_tokens']/config['dataset2_target_tokens']*100:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("End of report\n")
        f.write("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Sample and merge by token budget from two datasets (supports Arrow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Sample 0.5B tokens from each JSONL as training data
  python sample_datasets.py \\
      --dataset1 data1.jsonl \\
      --dataset2 data2.jsonl \\
      --output combined_1b.jsonl \\
      --total-tokens-human 1B

  # Generate both training and eval data (sample from remaining data)
  python sample_datasets.py \\
      --dataset1 data1.jsonl \\
      --dataset2 data2.jsonl \\
      --output combined_1b.jsonl \\
      --total-tokens-human 1B \\
      --eval-tokens-human 10M \\
      --eval-output eval_data.jsonl

  # Load and sample from two Arrow directories
  python sample_datasets.py \\
      --dataset1 ./arrow_data1 \\
      --dataset2 ./arrow_data2 \\
      --output combined_1b.jsonl \\
      --total-tokens-human 1B \\
      --dataset1-arrow \\
      --dataset2-arrow

  # Mixed formats: dataset1 Arrow, dataset2 JSONL, plus eval set
  python sample_datasets.py \\
      --dataset1 ./arrow_data \\
      --dataset2 data.jsonl \\
      --output combined.jsonl \\
      --total-tokens-human 2B \\
      --dataset1-arrow \\
      --ratio 0.6 \\
      --eval-tokens-human 50M \\
      --eval-output eval_data.jsonl
        """
    )
    
    parser.add_argument(
        '--dataset1',
        type=str,
        required=True,
        help='Dataset1 path (JSONL file or Arrow directory)'
    )
    
    parser.add_argument(
        '--dataset2',
        type=str,
        required=True,
        help='Dataset2 path (JSONL file or Arrow directory)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (jsonl)'
    )
    
    parser.add_argument(
        '--total-tokens',
        type=int,
        default=None,
        help='Total target token count (e.g., 1000000000 for 1B)'
    )
    
    parser.add_argument(
        '--total-tokens-human',
        type=str,
        default='1B',
        help='Total target tokens, human-readable (e.g., 1B, 1.5B, 500M; default: 1B)'
    )
    
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.5,
        help='Dataset1 token ratio (0-1, default: 0.5)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the final merged dataset'
    )
    
    parser.add_argument(
        '--dataset1-arrow',
        action='store_true',
        help='Dataset1 uses Arrow format (directory with multiple arrow files)'
    )
    
    parser.add_argument(
        '--dataset2-arrow',
        action='store_true',
        help='Dataset2 uses Arrow format (directory with multiple arrow files)'
    )

    parser.add_argument(
        '--eval-tokens',
        type=int,
        default=None,
        help='Eval token budget sampled from remaining data (e.g., 10000000 for 10M)'
    )

    parser.add_argument(
        '--eval-tokens-human',
        type=str,
        default=None,
        help='Eval token budget, human-readable (e.g., 10M, 100M, 1B)'
    )

    parser.add_argument(
        '--eval-output',
        type=str,
        default=None,
        help='Eval dataset output path (file or directory)'
    )

    args = parser.parse_args()
    
    # Parse training token budget
    if args.total_tokens is not None:
        total_tokens = args.total_tokens
    else:
        # Parse human-readable format
        human_str = args.total_tokens_human.upper().strip()
        if human_str.endswith('B'):
            total_tokens = int(float(human_str[:-1]) * 1_000_000_000)
        elif human_str.endswith('M'):
            total_tokens = int(float(human_str[:-1]) * 1_000_000)
        elif human_str.endswith('K'):
            total_tokens = int(float(human_str[:-1]) * 1_000)
        else:
            total_tokens = int(human_str)

    # Parse eval token budget
    eval_tokens = 0
    if args.eval_tokens is not None:
        eval_tokens = args.eval_tokens
    elif args.eval_tokens_human is not None:
        human_str = args.eval_tokens_human.upper().strip()
        if human_str.endswith('B'):
            eval_tokens = int(float(human_str[:-1]) * 1_000_000_000)
        elif human_str.endswith('M'):
            eval_tokens = int(float(human_str[:-1]) * 1_000_000)
        elif human_str.endswith('K'):
            eval_tokens = int(float(human_str[:-1]) * 1_000)
        else:
            eval_tokens = int(human_str)
    
    # Validate arguments
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be between 0 and 1")

    if total_tokens <= 0:
        parser.error("Total token count must be > 0")

    if eval_tokens < 0:
        parser.error("Eval token count must be >= 0")

    # Validate eval config
    if eval_tokens > 0 and args.eval_output is None:
        parser.error("If --eval-tokens or --eval-tokens-human is set, --eval-output is required")

    if args.eval_output is not None and eval_tokens == 0:
        parser.error("If --eval-output is set, --eval-tokens or --eval-tokens-human is required")

    # Check dataset paths
    if not Path(args.dataset1).exists():
        parser.error(f"Dataset1 path does not exist: {args.dataset1}")

    if not Path(args.dataset2).exists():
        parser.error(f"Dataset2 path does not exist: {args.dataset2}")

    # Run sampling and merging
    try:
        combine_and_save_datasets(
            dataset1_source=args.dataset1,
            dataset2_source=args.dataset2,
            output_path=args.output,
            total_tokens=total_tokens,
            ratio=args.ratio,
            shuffle_final=not args.no_shuffle,
            seed=args.seed,
            dataset1_use_arrow=args.dataset1_arrow,
            dataset2_use_arrow=args.dataset2_arrow,
            eval_tokens=eval_tokens,
            eval_output_path=args.eval_output
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

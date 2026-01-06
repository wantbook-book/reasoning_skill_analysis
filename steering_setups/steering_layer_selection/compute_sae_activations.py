#!/usr/bin/env python3
"""
compute_sae_activations.py

Compute SAE activations for a pretokenized dataset.
Supports identifying and separating reasoning-related tokens (single tokens and multi-token sequences).
Optionally saves both hidden states and SAE activations.
"""

import torch
from sae_lens import SAE
from datasets import load_dataset, load_from_disk
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np
import json
import argparse
from pathlib import Path
import h5py
from sae_lens.load_model import load_model
import gc
import psutil
import debugpy
# debugpy.listen(5679)
# debugpy.wait_for_client()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute SAE activations for a pretokenized dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python compute_sae_activations.py \\
      --sae-path /path/to/sae \\
      --dataset /path/to/dataset.jsonl \\
      --output ./output

  # Separate reasoning tokens (save separately)
  python compute_sae_activations.py \\
      --sae-path /path/to/sae \\
      --dataset /path/to/dataset.jsonl \\
      --output ./output \\
      --reasoning-tokens reasoning_tokens.json \\
      --separate-reasoning \\
      --save-token-masks \\
      --save-hidden-states

  # Save only reasoning tokens
  python compute_sae_activations.py \\
      --sae-path /path/to/sae \\
      --dataset /path/to/dataset.jsonl \\
      --output ./output \\
      --reasoning-tokens reasoning_tokens.json \\
      --only-reasoning \\
      --save-hidden-states
        """
    )
    
    # Required parameters
    parser.add_argument(
        '--sae-path',
        type=str,
        required=True,
        help='Local path to the SAE model.'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset path (JSONL file or Arrow directory).'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path.'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Language model path (defaults to SAE config if omitted).'
    )
    
    parser.add_argument(
        '--local-model',
        action='store_true',
        help='Use local model files only (set local_files_only=True).'
    )
    
    parser.add_argument(
        '--hook-name',
        type=str,
        default=None,
        help='Hook point name (defaults to SAE config if omitted).'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--use-arrow',
        action='store_true',
        help='Interpret dataset as Arrow format.'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (default: 8).'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Max sequence length; truncate beyond this (default: None, no truncation).'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max samples to process for testing (default: None, process all).'
    )
    
    # Reasoning token parameters
    parser.add_argument(
        '--reasoning-tokens',
        type=str,
        default=None,
        help='JSON file path for reasoning token list.'
    )
    
    parser.add_argument(
        '--separate-reasoning',
        action='store_true',
        help='Save activations for reasoning and normal tokens separately.'
    )
    
    parser.add_argument(
        '--only-reasoning',
        action='store_true',
        help='Save activations for reasoning tokens only.'
    )
    
    parser.add_argument(
        '--save-token-masks',
        action='store_true',
        help='Save token-type masks (mark which are reasoning tokens).'
    )
    
    # Hidden state parameters
    parser.add_argument(
        '--save-hidden-states',
        action='store_true',
        help='Save hidden states (raw model activations, not SAE-encoded).'
    )
    
    # Output parameters
    parser.add_argument(
        '--save-token-level',
        action='store_true',
        help='Save token-level activations (may generate large JSON files).'
    )
    
    parser.add_argument(
        '--no-save-npy',
        action='store_true',
        help='Do not save NumPy outputs (HDF5 only).'
    )
    
    parser.add_argument(
        '--no-save-h5',
        action='store_true',
        help='Do not save HDF5 outputs (NumPy only).'
    )
    
    # Device parameters
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Compute device (cuda/cpu; default: auto-detect).'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type (default: float32).'
    )
    
    # Misc parameters
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output.'
    )

    parser.add_argument(
        '--normal_sample_limit',
        type=int,
        default=0,
        help='Save up to N normal-token SAE/hidden-state vectors (default: 0 = disable).'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.no_save_npy and args.no_save_h5:
        parser.error("Cannot set both --no-save-npy and --no-save-h5.")
    
    if args.only_reasoning and not args.reasoning_tokens:
        parser.error("--only-reasoning requires --reasoning-tokens.")
    
    if args.separate_reasoning and not args.reasoning_tokens:
        parser.error("--separate-reasoning requires --reasoning-tokens.")
    
    if not Path(args.sae_path).exists():
        parser.error(f"SAE path does not exist: {args.sae_path}")
    
    if not Path(args.dataset).exists():
        parser.error(f"Dataset path does not exist: {args.dataset}")
    
    if args.reasoning_tokens and not Path(args.reasoning_tokens).exists():
        parser.error(f"Reasoning token file does not exist: {args.reasoning_tokens}")
    
    return args


def create_h5_writers(output_dir, config_info, args):
    """Create HDF5 writers - only create full datasets for reasoning."""
    writers = {}
    
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'bfloat16': np.float32  # HDF5 does not support bfloat16, store as float32
    }
    np_dtype = dtype_map[args.dtype]
    
    # Only create HDF5 files for reasoning (full save)
    if args.reasoning_tokens and (args.only_reasoning or args.separate_reasoning):
        f_reason_sae = h5py.File(output_dir / "sae_activations_reasoning.h5", 'w')
        writers['reasoning_sae'] = {
            'file': f_reason_sae,
            'dataset': f_reason_sae.create_dataset(
                'activations',
                shape=(0, config_info['d_sae']),
                maxshape=(None, config_info['d_sae']),
                dtype=np_dtype,
                chunks=(1000, config_info['d_sae']),
                compression='gzip',
                compression_opts=4
            ),
            'count': 0
        }
        
        if args.save_hidden_states:
            f_reason_hs = h5py.File(output_dir / "hidden_states_reasoning.h5", 'w')
            writers['reasoning_hs'] = {
                'file': f_reason_hs,
                'dataset': f_reason_hs.create_dataset(
                    'hidden_states',
                    shape=(0, config_info['d_in']),
                    maxshape=(None, config_info['d_in']),
                    dtype=np_dtype,
                    chunks=(1000, config_info['d_in']),
                    compression='gzip',
                    compression_opts=4
                ),
                'count': 0
            }
    if args.normal_sample_limit > 0:
        f_normal_sae = h5py.File(output_dir / "sae_activations_normal_sampled.h5", 'w')
        writers['normal_sae'] = {
            'file': f_normal_sae,
            'dataset': f_normal_sae.create_dataset(
                'activations',
                shape=(0, config_info['d_sae']),
                maxshape=(None, config_info['d_sae']),
                dtype=np_dtype,
                chunks=(1000, config_info['d_sae']),
                compression='gzip',
                compression_opts=4
            ),
            'count': 0
        }
        
        if args.save_hidden_states:
            f_normal_hs = h5py.File(output_dir / "hidden_states_normal_sampled.h5", 'w')
            writers['normal_hs'] = {
                'file': f_normal_hs,
                'dataset': f_normal_hs.create_dataset(
                    'hidden_states',
                    shape=(0, config_info['d_in']),
                    maxshape=(None, config_info['d_in']),
                    dtype=np_dtype,
                    chunks=(1000, config_info['d_in']),
                    compression='gzip',
                    compression_opts=4
                ),
                'count': 0
            }
    return writers


def create_mean_accumulators(config_info, args):
    """Create mean vector accumulators (for all and normal)."""
    accumulators = {}
    
    # Accumulator for all
    if not args.only_reasoning:
        accumulators['all_sae'] = {
            'sum': np.zeros(config_info['d_sae'], dtype=np.float64),  # Use float64 for precision
            'count': 0
        }
        
        if args.save_hidden_states:
            accumulators['all_hs'] = {
                'sum': np.zeros(config_info['d_in'], dtype=np.float64),
                'count': 0
            }
    
    # Accumulator for normal
    if args.separate_reasoning and not args.only_reasoning:
        accumulators['normal_sae'] = {
            'sum': np.zeros(config_info['d_sae'], dtype=np.float64),
            'count': 0
        }
        
        if args.save_hidden_states:
            accumulators['normal_hs'] = {
                'sum': np.zeros(config_info['d_in'], dtype=np.float64),
                'count': 0
            }
    
    return accumulators


def update_mean_accumulator(accumulator, data):
    """
    Update the mean accumulator with an online algorithm (avoid overflow).
    
    Formula: mean_new = mean_old + (batch_mean - mean_old) * batch_size / (count + batch_size)
    
    Benefits:
    1. Numerically stable, avoids overflow
    2. Memory efficient, keeps only the current mean
    3. Supports batch-by-batch updates
    
    Args:
        accumulator: dict with 'mean' and 'count'
        data: numpy array of shape (n, features) or (n, 1, features)
    """
    if data.size == 0:
        return
    
    # Remove extra dimension (n, 1, features) -> (n, features)
    if data.ndim == 3 and data.shape[1] == 1:
        data = data.squeeze(1)
    
    # Cast to float64 for precision
    data = data.astype(np.float64)
    
    # Batch size and mean
    batch_size = len(data)
    batch_mean = data.mean(axis=0)
    
    # Existing sample count
    old_count = accumulator['count']
    new_count = old_count + batch_size
    
    # Online mean update
    # mean_new = (old_count * mean_old + batch_size * batch_mean) / new_count
    # Rewritten as: mean_new = mean_old + (batch_mean - mean_old) * batch_size / new_count
    if old_count == 0:
        # First batch, use directly
        accumulator['mean'] = batch_mean
    else:
        # Incremental update
        delta = batch_mean - accumulator['mean']
        accumulator['mean'] += delta * (batch_size / new_count)
    
    # Update count
    accumulator['count'] = new_count


def finalize_mean_accumulators(accumulators, output_dir, args):
    """Compute and save mean vectors."""
    saved_files = []
    
    for key, acc in accumulators.items():
        if acc['count'] == 0:
            print(f"Warning: {key} has no data; skipping save")
            continue
        
        # Mean already computed via online updates
        mean_vector = acc['mean'].astype(np.float32)
        
        # Determine filename
        if 'sae' in key:
            if 'all' in key:
                filename = "sae_activations_all_mean.npy"
            elif 'normal' in key:
                filename = "sae_activations_normal_mean.npy"
            else:
                filename = f"{key}_mean.npy"
        else:  # hidden states
            if 'all' in key:
                filename = "hidden_states_all_mean.npy"
            elif 'normal' in key:
                filename = "hidden_states_normal_mean.npy"
            else:
                filename = f"{key}_mean.npy"
        
        # Save
        filepath = output_dir / filename
        np.save(filepath, mean_vector)
        saved_files.append(str(filepath))
        
        print(f"✓ Saved mean vector {key}: {filepath}")
        print(f"  Shape: {mean_vector.shape}, samples: {acc['count']:,}")
    
    return saved_files


def append_to_h5(writer_info, data):
    """Append data to an HDF5 dataset."""
    if data.size == 0:
        return
    
    # Remove extra dimension (n, 1, features) -> (n, features)
    if data.ndim == 3 and data.shape[1] == 1:
        data = data.squeeze(1)
    
    dataset = writer_info['dataset']
    count = writer_info['count']
    new_count = count + len(data)
    
    # Expand dataset
    dataset.resize(new_count, axis=0)
    
    # Write data
    dataset[count:new_count] = data
    
    # Update count
    writer_info['count'] = new_count


def close_h5_writers(writers):
    """Close all HDF5 files."""
    for writer_info in writers.values():
        writer_info['file'].close()


def load_reasoning_tokens(token_file_path, tokenizer):
    """
    Load the reasoning token list and convert to token IDs.
    Supports single tokens and multi-token sequences.
    
    Args:
        token_file_path: JSON file path
        tokenizer: tokenizer
    
    Returns:
        single_token_ids: set of single token IDs
        multi_token_sequences: list of token ID sequences
        reasoning_token_strings: list of token strings
        token_id_mapping: dict mapping string to token IDs
    """
    print(f"\nLoading reasoning token list: {token_file_path}")
    
    with open(token_file_path, 'r', encoding='utf-8') as f:
        token_strings = json.load(f)
    
    print(f"Reasoning token count: {len(token_strings)}")
    
    # Store single tokens and multi-token sequences separately
    single_token_ids = set()
    multi_token_sequences = []
    token_id_mapping = {}
        
    for token_str in token_strings:
        # Encode with tokenizer
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        
        if len(token_ids) == 1:
            # Single token
            token_id = token_ids[0]
            single_token_ids.add(token_id)
            token_id_mapping[token_str] = [token_id]
            print(f"  Single token: '{token_str}' -> {token_id}")
        elif len(token_ids) > 1:
            # Multi-token sequence
            multi_token_sequences.append({
                'string': token_str,
                'ids': token_ids,
                'length': len(token_ids)
            })
            token_id_mapping[token_str] = token_ids
            print(f"  Multi-token: '{token_str}' -> {token_ids}")
        else:
            print(f"  Warning: '{token_str}' could not be encoded")
    
    print("\nStats:")
    print(f"  Single tokens: {len(single_token_ids)}")
    print(f"  Multi-token sequences: {len(multi_token_sequences)}")
    
    # Sort multi-token sequences by length (match longer first)
    multi_token_sequences.sort(key=lambda x: x['length'], reverse=True)
    
    if len(single_token_ids) > 0:
        print(f"\nExample single token IDs: {list(single_token_ids)[:10]}")
    
    if len(multi_token_sequences) > 0:
        print("\nExample multi-token sequences:")
        for seq in multi_token_sequences[:5]:
            print(f"  '{seq['string']}' ({seq['length']} tokens): {seq['ids']}")
    
    return single_token_ids, multi_token_sequences, token_strings, token_id_mapping


def find_multi_token_matches(tokens, multi_token_sequences):
    """
    Optimized multi-token matching using a sliding window and precomputation.
    """
    if not multi_token_sequences:
        return {}
    
    batch_size, seq_len = tokens.shape
    tokens_np = tokens.cpu().numpy()
    matches = {}
    
    # Group by length to avoid checking all patterns each time
    patterns_by_length = {}
    for seq_info in multi_token_sequences:
        length = seq_info['length']
        if length not in patterns_by_length:
            patterns_by_length[length] = []
        patterns_by_length[length].append(seq_info)
    
    # Process from longest to shortest
    for length in sorted(patterns_by_length.keys(), reverse=True):
        patterns = patterns_by_length[length]
        
        # Convert to numpy arrays for faster comparison
        pattern_arrays = [np.array(p['ids']) for p in patterns]
        
        for b in range(batch_size):
            for start_pos in range(seq_len - length + 1):
                # Skip if already matched
                if any((b, start_pos + i) in matches for i in range(length)):
                    continue
                
                # Extract window
                window = tokens_np[b, start_pos:start_pos+length]
                
                # Compare with all same-length patterns (vectorized)
                for i, pattern in enumerate(pattern_arrays):
                    if np.array_equal(window, pattern):
                        seq_info = patterns[i]
                        matches[(b, start_pos)] = {
                            'string': seq_info['string'],
                            'ids': seq_info['ids'],
                            'length': length,
                            'first_token_id': seq_info['ids'][0]
                        }
                        # Mark placeholders
                        for j in range(1, length):
                            matches[(b, start_pos + j)] = None
                        break
    
    # Remove placeholders
    return {k: v for k, v in matches.items() if v is not None}


def create_reasoning_mask(tokens, single_token_ids, multi_token_sequences):
    """
    Create a reasoning-token mask supporting single tokens and multi-token sequences.
    For multi-token sequences, only the last token is marked.
    
    Args:
        tokens: token IDs with shape (batch, seq_len)
        single_token_ids: set of single reasoning token IDs
        multi_token_sequences: list of multi-token patterns
    
    Returns:
        mask: boolean mask of shape (batch, seq_len), True indicates a reasoning token
        multi_token_matches: dict of multi-token match positions
    """
    batch_size, seq_len = tokens.shape
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    
    # 1. Mark single tokens
    for token_id in single_token_ids:
        mask |= (tokens == token_id)
    
    # 2. Find and mark multi-token sequences (only the last token)
    multi_token_matches = {}
    if multi_token_sequences:
        matches = find_multi_token_matches(tokens, multi_token_sequences)
        
        for (b, start_pos), match_info in matches.items():
            # Mark only the last token of the sequence (not the first)
            last_pos = start_pos + match_info['length'] - 1
            mask[b, last_pos] = True
            
            # Update match info with last-token position
            multi_token_matches[(b, last_pos)] = {
                **match_info,
                'start_pos': start_pos,  # Start position
                'last_pos': last_pos,     # Last position
                'last_token_id': match_info['ids'][-1]  # Last token ID
            }
    
    return mask, multi_token_matches


def extract_activations_with_multi_tokens(
    tokens, activations, single_token_ids, multi_token_sequences, hs_acts=None
):
    """
    Vectorized extraction of activations (supports SAE activations and hidden states).
    For multi-token sequences, only the last token's activation is extracted.
    
    Args:
        tokens: token IDs with shape (batch, seq_len)
        activations: activations with shape (batch, seq_len, features) (SAE or hidden states)
        single_token_ids: set of single reasoning token IDs
        multi_token_sequences: list of multi-token patterns
        hs_acts: optional hidden states
    
    Returns:
        reasoning_acts: activations for reasoning tokens
        normal_acts: activations for normal tokens
        match_info: match metadata dictionary
        reasoning_hs_acts: hidden states for reasoning tokens (if hs_acts provided)
        normal_hs_acts: hidden states for normal tokens (if hs_acts provided)
        mask: mask
    """
    batch_size, seq_len, features = activations.shape
    if hs_acts is not None:
        _, _, hs_features = hs_acts.shape
    
    # Create mask (keep on GPU)
    mask, multi_token_matches = create_reasoning_mask(
        tokens, single_token_ids, multi_token_sequences
    )
    
    # Stats
    match_info = {
        'single_token_matches': 0,
        'multi_token_matches': len(multi_token_matches),
        'multi_token_details': []
    }
    
    # Handle different dtypes
    # NumPy does not support bfloat16; cast to float32
    if activations.dtype == torch.bfloat16:
        activations = activations.float()
    elif activations.dtype == torch.float16:
        activations = activations.float()
        
    # Convert to numpy (one-time conversion)
    mask_np = mask.cpu().numpy()
    activations_np = activations.cpu().numpy()
    
    # Vectorized extraction - key optimization
    # Reshape (batch, seq, features) -> (batch*seq, features)
    flat_acts = activations_np.reshape(-1, features)
    flat_mask = mask_np.reshape(-1)
    
    # Extract all reasoning activations in one shot
    reasoning_acts = flat_acts[flat_mask]  # One indexing pass
    normal_acts = flat_acts[~flat_mask]
    
    # Expand dims for compatibility (n, features) -> (n, 1, features)
    reasoning_acts = reasoning_acts[:, np.newaxis, :] if reasoning_acts.size > 0 else np.empty((0, 1, features))
    if normal_acts is not None:
        normal_acts = normal_acts[:, np.newaxis, :] if normal_acts.size > 0 else np.empty((0, 1, features))
    
    reasoning_hs_acts = None
    normal_hs_acts = None
    if hs_acts is not None:
        # Handle different dtypes
        if hs_acts.dtype == torch.bfloat16 or hs_acts.dtype == torch.float16:
            hs_acts = hs_acts.float()
            
        hs_acts_np = hs_acts.cpu().numpy()
        
        # Vectorized extraction
        flat_hs_acts = hs_acts_np.reshape(-1, hs_features)
        reasoning_hs_acts = flat_hs_acts[flat_mask]
        normal_hs_acts = flat_hs_acts[~flat_mask]
        
        # Expand dims for compatibility
        reasoning_hs_acts = reasoning_hs_acts[:, np.newaxis, :] if reasoning_hs_acts.size > 0 else np.empty((0, 1, hs_features))
        if normal_hs_acts is not None:
            normal_hs_acts = normal_hs_acts[:, np.newaxis, :] if normal_hs_acts.size > 0 else np.empty((0, 1, hs_features))
            
    # Count single-token matches
    match_info['single_token_matches'] = int(flat_mask.sum()) - len(multi_token_matches)
    
    # Populate multi-token match details
    for (b, last_pos), match in multi_token_matches.items():
        match_info['multi_token_details'].append({
            'batch': b,
            'start_position': match['start_pos'],
            'last_position': last_pos,
            'sequence_length': match['length'],
            'string': match['string'],
            'token_ids': match['ids'],
            'extracted_token_id': match['last_token_id']
        })
    
    return reasoning_acts, normal_acts, match_info, reasoning_hs_acts, normal_hs_acts, mask, flat_acts, flat_hs_acts


def get_sae_activations(tokens, model, sae, hook_name, return_hidden_states=False):
    """
    Compute SAE activations and/or hidden states for given tokens.
    
    Args:
        tokens: token IDs with shape (batch, seq_len)
        model: HookedTransformer model
        sae: SAE model
        hook_name: layer name to extract activations from
        return_hidden_states: whether to return raw hidden states
    
    Returns:
        If return_hidden_states=False:
            sae_activations: SAE activations with shape (batch, seq_len, sae_features)
        If return_hidden_states=True:
            (sae_activations, hidden_states): SAE activations and raw hidden states
    """
    with torch.no_grad():
        # Run model and cache activations from the target layer
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        
        # Extract hidden states for the layer
        hidden_states = cache[hook_name]  # shape: (batch, seq_len, d_model)
        
        # Encode with SAE
        sae_activations = sae.encode(hidden_states)  # shape: (batch, seq_len, sae_features)
        
        if return_hidden_states:
            return sae_activations, hidden_states
        else:
            return sae_activations


def process_batch_streaming(
    batch, device, model, sae, hook_name, args,
    single_token_ids, multi_token_sequences,
    h5_writers,
    mean_accumulators,  # New: mean vector accumulators
    all_masks,
    total_tokens, total_reasoning_tokens, total_normal_tokens,
    batch_match_info
):
    """Stream batches - write reasoning to HDF5, accumulate all/normal means."""

    def append_sampled_vectors(sae_data, hs_data=None):
        """Append sampled activations up to normal_sample_limit."""
        if args.normal_sample_limit <= 0 or 'normal_sae' not in h5_writers:
            return
        if sae_data is None or sae_data.size == 0:
            return
        writer_sae = h5_writers['normal_sae']
        current = writer_sae['count']
        if current >= args.normal_sample_limit:
            return
        sample_size = min(args.normal_sample_limit - current, sae_data.shape[0])
        append_to_h5(writer_sae, sae_data[:sample_size])
        if args.save_hidden_states and hs_data is not None and 'normal_hs' in h5_writers:
            append_to_h5(h5_writers['normal_hs'], hs_data[:sample_size])
    
    # Get token IDs
    if isinstance(batch, dict):
        tokens = torch.tensor(batch["input_ids"]).to(device)
    else:
        tokens = torch.tensor([item["input_ids"] for item in batch]).to(device)
    
    if args.max_length is not None and tokens.shape[1] > args.max_length:
        tokens = tokens[:, :args.max_length]
    
    # Compute activations
    if args.save_hidden_states:
        sae_acts, hidden_acts = get_sae_activations(tokens, model, sae, hook_name, return_hidden_states=True)
    else:
        sae_acts = get_sae_activations(tokens, model, sae, hook_name, return_hidden_states=False)
        hidden_acts = None
    
    # Convert data types
    if args.dtype == 'float16':
        sae_acts = sae_acts.half()
        if hidden_acts is not None:
            hidden_acts = hidden_acts.half()
    elif args.dtype == 'bfloat16':
        sae_acts = sae_acts.bfloat16()
        if hidden_acts is not None:
            hidden_acts = hidden_acts.bfloat16()
    
    batch_size, seq_len = tokens.shape
    batch_total_tokens = batch_size * seq_len
    
    # Extract activations
    if single_token_ids is not None or multi_token_sequences is not None:
        batch_reasoning_sae, batch_normal_sae, match_info, batch_reasoning_hs, batch_normal_hs, mask, flat_sae_acts, flat_hs_acts = \
            extract_activations_with_multi_tokens(
                tokens, sae_acts,
                single_token_ids if single_token_ids else set(),
                multi_token_sequences if multi_token_sequences else [],
                hidden_acts
            )
        
        batch_reasoning_tokens = len(batch_reasoning_sae)
        batch_normal_tokens = len(batch_normal_sae) if batch_normal_sae is not None else 0
        
        total_tokens += batch_total_tokens
        total_reasoning_tokens += batch_reasoning_tokens
        total_normal_tokens += batch_normal_tokens
        batch_match_info.append(match_info)
        
        # ========== Process reasoning activations (full HDF5 save) ==========
        if args.only_reasoning:
            if batch_reasoning_sae.size > 0:
                append_to_h5(h5_writers['reasoning_sae'], batch_reasoning_sae)
            if args.save_hidden_states and batch_reasoning_hs is not None and batch_reasoning_hs.size > 0:
                append_to_h5(h5_writers['reasoning_hs'], batch_reasoning_hs)
        
        elif args.separate_reasoning:
            # Save reasoning (full HDF5 save)
            if batch_reasoning_sae.size > 0:
                append_to_h5(h5_writers['reasoning_sae'], batch_reasoning_sae)
            if args.save_hidden_states and batch_reasoning_hs is not None and batch_reasoning_hs.size > 0:
                append_to_h5(h5_writers['reasoning_hs'], batch_reasoning_hs)
            
            # ========== Accumulate normal into mean ==========
            if batch_normal_sae is not None and batch_normal_sae.size > 0:
                update_mean_accumulator(mean_accumulators['normal_sae'], batch_normal_sae)
            if args.save_hidden_states and batch_normal_hs is not None and batch_normal_hs.size > 0:
                update_mean_accumulator(mean_accumulators['normal_hs'], batch_normal_hs)
            
            append_sampled_vectors(batch_normal_sae, batch_normal_hs)
            
            # ========== Accumulate all into mean ==========
            # sae_acts_np = sae_acts.float().cpu().numpy() if sae_acts.dtype in [torch.bfloat16, torch.float16] else sae_acts.cpu().numpy()
            # sae_acts_flat = sae_acts_np.reshape(-1, sae_acts_np.shape[-1])
            update_mean_accumulator(mean_accumulators['all_sae'], flat_sae_acts)
            
            if args.save_hidden_states and hidden_acts is not None:
                # hidden_acts_np = hidden_acts.float().cpu().numpy() if hidden_acts.dtype in [torch.bfloat16, torch.float16] else hidden_acts.cpu().numpy()
                # hidden_acts_flat = hidden_acts_np.reshape(-1, hidden_acts_np.shape[-1])
                update_mean_accumulator(mean_accumulators['all_hs'], flat_hs_acts)
                # del hidden_acts_np, hidden_acts_flat
                append_sampled_vectors(flat_sae_acts, flat_hs_acts)
                del flat_hs_acts
            else:
                append_sampled_vectors(flat_sae_acts, None)
            
            # del sae_acts_np, sae_acts_flat
            del flat_sae_acts
        
        else:
            # No separate_reasoning; accumulate all only
            # sae_acts_np = sae_acts.float().cpu().numpy() if sae_acts.dtype in [torch.bfloat16, torch.float16] else sae_acts.cpu().numpy()
            # sae_acts_flat = sae_acts_np.reshape(-1, sae_acts_np.shape[-1])
            update_mean_accumulator(mean_accumulators['all_sae'], flat_sae_acts)
            
            if args.save_hidden_states and hidden_acts is not None:
                # hidden_acts_np = hidden_acts.float().cpu().numpy() if hidden_acts.dtype in [torch.bfloat16, torch.float16] else hidden_acts.cpu().numpy()
                # hidden_acts_flat = hidden_acts_np.reshape(-1, hidden_acts_np.shape[-1])
                update_mean_accumulator(mean_accumulators['all_hs'], flat_hs_acts)
                # del hidden_acts_np, hidden_acts_flat
                append_sampled_vectors(flat_sae_acts, flat_hs_acts)
                del flat_hs_acts
            else:
                append_sampled_vectors(flat_sae_acts, None)
            
            # del sae_acts_np, sae_acts_flat
            del flat_sae_acts
        
        # Masks can still be accumulated in memory (small footprint)
        if args.save_token_masks:
            all_masks.append(mask.cpu().numpy())
        
        # Delete numpy arrays
        del batch_reasoning_sae, batch_normal_sae
        if batch_reasoning_hs is not None:
            del batch_reasoning_hs
        if batch_normal_hs is not None:
            del batch_normal_hs
    
    else:
        # No reasoning tokens; accumulate all only
        sae_acts_np = sae_acts.float().cpu().numpy() if sae_acts.dtype in [torch.bfloat16, torch.float16] else sae_acts.cpu().numpy()
        sae_acts_flat = sae_acts_np.reshape(-1, sae_acts_np.shape[-1])
        update_mean_accumulator(mean_accumulators['all_sae'], sae_acts_flat)
        
        if args.save_hidden_states and hidden_acts is not None:
            hidden_acts_np = hidden_acts.float().cpu().numpy() if hidden_acts.dtype in [torch.bfloat16, torch.float16] else hidden_acts.cpu().numpy()
            hidden_acts_flat = hidden_acts_np.reshape(-1, hidden_acts_np.shape[-1])
            update_mean_accumulator(mean_accumulators['all_hs'], hidden_acts_flat)
            append_sampled_vectors(sae_acts_flat, hidden_acts_flat)
            del hidden_acts_np, hidden_acts_flat
        else:
            append_sampled_vectors(sae_acts_flat, None)
        
        
        del sae_acts_np, sae_acts_flat
        
        total_tokens += batch_total_tokens
    
    # Explicitly free GPU memory
    del tokens, sae_acts
    if hidden_acts is not None:
        del hidden_acts
    
    return total_tokens, total_reasoning_tokens, total_normal_tokens


def print_memory_usage(device, prefix=""):
    """Print current memory usage."""
    if device.startswith('cuda'):
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"{prefix}GPU memory - allocated: {allocated:.2f} GB, "
              f"reserved: {reserved:.2f} GB, peak: {max_allocated:.2f} GB")
    
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024**2
    print(f"{prefix}RAM: {ram_mb:.2f} MB")


def main():
    args = parse_args()
    
    print("="*60)
    print("Computing dataset activations with SAE")
    if args.reasoning_tokens:
        print("(supports separating reasoning tokens)")
    if args.save_hidden_states:
        print("(also saves hidden states)")
    print("(normal/all saved as means only; reasoning saved in full)")
    print("="*60)
    
    # ============================================
    # 1. Load SAE and model
    # ============================================
    
    print(f"\nLoading SAE from local path: {args.sae_path}")
    sae = SAE.load_from_pretrained(args.sae_path)
    
    # Get model info from SAE config
    print("\nSAE config:")
    print(f"  Model name: {sae.cfg.metadata.model_name}")
    print(f"  Hook point: {sae.cfg.metadata.hook_name}")
    print(f"  d_in: {sae.cfg.d_in}")
    print(f"  d_sae: {sae.cfg.d_sae}")
    
    # Resolve model name and hook point
    model_name = args.model_path if args.model_path else sae.cfg.metadata.model_name
    hook_name = args.hook_name if args.hook_name else sae.cfg.metadata.hook_name
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading language model: {model_name}")
    print(f"Device: {device}")
    
    # Load model parameters
    load_kwargs = {'device': device}
    if args.local_model:
        model = load_model(
            model_class_name=sae.cfg.metadata.model_class_name,
            model_name=model_name,
            device=device,
            model_from_pretrained_kwargs=sae.cfg.metadata.model_from_pretrained_kwargs,
        )
    else:
        model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
    print("Model loaded")
    
    # Move SAE to the same device
    sae = sae.to(device)
    
    # ============================================
    # 2. Load reasoning token list (if provided)
    # ============================================
    
    single_token_ids = None
    multi_token_sequences = None
    reasoning_token_strings = None
    token_id_mapping = None
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.reasoning_tokens:
        single_token_ids, multi_token_sequences, reasoning_token_strings, token_id_mapping = \
            load_reasoning_tokens(args.reasoning_tokens, model.tokenizer)
        
        # Save token ID mapping
        mapping_path = output_dir / "reasoning_token_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_strings': reasoning_token_strings,
                'token_id_mapping': {k: v for k, v in token_id_mapping.items()},
                'single_token_ids': list(single_token_ids),
                'multi_token_sequences': [
                    {
                        'string': seq['string'],
                        'ids': seq['ids'],
                        'length': seq['length']
                    }
                    for seq in multi_token_sequences
                ]
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved token ID mapping: {mapping_path}")
    
    # ============================================
    # 3. Load dataset
    # ============================================
    
    print(f"\nLoading dataset: {args.dataset}")
    print(f"Format: {'Arrow' if args.use_arrow else 'JSONL'}")
    
    if args.use_arrow:
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    
    # If max samples specified, truncate dataset
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limiting samples to: {args.max_samples}")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Inspect the first sample
    if len(dataset) > 0:
        sample = dataset[0]
        if args.verbose:
            print("\nDataset sample example:")
            print(f"  Keys: {list(sample.keys())}")
        if 'input_ids' in sample:
            print(f"  input_ids length: {len(sample['input_ids'])}")
    
    # ============================================
    # 4. Batch-process dataset
    # ============================================
    
    print("\nProcessing config:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length if args.max_length else 'no truncation'}")
    print(f"  Hook name: {hook_name}")
    print(f"  Data type: {args.dtype}")
    print(f"  Save hidden states: {args.save_hidden_states}")
    if args.reasoning_tokens:
        print(
            "  Reasoning tokens: "
            f"{len(single_token_ids) if single_token_ids else 0} single-token + "
            f"{len(multi_token_sequences) if multi_token_sequences else 0} multi-token sequences"
        )
        print(
            "  Separation mode: "
            f"{'reasoning only' if args.only_reasoning else 'separate' if args.separate_reasoning else 'all'}"
        )
    print("  Save strategy: reasoning saved in full, all/normal saved as means")
    print(f"  Output directory: {args.output}")
    print("\nStarting dataset processing...")
    
    # Prepare config info (for HDF5)
    config_info = {
        'sae_path': args.sae_path,
        'model_name': model_name,
        'hook_name': hook_name,
        'dataset_path': args.dataset,
        'use_arrow': args.use_arrow,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'max_samples': args.max_samples,
        'dtype': args.dtype,
        'device': device,
        'd_in': sae.cfg.d_in,
        'd_sae': sae.cfg.d_sae,
        'save_hidden_states': args.save_hidden_states,
    }
    
    # Create HDF5 writers (reasoning only)
    h5_writers = create_h5_writers(output_dir, config_info, args)
    
    # Create mean accumulators (all and normal)
    mean_accumulators = create_mean_accumulators(config_info, args)
    
    # Store masks
    all_masks = []
    
    # Initialize counters
    total_tokens = 0
    total_reasoning_tokens = 0
    total_normal_tokens = 0
    batch_match_info = []
    
    try:
        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
            batch = dataset[i:i+args.batch_size]
            print_memory_usage(device, "[Before batch] ")
            total_tokens, total_reasoning_tokens, total_normal_tokens = \
                process_batch_streaming(
                    batch, device, model, sae, hook_name, args,
                    single_token_ids, multi_token_sequences,
                    h5_writers,
                    mean_accumulators,  # Pass accumulators
                    all_masks,
                    total_tokens, total_reasoning_tokens, total_normal_tokens,
                    batch_match_info
                )
            print_memory_usage(device, "[After batch] ")
                
            # Periodically clear cache
            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        # Ensure all HDF5 files are closed
        print("\nClosing HDF5 files...")
        close_h5_writers(h5_writers)
    
    # ============================================
    # 5. Print stats
    # ============================================
    
    print("\nProcessing stats:")
    print(f"  Total tokens: {total_tokens:,}")
    
    # Initialize variables
    total_single_matches = 0
    total_multi_matches = 0
    multi_token_stats = {}
    
    if single_token_ids is not None or multi_token_sequences is not None:
        print(f"  Reasoning tokens: {total_reasoning_tokens:,} ({total_reasoning_tokens/total_tokens*100:.2f}%)")
        print(f"  Normal tokens: {total_normal_tokens:,} ({total_normal_tokens/total_tokens*100:.2f}%)")
        
        # Aggregate match info
        total_single_matches = sum(info['single_token_matches'] for info in batch_match_info)
        total_multi_matches = sum(info['multi_token_matches'] for info in batch_match_info)
        
        print("\nMatch details:")
        print(f"  Single-token matches: {total_single_matches:,}")
        print(f"  Multi-token sequence matches: {total_multi_matches:,} (last token only)")
        
        if total_multi_matches > 0:
            # Count matches per multi-token sequence
            multi_token_stats = {}
            for info in batch_match_info:
                for detail in info['multi_token_details']:
                    string = detail['string']
                    multi_token_stats[string] = multi_token_stats.get(string, 0) + 1
            
            print("\nMulti-token match stats (Top 10):")
            print("  Note: only the last token activation is kept for the sequences below")
            sorted_stats = sorted(multi_token_stats.items(), key=lambda x: x[1], reverse=True)
            for idx, (string, count) in enumerate(sorted_stats[:10], 1):
                # Show token breakdown
                token_ids = token_id_mapping.get(string, [])
                if token_ids:
                    decoded_tokens = [model.tokenizer.decode([tid]) for tid in token_ids]
                    print(f"  {idx}. '{string}': {count:,} times")
                    print(f"      Token sequence: {decoded_tokens} -> keep last: '{decoded_tokens[-1]}'")
                else:
                    print(f"  {idx}. '{string}': {count:,} times")
            
            # Save detailed multi-token match info
            multi_match_path = output_dir / "multi_token_matches.json"
            all_multi_matches = []
            for info in batch_match_info:
                all_multi_matches.extend(info['multi_token_details'])
            
            with open(multi_match_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_matches': total_multi_matches,
                    'match_statistics': multi_token_stats,
                    'all_matches': all_multi_matches[:1000]  # Save first 1000 detailed matches only
                }, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Saved multi-token match details: {multi_match_path}")
    
    # ============================================
    # 6. Save results
    # ============================================
    
    print(f"\nSaving results to {args.output}...")
    
    # Initialize saved_files
    saved_files = []
    
    # ========== Compute and save mean vectors ==========
    print("\nComputing and saving mean vectors...")
    mean_files = finalize_mean_accumulators(mean_accumulators, output_dir, args)
    saved_files.extend(mean_files)
    
    # Update config info
    config_info['total_tokens'] = int(total_tokens)
    config_info['save_strategy'] = 'reasoning: full, all/normal: mean only'
    
    if single_token_ids is not None or multi_token_sequences is not None:
        config_info.update({
            'reasoning_tokens_file': args.reasoning_tokens,
            'num_single_token_types': len(single_token_ids) if single_token_ids else 0,
            'num_multi_token_sequences': len(multi_token_sequences) if multi_token_sequences else 0,
            'total_reasoning_tokens': int(total_reasoning_tokens),
            'total_normal_tokens': int(total_normal_tokens),
            'reasoning_ratio': float(total_reasoning_tokens / total_tokens) if total_tokens > 0 else 0,
            'separate_reasoning': args.separate_reasoning,
            'only_reasoning': args.only_reasoning,
            'single_token_matches': int(total_single_matches),
            'multi_token_matches': int(total_multi_matches),
        })
    
    # Save masks (if needed)
    if all_masks and args.save_token_masks:
        masks = np.concatenate(all_masks, axis=0)
        mask_path = output_dir / "token_masks.npy"
        np.save(mask_path, masks)
        print(f"✓ Saved token mask: {mask_path}")
        saved_files.append(str(mask_path))
        del masks
    
    # Save config info
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    print(f"✓ Saved config info: {config_path}")
    saved_files.append(str(config_path))
    
    # Add HDF5 files to saved_files
    # for key, writer_info in h5_writers.items():
    #     saved_files.append(writer_info['file'].filename)
    
    # Save processing log
    log_path = output_dir / "processing_log.txt"
    with open(log_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SAE activation processing log\n")
        f.write("="*60 + "\n\n")
        
        f.write("[Input Config]\n")
        f.write(f"SAE path: {args.sae_path}\n")
        f.write(f"Dataset path: {args.dataset}\n")
        f.write(f"Dataset format: {'Arrow' if args.use_arrow else 'JSONL'}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Hook point: {hook_name}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("[Processing Config]\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Max length: {args.max_length if args.max_length else 'no truncation'}\n")
        f.write(f"Max samples: {args.max_samples if args.max_samples else 'all'}\n")
        f.write(f"Data type: {args.dtype}\n")
        f.write(f"Save hidden states: {args.save_hidden_states}\n")
        f.write("Save strategy: reasoning full, all/normal mean only\n\n")
        
        if single_token_ids is not None or multi_token_sequences is not None:
            f.write("[Reasoning Token Config]\n")
            f.write(f"Token file: {args.reasoning_tokens}\n")
            f.write(f"Single-token types: {len(single_token_ids) if single_token_ids else 0}\n")
            f.write(f"Multi-token sequences: {len(multi_token_sequences) if multi_token_sequences else 0}\n")
            f.write("Multi-token handling: keep only the last token activation\n")
            f.write(
                "Separation mode: "
                f"{'reasoning only' if args.only_reasoning else 'separate' if args.separate_reasoning else 'all'}\n"
            )
            f.write(f"Save masks: {args.save_token_masks}\n\n")
        
        f.write("[Results]\n")
        f.write(f"Total tokens: {total_tokens:,}\n")
        
        if single_token_ids is not None or multi_token_sequences is not None:
            f.write(f"Reasoning tokens: {total_reasoning_tokens:,} ({total_reasoning_tokens/total_tokens*100:.2f}%)\n")
            f.write(f"Normal tokens: {total_normal_tokens:,} ({total_normal_tokens/total_tokens*100:.2f}%)\n\n")
            
            f.write("[Match Details]\n")
            f.write(f"Single-token matches: {total_single_matches:,}\n")
            f.write(f"Multi-token sequence matches: {total_multi_matches:,}\n")
            
            if multi_token_stats:
                f.write("\nMulti-token match stats:\n")
                sorted_stats = sorted(multi_token_stats.items(), key=lambda x: x[1], reverse=True)
                for string, count in sorted_stats:
                    f.write(f"  '{string}': {count:,} times\n")
        
        f.write("\n[Save Strategy Notes]\n")
        f.write("- reasoning activations: full save to HDF5 (for distribution analysis)\n")
        f.write("- all/normal activations: save mean vectors only to .npy (space-saving)\n\n")
        
        f.write("[Output Files]\n")
        for file_path in saved_files:
            f.write(f"  - {file_path}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Saved processing log: {log_path}")
    
    # ============================================
    # 7. Summary output
    # ============================================
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    
    # Print file info
    print("\nGenerated files:")
    
    # HDF5 files (full reasoning data)
    # if h5_writers:
    #     print("\n[Reasoning Full Data - HDF5]")
    #     for key, writer_info in h5_writers.items():
    #         file_path = Path(writer_info['file'].filename)
    #         final_count = writer_info['count']
    #         file_size = file_path.stat().st_size / (1024 ** 3)  # GB
    #         print(f"  - {file_path.name}: {final_count:,} tokens ({file_size:.2f} GB)")
    
    # Mean vector files (all/normal)
    if mean_files:
        print("\nAll/Normal Mean Vectors - NumPy")
        for file_path in mean_files:
            path = Path(file_path)
            file_size = path.stat().st_size / (1024 ** 2)  # MB
            mean_vec = np.load(file_path)
            print(f"  - {path.name}: shape={mean_vec.shape} ({file_size:.2f} MB)")
    
    # Other files
    print("\nOther files")
    for file_path in saved_files:
        if not str(file_path).endswith('.h5') and not str(file_path).endswith('_mean.npy'):
            print(f"  - {file_path}")
    print(f"  - {log_path}")
    
    print("\nData stats")
    print(f"  Total tokens: {total_tokens:,}")
    
    if single_token_ids is not None or multi_token_sequences is not None:
        print(f"  Reasoning tokens: {total_reasoning_tokens:,} ({total_reasoning_tokens/total_tokens*100:.2f}%)")
        print(f"  Normal tokens: {total_normal_tokens:,} ({total_normal_tokens/total_tokens*100:.2f}%)")
        print(f"\n  Single-token matches: {total_single_matches:,}")
        print(f"  Multi-token sequence matches: {total_multi_matches:,}")
    
    # Print read examples
    print(f"\n{'='*60}")
    print("Read examples:")
    print(f"{'='*60}")
    
    # Reading full reasoning data
    # if h5_writers:
    #     print("\n# ========== Read full reasoning data (for distribution analysis) ==========")
    #     print(f"\nimport h5py")
    #     print(f"import numpy as np")
        
    #     if 'reasoning_sae' in h5_writers:
    #         print("\n# Read reasoning SAE activations:")
    #         print(f"with h5py.File('{output_dir / 'sae_activations_reasoning.h5'}', 'r') as f:")
    #         print("    reasoning_acts = f['activations'][:]  # Full load")
    #         print(f"    print(f'Shape: {{reasoning_acts.shape}}')  # ({total_reasoning_tokens}, {config_info['d_sae']})")
    #         print("    # Or read in chunks")
    #         print(f"    chunk = f['activations'][1000:2000]")
        
    #     if args.save_hidden_states and 'reasoning_hs' in h5_writers:
    #         print("\n# Read reasoning hidden states:")
    #         print(f"with h5py.File('{output_dir / 'hidden_states_reasoning.h5'}', 'r') as f:")
    #         print(f"    reasoning_hs = f['hidden_states'][:]")
    #         print(f"    print(f'Shape: {{reasoning_hs.shape}}')  # ({total_reasoning_tokens}, {config_info['d_in']})")
    
    # Mean vector reads
    if mean_files:
        print("\n# ========== Read all/normal mean vectors ==========")
        print(f"\nimport numpy as np")
        
        if not args.only_reasoning:
            print("\n# Read mean vector for all SAE activations:")
            print(f"all_mean = np.load('{output_dir / 'sae_activations_all_mean.npy'}')")
            print(f"print(f'All mean shape: {{all_mean.shape}}')  # ({config_info['d_sae']},)")
            print(f"print(f'Computed from {total_tokens:,} tokens')")
        
        if args.separate_reasoning and not args.only_reasoning:
            print("\n# Read mean vector for normal SAE activations:")
            print(f"normal_mean = np.load('{output_dir / 'sae_activations_normal_mean.npy'}')")
            print(f"print(f'Normal mean shape: {{normal_mean.shape}}')  # ({config_info['d_sae']},)")
            print(f"print(f'Computed from {total_normal_tokens:,} tokens')")
        
        if args.save_hidden_states:
            print("\n# Read mean vectors for hidden states:")
            if not args.only_reasoning:
                print(f"all_hs_mean = np.load('{output_dir / 'hidden_states_all_mean.npy'}')")
                print(f"print(f'All HS mean shape: {{all_hs_mean.shape}}')  # ({config_info['d_in']},)")
            
            if args.separate_reasoning and not args.only_reasoning:
                print(f"normal_hs_mean = np.load('{output_dir / 'hidden_states_normal_mean.npy'}')")
                print(f"print(f'Normal HS mean shape: {{normal_hs_mean.shape}}')  # ({config_info['d_in']},)")
    
    # Config read example
    print("\n# ========== Read config ==========")
    print(f"\nimport json")
    print(f"with open('{config_path}', 'r') as f:")
    print(f"    config = json.load(f)")
    print(f"    print('d_model:', config['d_in'])")
    print(f"    print('sae_features:', config['d_sae'])")
    print(f"    print('total_tokens:', config['total_tokens'])")
    print(f"    print('save_strategy:', config['save_strategy'])")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


        

import json
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login

def load_and_process_lmsys(tokenizer_path, output_dir, max_samples=None):
    """
    Load the lmsys-chat-1m dataset from HuggingFace,
    filter language="English" and turn<3 samples,
    and use the tokenizer chat template to process conversations.
    """
    
    print("\n" + "="*50)
    print("Processing lmsys-chat-1m dataset")
    print("="*50)
    
    # 1. Load tokenizer
    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return []
    
    # 2. Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train"
    )
    print(f"Original dataset size: {len(dataset)}")
    
    # 3. Filter: language="English" and turn<3
    print("\nFiltering data...")
    print("Filter condition: language='English' AND turn < 3")
    
    def filter_function(example):
        language = example.get("language", "")
        # conversation length indicates the number of turns
        conversation = example.get("conversation", [])
        turn = example.get("turn", 1)
        
        return language == "English" and turn < 6
    
    filtered_dataset = dataset.filter(filter_function)
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Filter ratio: {len(filtered_dataset)/len(dataset)*100:.2f}%")
    
    # If a max sample count is specified, truncate
    if max_samples and max_samples < len(filtered_dataset):
        filtered_dataset = filtered_dataset.select(range(max_samples))
        print(f"Taking the first {max_samples} samples")
    
    # 4. Process data
    print("\nProcessing data...")
    processed_data = []
    failed_count = 0
    
    for idx, example in enumerate(filtered_dataset):
        try:
            # Extract fields
            conversation = example.get("conversation", [])
            language = example.get("language", "")
            turn = len(conversation)
            
            if not conversation:
                failed_count += 1
                continue
            
            # Use tokenizer chat template
            text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Create new data item
            processed_item = {
                "text": text,
                # "conversation": conversation,
                # "language": language,
                "turn": turn,
                # "model": example.get("model", ""),
                # "conversation_id": example.get("conversation_id", ""),
                # "timestamp": example.get("timestamp", "")
            }
            
            processed_data.append(processed_item)
            
            # Print progress
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1}/{len(filtered_dataset)} records (failed: {failed_count})")
        
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Only print the first 5 errors
                print(f"Error processing record {idx}: {e}")
            continue
    
    print(f"\nProcessing complete: succeeded {len(processed_data)}, failed {failed_count}")
    
    # 5. Stats
    print("\n=== Data Stats ===")
    turn_1 = sum(1 for item in processed_data if item["turn"] == 1)
    turn_2 = sum(1 for item in processed_data if item["turn"] == 2)
    print(f"Turn = 1 records: {turn_1}")
    print(f"Turn = 2 records: {turn_2}")
    
    # 6. Save as JSONL
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "lmsys_chat_processed.jsonl")
    print(f"\nSaving data to {output_jsonl}...")
    
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(processed_data)} records to {output_jsonl}")
    
    # 7. Save one sample as JSON
    if processed_data:
        sample_file = os.path.join(output_dir, "lmsys_sample_turn1.json")
        turn_1_data = [item for item in processed_data if item["turn"] == 1]
        if len(turn_1_data) == 0:
            turn_1_data = processed_data[0]
        else:
            turn_1_data = turn_1_data[0]
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(turn_1_data, f, ensure_ascii=False, indent=2)
        print(f"Saved sample data to {sample_file}")
        
        # If there are turn=2 records, also save a sample
        turn_2_data = [item for item in processed_data if item["turn"] == 2]
        if turn_2_data:
            sample_file_turn2 = os.path.join(output_dir, "lmsys_sample_turn2.json")
            with open(sample_file_turn2, "w", encoding="utf-8") as f:
                json.dump(turn_2_data[0], f, ensure_ascii=False, indent=2)
            print(f"\nSaved turn=2 sample data to {sample_file_turn2}")
    
    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Process lmsys-chat-1m: filter English language and turn<3 conversations."
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Tokenizer path or name, e.g., Qwen/Qwen2.5-7B-Instruct."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_lmsys",
        help="Output directory for processed data (default: ./processed_lmsys)."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples for testing (default: process all data)."
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets/models. If omitted, use cached token or HF_TOKEN."
    )

    args = parser.parse_args()

    # HuggingFace authentication
    print("="*50)
    print("HuggingFace authentication")
    print("="*50)

    try:
        if args.hf_token:
            print("Logging in with the provided HF token...")
            login(token=args.hf_token)
            print("✓ HuggingFace login successful")
        else:
            # Try environment variable or cached token
            hf_token_env = os.environ.get("HF_TOKEN")
            if hf_token_env:
                print("Logging in with HF_TOKEN from environment...")
                login(token=hf_token_env)
                print("✓ HuggingFace login successful")
            else:
                print("No HF token provided; trying cached credentials...")
                # If no token is provided, try cached credentials
                try:
                    login()
                    print("✓ Logged in with cached credentials")
                except Exception:
                    print("⚠ Warning: HF token not found; gated datasets may fail to load")
                    print("  You can provide a token via:")
                    print("  1. Use the --hf-token argument")
                    print("  2. Set the HF_TOKEN environment variable")
                    print("  3. Run 'huggingface-cli login' for interactive login")
    except Exception as e:
        print(f"⚠ HuggingFace login warning: {e}")
        print("Continuing to load the dataset...")

    print("\n" + "="*50)
    print("lmsys-chat-1m dataset processing script")
    print("="*50)
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Output directory: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("Filter condition: language='English' AND turn < 3")

    try:
        processed_data = load_and_process_lmsys(
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        
        print("\n" + "="*50)
        print("Processing complete!")
        print("="*50)
        print(f"Processed files saved in: {args.output_dir}")
        print(f"- lmsys_chat_processed.jsonl: {len(processed_data)} records")
        print("- lmsys_sample.json: sample data (turn=1)")
        print("- lmsys_sample_turn2.json: sample data (turn=2, if available)")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

import json
import argparse
from pathlib import Path
from datasets import load_dataset

def load_and_process_dataset(prompt_file_path, output_dir):
    """
    Load the OpenThoughts-114k dataset from HuggingFace,
    filter the math domain, and format it into the target schema.
    """
    
    # 1. Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "open-thoughts/OpenThoughts-114k",
        "metadata",
        split="train"
    )
    print(f"Original dataset size: {len(dataset)}")
    
    # 2. Load prompt file
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            PROMPT = f.read()
        print(f"Loaded prompt successfully: {PROMPT[:100]}...")
    except FileNotFoundError as e:
        print(f"Error: prompt file not found: {prompt_file_path}")
        raise e
    
    # 3. Define chat template
    chat_template = """<｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜><think>
{assistant_reasoning}
<\think>
{assistant_solution}<｜end▁of▁sentence｜>"""

    # 4. Filter domain="math"
    print("Filtering math-domain data...")
    filtered_dataset = dataset.filter(lambda x: x.get("domain") == "math")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    # 5. Process data and build the text field
    print("Processing data...")
    processed_data = []
    
    for idx, example in enumerate(filtered_dataset):
        # Extract fields
        problem = example.get("problem", "")
        deepseek_reasoning = example.get("deepseek_reasoning", "")
        deepseek_solution = example.get("deepseek_solution", "")
        domain = example.get("domain", "")
        
        # Assemble text
        text = chat_template.format(
            system="",
            user=problem + PROMPT,
            assistant_reasoning=deepseek_reasoning,
            assistant_solution=deepseek_solution
        )
        
        # Create new data item
        processed_item = {
            "text": text,
            # "problem": problem,
            # "deepseek_reasoning": deepseek_reasoning,
            # "deepseek_solution": deepseek_solution,
            # "domain": domain
        }
        
        processed_data.append(processed_item)
        
        # Print progress
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(filtered_dataset)} records")
    
    # 6. Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 7. Save as JSONL
    output_jsonl = output_path / "openthoughts_math_processed.jsonl"
    print(f"\nSaving data to {output_jsonl}...")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed_data)} records to {output_jsonl}")

    # 8. Save one sample as JSON
    if processed_data:
        sample_file = output_path / "sample_data.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(processed_data[0], f, ensure_ascii=False, indent=2)
        print(f"Saved sample data to {sample_file}")
        
        # Print sample preview
        print("\n=== Sample Preview ===")
        print(f"Text field length: {len(processed_data[0]['text'])} chars")
        print(f"First 500 chars of text:\n{processed_data[0]['text'][:500]}...")
    
    return processed_data

def main():
    parser = argparse.ArgumentParser(
        description="Load OpenThoughts-114k from HuggingFace and process the math domain."
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to the prompt file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for processed data (default: ./output)."
    )

    args = parser.parse_args()

    try:
        processed_data = load_and_process_dataset(args.prompt_file, args.output_dir)
        print(f"\nDone. Processed {len(processed_data)} records.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

import os
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig
from huggingface_hub import login

# HuggingFace authentication
# print("=" * 50)
# print("HuggingFace authentication")
# print("=" * 50)

# try:
#     hf_token = os.environ.get("HF_TOKEN")
#     if hf_token:
#         print("Logging in with HF_TOKEN from environment...")
#         login(token=hf_token)
#         print("✓ HuggingFace login successful")
#     else:
#         print("HF_TOKEN not found; trying cached credentials...")
#         try:
#             login()
#             print("✓ Logged in with cached credentials")
#         except Exception:
#             print("⚠ Warning: HF token not found")
#             print("  You can provide a token via:")
#             print("  1. Set the HF_TOKEN environment variable")
#             print("  2. Run 'huggingface-cli login' for interactive login")
#             raise
# except Exception as e:
#     print(f"❌ HuggingFace login failed: {e}")
#     print("If you need to upload to HuggingFace, log in first.")
#     exit(1)

print("\n" + "=" * 50)
print("Starting dataset pretokenization")
print("=" * 50)

# tokenizer_name = "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer_name = "/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# dataset_file = "/dataset/processed_openthoughs/openthoughts_math_processed.jsonl"
# dataset_file = "/SAE-Reasoning/vllm_sae_evaluation/outputs/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/sampling_16/math_500/math500_processed.jsonl"
dataset_file = "/SAE-Reasoning/vllm_sae_evaluation/outputs/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/sampling_16/math_500/math500_processed.jsonl"
# hf_repo_id = "akaifun/openthoughs-tokenized-deepseek-qwen-7b"
# hf_repo_id = "akaifun/lmsys-tokenized-deepseek-qwen-7b"
# save_path = "/SAE-Reasoning/vllm_sae_evaluation/outputs/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/sampling_16/tokenized-deepseek-qwen-7b"
save_path = "/SAE-Reasoning/vllm_sae_evaluation/outputs/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/sampling_16/tokenized-deepseek-llama-8b"
# save_path = "/dataset/lmsys-tokenized-deepseek-llama-8b"
# save_path = "/dataset/openthoughts-tokenized-deepseek-llama-8b"
# save_path = "/dataset/openthoughts-tokenized-deepseek-qwen-7b"


cfg = PretokenizeRunnerConfig(
    tokenizer_name=tokenizer_name,
    dataset_path="json",  # Specify dataset format as json
    data_files=[dataset_file],  # Use data_files to specify a local file
    shuffle=True,
    num_proc=8,  # increase this number depending on how many CPUs you have
    # tweak these settings depending on the model
    context_size=1024,
    # Use identifiers instead of raw token strings.
    # "bos" maps to tokenizer.bos_token_id (151646)
    # "eos" maps to tokenizer.eos_token_id (151643)
    begin_batch_token="bos",  # Token that begins each batch
    begin_sequence_token=None,  # Token that begins each sequence
    sequence_separator_token="eos",  # Separator token between sequences
    seed=42,
    column_name="text",
    # uncomment to upload to huggingface
    # hf_repo_id="your-username/c4-10k-tokenized-gpt2"
    # uncomment to save the dataset locally
    save_path=save_path
    # hf_repo_id=hf_repo_id
)

dataset = PretokenizeRunner(cfg).run()

# Calculate and print total number of tokens
total_tokens = 0
for example in dataset:
    total_tokens += len(example['input_ids'])

print(f"Total tokens in dataset: {total_tokens}")

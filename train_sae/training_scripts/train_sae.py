import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

total_training_steps = 30_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

# Set training hyperparameters per the paper:
# - LR: linearly decay to 0 over the last 20% of steps
# - L1 coeff: linearly warm up from 0 to 5 over the first 5% of steps
# - Gradient clipping: clip to 1
lr_warm_up_steps = 0  # No LR warmup
lr_decay_steps = total_training_steps // 5  # LR decay over last 20% of steps (6000 steps)
l1_warm_up_steps = total_training_steps // 20  # L1 warmup over first 5% of steps (1500 steps)
wandb_project="lmsys-openthoughts-1B-deepseek-qwen-7b-l20-resid_post"
# wandb_project="lmsys-openthoughts-1B-deepseek-llama-8b-resid_post"
cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # our model
    # model_name="/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    model_class_name="AutoModelForCausalLM",  # Load with HuggingFace AutoModelForCausalLM
    hook_name="model.layers.0",  # HF hook name format (residual stream at each layer output)
    dataset_path="json",  # Load local files in JSON format
    # data_files=["/dataset/lmsys-openthoughts-1B-tokenized-deepseek-llama-8b/combined_1.00b_ratio50.jsonl"],
    data_files=["/dataset/lmsys-openthoughts-1B-tokenized-deepseek-qwen-7b/combined_1.00b_ratio50.jsonl"],
    is_dataset_tokenized=True,
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    sae=StandardTrainingSAEConfig(
        # d_in=4096,  # DeepSeek-R1-Distill-Qwen-7B hidden size is 3584
        d_in=3584,
        d_sae=65536,  # the width of the SAE. Larger will result in better stats but slower training.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_activations="expected_average_only_in",
        l1_coefficient=5,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    ),
    # Training Parameters
    # Per paper: LR decays linearly to 0 in the last 20% of steps
    lr=5e-5,  # Initial learning rate
    lr_end=1e-10,  # LR decays close to 0 (cannot be 0.0 or code errors)
    adam_beta1=0.9,  # Adam beta1
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # Use constant + linear decay for linear schedule
    lr_warm_up_steps=lr_warm_up_steps,  # No warmup
    lr_decay_steps=lr_decay_steps,  # Linear decay in last 20% of steps
    train_batch_size_tokens=batch_size,
    context_size=1024,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=16,  # Reduce buffer size to save VRAM (was 64)
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=8,  # Reduce batch size to save VRAM (was 16)
    act_store_device="cpu",  # Place activation store on CPU to save GPU VRAM
    # Resampling protocol
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # SwanLab logging
    logger=LoggingConfig(
        log_to_wandb=True,  # Enable logging (uses swanlab in practice)
        wandb_project=wandb_project,
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),
    # Misc
    device=device,
    seed=42,
    n_checkpoints=5,
    checkpoint_path="checkpoints/standard_sae",
    save_final_checkpoint=True,
    dtype="float32",
    autocast=True,
    autocast_lm=True,
    # Model load parameters
    model_from_pretrained_kwargs={
        "dtype": "bfloat16",  # Use bfloat16 to save VRAM (string avoids JSON issues)
    },
)
# look at the next cell to see some instruction for what to do while this is running.
sparse_autoencoder = SAETrainingRunner(cfg).run()

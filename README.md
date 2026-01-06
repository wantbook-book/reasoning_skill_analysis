<h1 align="center"> Rethinking Performance Gains in Long-Reasoning Models: From Critical Tokens to Reasoning Skills</h1>


Official codebase for the paper  
📄 **"Rethinking Performance Gains in Long-Reasoning Models: From Critical Tokens to Reasoning Skills"**. This SAE training code is based on the [SAELens](https://github.com/decoderesearch/SAELens) framework, and the evaluation code is based on the [Math-Verify](https://github.com/huggingface/Math-Verify) and [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro).

## 🧾 Overview

**Abstract:** 

Large reasoning models (LRMs) have recently achieved unprecedented gains on complex language tasks, largely due to structured chain-of-thought (CoT) trajectories that invoke diverse reasoning skills. Prior analyses identify critical tokens that steer these trajectories and correlate with stronger reasoning, suggesting that different *critical tokens* elicit different underlying *reasoning skills*. Yet these token-level findings are rarely grounded in an explicit skill taxonomy, leaving it unclear which skills are engaged and which ultimately drive the gains.

Here, we bridge *critical tokens* to *reasoning skills* by organizing token-anchored phrases into skill-specific lexicons, enabling a systematic study of how distinct skills contribute to performance.

Using a suite of activation- and logit-based steering methods to modulate skill usage, we find that:
- (1) reasoning performance is driven by the skills exercised during generation rather than by output length;
- (2) overuse of skills miscalibrates response distribution variance and degrades performance;
- (3) optimal performance requires task-specific skill configurations.

Overall, our findings suggest using skill usage as an RLVR auxiliary reward and as a handle for measuring and controlling sampling diversity.

## 🧭 Contents

- 🗂️ [Directory Overview](#directory-overview)
- ⚙️ [Installation](#installation)
- 🧩 [Reasoning-Skill Lexicon Extraction](#reasoning-skill-lexicon-extraction)
  - 📊 [N-gram Frequency Analysis](#n-gram-frequency-analysis)
  - 🤖 [API Classification](#api-classification)
  - 🧑‍🔬 [Human Review](#human-review)
- 🧠 [SAE Training](#sae-training)
  - 🧹 [Preprocess Dataset and Tokenize](#preprocess-dataset-and-tokenize)
  - 🚀 [Training](#training)
- 🎛️ [Steering Layer Selection](#steering-layer-selection)
- 🧱 [Steering Vector Construction](#steering-vector-construction)
  - 🧷 [SAE Steering Feature Selection](#sae-steering-feature-selection)
  - 🧮 [HS Steering Vector Construction](#hs-steering-vector-construction)
- 📏 [Steering Base Scale Computation](#steering-base-scale-computation)
  - 📐 [SAE-Steer Base Scale](#sae-steer-base-scale)
  - 🧪 [HS-Steer Base Scale](#hs-steer-base-scale)
- 🔧 [Intervention Inference](#intervention-inference)
  - 🧠 [SAE-Steer](#sae-steer)
  - 🧲 [HS-Steer](#hs-steer)
  - ⚡ [Logit-Boost](#logit-boost)
- 🧪 [Evaluation](#evaluation)
- 🔎 [Analysis](#analysis)
  - 🧵 [SAE Textual Activation Visualization](#sae-textual-activation-visualization)
  - 📉 [Logit Change Across Layers Visualization](#logit-change-across-layers-visualization)
- ❓ [FAQs](#faqs)

## 🗂️ Directory Overview

- `train_sae/`: SAE training (based on SAELens).
- `steering_inference/`: Evaluation and steering scripts (vLLM-based).
- `analysis/`: Visualizations for SAE activation text spans and the effects of steering vectors on logit changes.
- `steering_setups/`: Skill-specific lexicon construction, steering layer selection, and steering vector construction.

## ⚙️ Installation

We recommend using Python 3.12. The environment has been tested on Ubuntu 22.04.

```bash
pip install -r requirements.txt
```

## 🧩 Reasoning-Skill Lexicon Extraction

### 📊 N-gram Frequency Analysis

Script: `steering_setups/lexicon_construction/analyse_word_frequency.py`

Parameters:
- `--file1`: First JSONL input file. Each line should contain a `code` field (list of strings). [TODO: confirm schema]
- `--file2`: Second JSONL input file (same schema as `--file1`).
- `--output_dir`: Output directory for charts and JSON summaries.
- `--top_k`: Top-K terms to keep per n-gram (default: 30).
- `--min_count`: Minimum frequency for n-grams (default: 2).

Notes:
- As described in the paper, the script computes 1- to 5-gram statistics.

Script template:
```bash
FILE1=long_reasoning_think_resp
FILE2=short_reasoning_resp
TOP_K=20000
MIN_COUNT=2
OUTPUT_DIR=YOUR_DIR
python analyse_word_frequency.py \
    --file1 $FILE1 \
    --file2 $FILE2 \
    --output_dir $OUTPUT_DIR \
    --top_k $TOP_K \
    --min_count $MIN_COUNT
```

### 🤖 API Classification

Script: `steering_setups/lexicon_construction/classify_lexicon_api.py`

Parameters:
- `--prompt`: Prompt template file path.
- `--phrases`: Input phrases file, one phrase per line.
- `--out`: Output JSONL path.
- `--model`: Model name (default: `gpt-5-mini`).
- `--batch-size`: Phrases per API call (default: 200).
- `--temperature`: Sampling temperature (default: 0.0).
- `--max-output-tokens`: Max output tokens per call (default: 8000; set <= 0 to disable).
- `--placeholder`: Placeholder string in the prompt template (default: `<<<PASTE YOUR PHRASES HERE, ONE PER LINE>>>`).
- `--resume`: Resume from existing output by line count.

Notes:
- We recommend `--temperature 0.1` for classification.

Script template:

```bash
python classify_lexicon_api.py \
    --prompt vllm_sae_evaluation/prompts/classify_reasoning_lexicons_gpt.txt \
    --phrases N_GRAM_FRASES_OUTPUT \
    --out $OUT_JSONL \
    --model API_MODEL_NAME \
    --batch-size BATCH_SIZE \
    --temperature TEMP \
    --max-output-tokens MAX_OUTPUT_TOKENS
```

### 🧑‍🔬 Human Review

Keep lexicon items where multiple model labels agree, then manually review the remaining phrases. We retain phrases that reliably cue the targeted reasoning skill and also exhibit similar cueing behavior on non-reasoning tasks, verified via sampled inspection on the LMSYS-Chat-1M dataset.

## 🧠 SAE Training

### 🧹 Preprocess Dataset and Tokenize

#### 📘 OpenThoughts math subset

Script: `train_sae/training_scripts/process_openthoughts.py`

Parameters:
- `--prompt-file`: Path to a prompt template file (required).
- `--output-dir`: Output directory for processed JSONL (default: `./output`).

Notes:
- Downloads `open-thoughts/OpenThoughts-114k` and filters `domain == "math"`.
- Output files: `openthoughts_math_processed.jsonl` and `sample_data.json`.
- Prompt file: use `vllm_sae_evaluation/prompts/math_cot.txt`

Script template:
```bash
python process_openthoughts.py \
    --prompt-file vllm_sae_evaluation/prompts/math_cot.txt \
    --output-dir YOUR_DIR
```

#### 💬 LMSYS conversations

Script: `train_sae/training_scripts/process_lmsys.py`

Parameters:
- `--tokenizer-path`: HF tokenizer path or name (required).
- `--output-dir`: Output directory (default: `./processed_lmsys`).
- `--max-samples`: Optional cap for debugging.
- `--hf-token`: HuggingFace token; if omitted, uses `HF_TOKEN` env or cached login.

Notes:
- Filters `language == "English"` and `turn < 6`.
- Output files: `lmsys_chat_processed.jsonl`, `lmsys_sample_turn1.json`, `lmsys_sample_turn2.json`.

Script template:
```bash
# Use a tokenizer that supports apply_chat_template for conversation formatting.
python process_lmsys.py \
    --tokenizer-path MODEL_TOKENIZER \
    --output-dir YOUR_DIR \
    --hf-token YOUR_HF_TOKEN
```

### Pre-tokenization

Use `train_sae/training_scripts/pretokenizing_datasets.py`.

### 🚀 Training

Script: `SAELens/my_training/train_sae.py`

This script does not expose CLI flags. Edit the config values in the file before running.

Key parameters to set:
- `model_name`: Base model path.
- `hook_name`: Hook point name (e.g., `model.layers.0`).
- `data_files`: List of tokenized JSONL files.
- `d_in`, `d_sae`: SAE dimensions.
- `l1_coefficient`, `l1_warm_up_steps`: Sparsity configuration.
- `lr`, `lr_end`, `lr_decay_steps`, `lr_scheduler_name`: Learning rate schedule.
- `train_batch_size_tokens`, `context_size`, `training_tokens`: Training scale.
- `n_batches_in_buffer`, `store_batch_size_prompts`, `act_store_device`: Activation store settings.
- `checkpoint_path`, `n_checkpoints`, `save_final_checkpoint`: Checkpoint settings.
- `wandb_project` and `logger` settings.

Recommended parameters:
See `SAELens/my_training/train_sae.py` for the recommended hyperparameters.

## 🎛️ Steering Layer Selection

1. Compute SAE and HS representations on MATH responses.

   Script: `steering_setups/steering_layer_selection/compute_sae_activations.py`

   Parameters (required):
   - `--sae-path`: Local SAE path.
   - `--dataset`: Dataset path (JSONL or Arrow directory).
   - `--output`: Output directory.

   Parameters (model):
   - `--model-path`: Override LM path (optional).
   - `--local-model`: Use local files only.
   - `--hook-name`: Override hook point name.

   Parameters (dataset):
   - `--use-arrow`: Interpret `--dataset` as Arrow format.
   - `--batch-size`: Batch size (default: 8).
   - `--max-length`: Max sequence length.
   - `--max-samples`: Max number of samples.

   Parameters (reasoning tokens):
   - `--reasoning-tokens`: JSON file of reasoning token strings.
   - `--separate-reasoning`: Save reasoning and normal tokens separately.
   - `--only-reasoning`: Save only reasoning tokens.
   - `--save-token-masks`: Save token-type masks.

   Parameters (hidden states):
   - `--save-hidden-states`: Save raw hidden states.

   Parameters (output):
   - `--save-token-level`: Save token-level activations.
   - `--no-save-npy`: Disable NumPy output.
   - `--no-save-h5`: Disable HDF5 output.
   - `--normal_sample_limit`: Save up to N normal-token vectors.

   Parameters (device):
   - `--device`: `cuda` or `cpu` (auto if omitted).
   - `--dtype`: `float32`, `float16`, or `bfloat16`.

   Parameters (misc):
   - `--verbose`: Verbose logging.

   Notes:
   - `--only-reasoning` and `--separate-reasoning` require `--reasoning-tokens`.
   - Do not set both `--no-save-npy` and `--no-save-h5`.

    Script template:
    ```bash
    # We sample 500 questions from the MATH training set and use an LRM to generate responses.
    DATASET=YOUR_SAMPLED_DATASET
    # Used later to compute HS steering base scale by projecting non-reasoning (or non-skill) token representations onto the HS steering vector.
    NORMAL_SAMPLE_LIMIT=10000
    # Compute separately for each reasoning skill.
    python compute_sae_activations.py \
        --sae-path YOUR_SAE \
        --dataset $DATASET \
        --output YOUR_OUTPUT_DIR \
        --reasoning-tokens SAELens/my_training/reasoning_tokens/{skill_name}.json \
        --separate-reasoning \
        --save-hidden-states \
        --local-model \
        --batch-size 48 \
        --model-path LRM_PATH \
        --hook-name model.layers.{layer_idx} \
        --normal_sample_limit $NORMAL_SAMPLE_LIMIT
    ```


2. Count active dimensions using contrast vectors and thresholds.

   Script: `steering_setups/steering_layer_selection/draw_layer_neuron_dist.py`

   Parameters:
   - `--base_dir`: Template path containing layer outputs, with `{layer}` placeholder.
   - `--output_dir`: Output directory for plots.
   - `--reasoning_names`: Reasoning subdirectories to process.
   - `--think_only_base`: Template path for think-only outputs (optional; requires `--resp_only_base`).
   - `--resp_only_base`: Template path for resp-only outputs (optional; requires `--think_only_base`).
   - `--comparison_label`: Label for think-only minus resp-only plots.
   - `--font_family`, `--label_fontsize`, `--title_fontsize`, `--legend_fontsize`, `--tick_fontsize`: Plot typography.
   - `--sae_color`, `--sae_linestyle`, `--sae_marker`, `--sae_markersize`, `--sae_linewidth`: SAE plot styling.
   - `--hs_color`, `--hs_linestyle`, `--hs_marker`, `--hs_markersize`, `--hs_linewidth`: HS plot styling.
   - `--save_pdf`: Also export PDF.
   - `--layers`: List of layers to process (e.g., `0 5 10 15`).

   Script template:
   ```bash
   # Lexicon-level, skill-specific.
    python draw_layer_neuron_dist.py \
        --base_dir YOUR_ACTIVATION_OUTPUT_DIR \
        --output_dir YOUR_OUTPUT_DIR \
        --label_fontsize 34 \
        --legend_fontsize 32 \
        --tick_fontsize 34 \
        --sae_color '#FFA269' \
        --sae_linestyle '-' \
        --sae_linewidth 8 \
        --hs_linestyle '-' \
        --hs_linewidth 8 \
        --sae_marker 'o' \
        --hs_color '#BDE7D8' \
        --hs_marker 's' \
        --hs_markersize 20 \
        --save_pdf \
        --layers LAYER_LIST # e.g. 0 5 10 15 20 25 31
    # Output-level.
    python draw_layer_neuron_dist.py \
        --reasoning_names REASONING_NAMES \
        --think_only_base YOUR_LRM_THINK_ACTIVATION_OUTPUT_DIR \
        --resp_only_base YOUR_SRM_RESP_ACTIVATION_OUTPUT_DIR \
        --output_dir YOUR_OUTPUT_DIR \
        --label_fontsize 34 \
        --legend_fontsize 32 \
        --tick_fontsize 34 \
        --sae_color '#FFA269' \
        --sae_linestyle '-' \
        --sae_linewidth 8 \
        --hs_linestyle '-' \
        --hs_linewidth 8 \
        --sae_marker 'o' \
        --hs_color '#BDE7D8' \
        --hs_marker 's' \
        --hs_markersize 20 \
        --save_pdf \
        --layers LAYER_LIST # e.g. 0 5 10 15 20 25 31
   ```

## 🧱 Steering Vector Construction

Use `steering_setups/steering_vector_construction/generate_hs_steering_vector.py` to compute contrast vectors and (optionally) normalized HS steering vectors. Parameters are listed below under [HS Steering Vector Construction](#hs-steering-vector-construction).

### 🧷 SAE Steering Feature Selection

#### 📐 Contrast Vector and Relative Vector Threshold Selection

Script: `steering_setups/steering_vector_construction/select_sae_features.py`

Parameters:
- `--base_dir`: Template path containing layer outputs, with `{layer}` placeholder.
- `--reasoning_name`: Reasoning subdirectory name.
- `--think_only_base`: Template dir for resp_think_only outputs (requires `--resp_only_base`).
- `--resp_only_base`: Template dir for resp_only outputs (requires `--think_only_base`).
- `--layers`: Comma-separated list of layers (e.g., `0,5,10`).
- `--delta_threshold`: Threshold on (normalized) delta_sae.
- `--no_normalize_delta`: Disable delta normalization.
- `--activation_percentile`: Percentile for reasoning activation filter.
- `--activation_scale`: Scaling factor for activation threshold.
- `--relative_percentile`: Percentile for relative threshold.
- `--eps`: Epsilon for relative calculation.
- `--output_dir`: Output directory.
- `--output_name`: Output JSON file name.

Script template:
```bash
python select_sae_features.py \
    --base_dir YOUR_LRM_ACTIVATION_OUTPUT_DIR \
    --reasoning_name "deep_reasoning" \
    --layers "20" \
    --delta_threshold 1.0 \
    --activation_percentile 90.0 \
    --activation_scale 0.1 \
    --relative_percentile 95.0 \
    --eps 1e-6 \
    --output_dir YOUR_OUTPUT_DIR \
    --output_name YOUR_OUTPUT_JSON_NAME # e.g. selected_sae_features.json
```

#### 🧭 K-Means Cluster Selection

Script: `steering_setups/steering_vector_construction/sae_cluster_analysis.py`

Parameters:
- `--selection_path`: Path to selected feature JSON.
- `--base_dir`: Template path containing layer outputs, with `{layer}` placeholder.
- `--reasoning_name`: Reasoning subdirectory name.
- `--think_only_base`: Template dir for resp_think_only outputs (requires `--resp_only_base`).
- `--resp_only_base`: Template dir for resp_only outputs (requires `--think_only_base`).
- `--layers`: Comma-separated list of layers.
- `--n_clusters`: Number of clusters.
- `--normalize_delta`: Normalize delta before ranking.
- `--sae_path_template`: SAE checkpoint path template with `{layer}` placeholder.
- `--output_dir`: Output directory.
- `--output_name`: Output JSON file name.
- `--font_family`, `--label_fontsize`, `--tick_fontsize`: Plot typography.
- `--cluster_cmap`, `--cluster_point_size`: Cluster plot styling.
- `--heatmap_cmap`: Similarity heatmap colormap.
- `--similarity_vmin`, `--similarity_vmax`: Heatmap scale bounds.
- `--cluster_cbar_labelsize`, `--cluster_cbar_ticksize`: Cluster colorbar styling.
- `--heatmap_cbar_labelsize`, `--heatmap_cbar_ticksize`: Heatmap colorbar styling.
- `--tsne_perplexity`: t-SNE perplexity.
- `--umap_n_neighbors`, `--umap_min_dist`, `--umap_metric`: UMAP settings.
- `--save_pdf`: Also export PDF.

Script template:
```bash
# Example SAE path template: path/to/lmsys-openthoughts-1B-deepseek-llama-8b-l{layer}-resid_post
# Recommend: candidate_feature_num / 3 for --tsne_perplexity and --umap_n_neighbors.
python sae_cluster_analysis.py \
    --selection_path YOUR/selected_sae_features.json \
    --base_dir YOUR_LRM_ACTIVATION_PATH \
    --sae_path_template YOUR_SAE_PATH_TEMPLATE \
    --layers "20" \
    --n_clusters N_CLUSTER \
    --output_dir YOUR_OUTPUT_DIR \
    --label_fontsize 44 \
    --tick_fontsize 44 \
    --cluster_point_size 130 \
    --cluster_cbar_labelsize 38 \
    --cluster_cbar_ticksize 38 \
    --heatmap_cbar_labelsize 38 \
    --heatmap_cbar_ticksize 38 \
    --tsne_perplexity 45 \
    --umap_n_neighbors 45 \
    --umap_min_dist 0.2 \
    --umap_metric cosine \
    --save_pdf
```


#### 📈 Sweep Steering Coefficient to Select the Best Feature

Use `steering_inference/generate_with_steering.py` to sweep steering coefficients (e.g., `--alpha`, `--c_m`) and evaluate downstream performance. See [Intervention Inference](#intervention-inference) for parameter details.

### 🧮 HS Steering Vector Construction

Script: `steering_setups/steering_vector_construction/generate_hs_steering_vector.py`

Parameters:
- `--base_dir`: Template path containing layer outputs, with `{layer}` placeholder.
- `--reasoning_name`: Reasoning subdirectory name.
- `--think_only_base`: Template dir for resp_think_only outputs (requires `--resp_only_base`).
- `--resp_only_base`: Template dir for resp_only outputs (requires `--think_only_base`).
- `--layers`: Comma-separated list of layers.
- `--normalize`: Unit-normalize the steering vector.
- `--project_reasoning`: Project hidden states onto the steering vector to compute stats.
- `--output_dir`: Output directory.
- `--output_name`: Summary JSON file name.

Script template:

```bash
python generate_hs_steering_vector.py \
    --base_dir YOUR_LRM_ACTIVATION_PATH \
    --reasoning_name deep_reasoning \
    --layers 15 \
    --normalize \
    --output_dir YOUR_OUTPUT_DIR \
    --output_name hs_steering_vectors.json \
    --project_reasoning
```

## 📏 Steering Base Scale Computation

### 📐 SAE-Steer Base Scale

For the SAE-Steer base scale, `c_{i*}` is defined as the difference between the mean activation and the 99th-percentile activation of feature `i*`.

Script: `steering_setups/steering_vector_construction/compute_sae_feature_dist.py`

Parameters:
- `--base_dir`: Template path containing layer outputs, with `{layer}` placeholder.
- `--reasoning_name`: Reasoning subdirectory name.
- `--resp_think_base_dir`: Template path for resp_think_only outputs (optional).
- `--layer_features`: List of `layer:feature` pairs (required), e.g., `10:42`.
- `--output_dir`: Output directory.
- `--output_name`: Output JSON file name.

```bash
# e.g. 25:7761 25:32937
python compute_sae_feature_dist.py \
    --layer_features list_of_layer:feature_idx \
    --base_dir YOUR_ACTIVATION_PATH \
    --reasoning_name extensive_exploration \
    --output_dir YOUR_OUTPUT_DIR \
    --output_name sae_feature_stats.json
```

### 🧪 HS-Steer Base Scale

For the HS-Steer base scale, `sigma` is defined as the mean standard deviation of projections of reasoning and non-reasoning hidden states onto `r_l`. Reasoning hidden states are computed from tokens within the `<think>` span in `D_MATH^L`, while non-reasoning hidden states are computed from response tokens in `D_MATH^S`.

Script: `steering_setups/steering_vector_construction/generate_hs_steering_vector.py`

Parameters:
- Use the same parameters as in [HS Steering Vector Construction](#hs-steering-vector-construction).
- Set `--project_reasoning` to compute projection statistics.

## 🔧 Intervention Inference

Script: `steering_inference/generate_with_steering.py`

Parameters (data and output):
- `--data_names`: Comma-separated dataset names (subfolders under `--data_dir`).
- `--data_dir`: Dataset root directory.
- `--split`: Dataset split name (e.g., `test`).
- `--output_dir`: Output directory.
- `--save_outputs`: Save outputs to JSONL.
- `--seed`: Random seed.

Parameters (sampling):
- `--temperature`: Sampling temperature.
- `--top_p`: Top-p sampling (forced to 1.0 when `temperature == 0`).
- `--n_sampling`: Number of samples per prompt.
- `--max_tokens_per_call`: Max tokens per generation.

Parameters (model and parallelism):
- `--model_name_or_path`: Base model path or name.
- `--pipeline_parallel_size`: Pipeline parallel size.
- `--gpu_util`: GPU memory utilization ratio.
- `--apply_chat_template`: Apply chat template to prompts.
- `--log_level`: Logging level.

Parameters (steering):
- `--sae_path`: Local SAE path.
- `--sae_release`: SAE release name from HF Hub.
- `--sae_id`: SAE ID for HF release.
- `--feature_idx`: Feature index to steer.
- `--c_m`: Base scale for SAE steering.
- `--hs_vec_path`: Hidden-state steering vector path.
- `--hook_layer`: Layer index for the steering hook.
- `--alpha`: Steering scale factor.
- `--steer_think_only`: Apply steering only within `<think>...</think>` span.
- `--think_end_token`: End token for thinking block (default: `</think>`).

Parameters (logit boost):
- `--logit_boost_json`: JSON list of words to boost.
- `--logit_boost_bias`: Bias value applied to boosted tokens.
- `--debug_logit_boost`: Run a quick comparison on the first prompt.

### 🧠 SAE-Steer

Required/typical flags:
- `--sae_path` (or `--sae_release` + `--sae_id`).
- `--feature_idx`.
- `--c_m`.
- `--hook_layer`.
- `--alpha`.


Script template:
```bash
python generate_with_steering.py \
    --data_names aime25,aime24,gpqa_diamond,mmlu_stem,math_500 \
    --data_dir dataset/steering_eval \
    --model_name_or_path LRM_PATH \
    --output_dir YOUR_OUTPUT_DIR \
    --split test \
    --n_sampling 4 \
    --sae_path YOUR_SAE_PATH \
    --hook_layer HOOK_LAYER \
    --feature_idx FEATURE_IDX \
    --alpha ALPHA \
    --c_m SCALE_BASE \
    --save_outputs \
    --apply_chat_template \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens_per_call 32768 \
    --steer_think_only \
    --think_end_token '</think>'
```

### 🧲 HS-Steer

Required/typical flags:
- `--hs_vec_path`.
- `--hook_layer`.
- `--alpha`.

Script template:
```bash
python generate_with_steering.py \
    --data_names aime25,aime24,gpqa_diamond,mmlu_stem,math_500 \
    --data_dir dataset/steering_eval \
    --model_name_or_path LRM_PATH \
    --output_dir YOUR_OUTPUT_DIR \
    --split test \
    --n_sampling 4 \
    --hs_vec_path YOUR_HS_STEERING_VEC_PATH \
    --hook_layer HOOK_LAYER \
    --alpha ALPHA \
    --c_m SCALE_BASE \
    --save_outputs \
    --apply_chat_template \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens_per_call 32768 \
    --steer_think_only \
    --think_end_token '</think>'
```

### ⚡ Logit-Boost

Required/typical flags:
- `--logit_boost_json`.
- `--logit_boost_bias`.
- `--debug_logit_boost` (optional for sanity checks).

Script template:
```bash
python generate_with_steering.py \
    --data_names aime25,aime24,gpqa_diamond,mmlu_stem,math_500 \
    --data_dir dataset/steering_eval \
    --model_name_or_path LRM_PATH \
    --output_dir YOUR_OUTPUT_DIR \
    --split test \
    --n_sampling 4 \
    --logit_boost_json BOOST_LEXICON_JSON_PATH \
    --logit_boost_bias BOOST_BIAS \
    --save_outputs \
    --apply_chat_template \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens_per_call 32768 \
    --steer_think_only \
    --think_end_token '</think>'
```

## 🧪 Evaluation

Script: `steering_inference/evaluate_outputs.py`

Parameters:
- `--input_template`: Template path with `{data_name}`, e.g., `/path/{data_name}/file.jsonl`.
- `--data_names`: Comma-separated list of dataset names.
- `--gold_is_latex`: Use LaTeX normalization when parsing gold answers.
- `--verbose`: Enable debug logging.

Notes:
- Input JSONL should include `resps` (list of responses) and `answer`. [TODO: confirm full schema]

Script template:
```bash
# e.g. path/to/{data_name}/test_outputs.jsonl
python evaluate_outputs.py \
    --input_template INPUT_TEMPLATE \
    --data_names aime24,aime25,gpqa_diamond,math_500,mmlu_stem \
    --gold_is_latex \
    --verbose
```

## 🔎 Analysis

### 🧵 SAE Textual Activation Visualization

Script: `analysis/basic_loading_and_analysing.py`

Parameters:
- `--model-name`: Base model name or path.
- `--model-class-name`: HF model class name.
- `--torch-dtype`: `float16`, `bfloat16`, etc.
- `--dashboard-tl-model-name`: TransformerLens model name for dashboard rendering.
- `--dataset-path`: Dataset loader type (e.g., `json`).
- `--data-files`: List of dataset files.
- `--streaming`: Enable streaming dataset loading.
- `--no-streaming`: Disable streaming.
- `--token-limit`: Number of tokens to sample.
- `--minibatch-size-features`: Batch size for feature extraction.
- `--minibatch-size-tokens`: Batch size for token processing.
- `--sae-path`: SAE checkpoint path (repeatable).
- `--features`: Feature list per SAE (repeatable, comma-separated or JSON list).
- `--output-file`: Output HTML file per SAE (repeatable).
- `--cuda-visible-devices`: Override CUDA device list.
- `--hf-endpoint`: HF endpoint override.
- `--tokenizers-parallelism`: Tokenizers parallelism setting.

Notes:
- `--features` and `--output-file` should align with `--sae-path` entries.

Script template:
```bash
# or Qwen/Qwen2.5-7B for --dashboard-tl-model-name.
# e.g. 47034,50541 for --features.
# Set small minibatch sizes to avoid OOM.
python basic_loading_and_analysing.py \
    --model-name LRM_PATH \
    --dashboard-tl-model-name meta-llama/Llama-3.1-8B \
    --data-files TOKENIZED_JSONL \
    --sae-path YOUR_SAE \
    --features FEATURES \
    --output-file llama_dr.html \
    --token-limit 10000000 \
    --minibatch-size-features 2 \
    --minibatch-size-tokens 16 \
    --cuda-visible-devices 0
```

### 📉 Logit Change Across Layers Visualization

Script: `analysis/logits_lens_with_features.py`

Parameters:
- `--sae-path`: SAE checkpoint path.
- `--model-name`: Base model name or path (required).
- `--model-class-name`: HF model class name.
- `--torch-dtype`: Torch dtype.
- `--feature-indices`: Feature indices to analyze (comma-separated).
- `--output-dir`: Output directory.
- `--top-k`: Top-k tokens to display.
- `--cosine-sim`: Use cosine similarity instead of logits.
- `--prompts-jsonl`: JSONL file with prompts.
- `--text-field`: Field name for prompt text (default: `text`).
- `--max-prompts`: Max number of prompts to analyze.
- `--max-length`: Max prompt length.
- `--batch-size`: Batch size.
- `--sample-seed`: Random seed for sampling prompts.
- `--steer-coeff`: Steering coefficient.
- `--steer-coeffs`: List of steering coefficients.
- `--layerwise-top-k`: Top-k tokens for layerwise views.
- `--logit-position`: `last` or `mean`.
- `--hidden-state-path`: Hidden-state file path.
- `--hidden-position`: Token position for hidden-state analysis.
- `--hidden-layer-index`: Layer index for hidden-state analysis.
- `--hidden-steer-coeff`: Steering coefficient for hidden-state analysis.
- `--hidden-label`: Label for hidden-state analysis.
- `--hidden-apply-norm`: Normalize hidden-state vectors.
- `--hf-endpoint`: HF endpoint override.
- `--tokenizers-parallelism`: Tokenizers parallelism setting.

Script template:

```bash
python logits_lens_with_features.py \
    --model-name LRM_PATH \
    --prompts-jsonl DATA_JSONL \
    --feature-indices 7761,50541 \
    --steer-coeffs 5.84,10 \
    --output-dir YOUR_OUTPUT_DIR \
    --layerwise-top-k 20 \
    --max-prompts 10000 \
    --max-length 4096 \
    --batch-size 2 \
    --logit-position mean
```

## ❓ FAQs


## 🙏 Acknowledgments

- [SAELens](https://github.com/decoderesearch/SAELens) for SAE training.
- [Math-Verify](https://github.com/huggingface/Math-Verify), and [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) for evaluation support.

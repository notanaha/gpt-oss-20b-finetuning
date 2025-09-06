# gpt-oss-20b Fine-Tuning (Azure ML)

## Overview
This project fine-tunes the `openai/gpt-oss-20b` model for Japanese reasoning.

The workflow includes:
1. Preparing an Azure ML Environment
2. Converting the APTO reasoning dataset into Harmony (messages) format
3. Registering the dataset as an Azure ML data asset
4. Launching supervised fine-tuning via an Azure ML job
5. Merging LoRA weights into the base model
6. Registering and deploying the merged model to a Managed Online Endpoint
7. Sending test inference requests

## Runtime Environments
- Notebook executed on Azure ML Compute Instance with kernel: `azureml_py310_sdkv2`
- You need to install packages in requirements.txt on top of `azureml_py310_sdkv2`
- Training job ran on a single node of: `Standard_NC40ads_H100_v5` (40 vCPU cores, 320 GB RAM, 128 GB disk, H100 GPU)

## Data Preparation
The dataset `APTOinc/japanese-reasoning-dataset-sample` is transformed so each record becomes a `messages` list:
- system: includes `reasoning language: Japanese` and persona text
- user: original question
- assistant: `thinking` (extracted from `<think>` tag) + final answer content

Both Parquet and JSONL outputs are saved:
```
apto_reasoning_harmony.parquet
apto_reasoning_harmony.jsonl
```
They are then registered as an Azure ML data asset (`apto_reasoning_harmony`).

## Training Configuration
- After training, the LoRA adapter is merged (`model.merge_and_unload()`) and saved under `outputs/merged`.
- For Data Parallel, set NUM_NODES > 1. Please note Standard_NC40ads_H100_v5 has only 1 GPU. 
<br>If error "Expected to mark a variable ready only once" occurs under DPP, uncomment SFTConfig params and PYTORCH_CUDA_ALLOC_CONF env variable. When applying LoRA to the MoE-style shared/base_layer while using gradient checkpointing (reentrant), this issue tends to occur more frequently under DDP. However, setting gradient_checkpointing_kwargs={"use_reentrant": False} and ddp_find_unused_parameters=False may cause OOM. In such a case, also need to set packing=False and max_seq_length. When this code was prepared, setting max_seq_length hit trl bug. 



## Model Registration & Deployment
1. Model registered as `gpt-oss-20b-jp-reasoner-01-pre` (type `custom_model`).
2. Scoring script: `src/score.py` builds messages from either a raw prompt or a full chat list and applies the tokenizer chat template.
3. Deployment uses `ManagedOnlineEndpoint` + `ManagedOnlineDeployment` on `Standard_NC40ads_H100_v5`.
4. Traffic routed 100% to deployment `blue`.

## Inference Request Pattern
Example JSON payload:
```json
{
  "messages": [
    {"role": "system", "content": "reasoning language: Japanese"},
    {"role": "user", "content": "オーストラリアの首都はどこですか？"}
  ],
  "max_new_tokens": 500
}
```
Score response returns `{ "output": "..." }`.

## Repository Structure
```
environment/
  Dockerfile          # Custom image build context
  requirements.txt    # Python dependencies
src/
  train.py            # LoRA + SFT training script
  score.py            # Online endpoint scoring script
apto_reasoning_harmony.parquet
apto_reasoning_harmony.jsonl
gpt-oss-finetuning.ipynb
README.md
```

## Repro Steps (High-Level)
1. Set environment variables: `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, `WORKSPACE_NAME`, `HF_TOKEN`.
2. Run the notebook cells in order (environment build, dataset conversion, data registration, training job submission, model registration, deployment).
3. Send a POST request with an auth key to the endpoint scoring URI.


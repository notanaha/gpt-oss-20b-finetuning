# gpt-oss-20b Fine-Tuning (Azure ML)

## Overview
This project fine-tunes the `openai/gpt-oss-20b` model for Japanese reasoning using a Harmony-style chat dataset with explicit <think> ... </think> reasoning traces.

The workflow includes:
1. Preparing a Docker-based Azure ML Environment
2. Converting the APTO reasoning dataset into Harmony (messages) format
3. Registering the dataset as an Azure ML data asset
4. Launching supervised fine-tuning (LoRA + MXFP4 quantization) via an Azure ML job
5. Merging LoRA weights into the base model
6. Registering and deploying the merged model to a Managed Online Endpoint
7. Sending test inference requests

## Runtime Environments
- Notebook executed on Azure ML Compute Instance with kernel: `azureml_py310_sdkv2`
- Training job ran on a single node of: `Standard_NC40ads_H100_v5` (40 vCPU cores, 320 GB RAM, 128 GB disk, H100 GPU)

## Key Technologies
- Azure Machine Learning (v2 SDK)
- Transformers / PEFT / TRL
- MXFP4 quantization for memory efficiency
- LoRA parameter-efficient fine-tuning

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

## Training Configuration (LoRA + SFT)
Important trainer arguments (see `src/train.py`):
- Quantization: `Mxfp4Config(dequantize=False)` (in-quantized forward)
- LoRA target modules: selected expert projection layers
- Batch: per-device 2, gradient accumulation 8
- Scheduler: cosine with min LR rate
- Checkpoint limit: 2
- Reporting: `trackio`

After training, the LoRA adapter is merged (`model.merge_and_unload()`) and saved under `outputs/merged`.

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

## Security & Tokens
- Authentication uses `DefaultAzureCredential` (managed identity or dev identity).
- Hugging Face token is passed via environment (`HF_TOKEN`).
- No plaintext secrets committed.

## Notes / Improvements
- Add evaluation metrics (accuracy, reasoning quality benchmarks).
- Automate pipeline with Azure ML jobs + CLI / GitHub Actions.
- Consider multi-node scaling if larger batch sizes are needed.
- Add prompt guardrails and output filtering for production.

## License
(Insert license information here if applicable.)

# gpt-oss-20b Fine-Tuning (Azure ML)

## Overview
This project fine-tunes the `openai/gpt-oss-20b` model for Japanese reasoning.

With the original model, the thought process is in English. 
<br>For instance, if the user asks: `オーストラリアの首都はどこですか？(What is the capital of Australia?)`
```
The user asks in Japanese: "オーストラリアの首都はどこですか？" (What is the capital of Australia?). The answer: Canberra. Should provide in Japanese. So respond: "オーストラリアの首都はキャンベラです." Possibly add more context. That's straightforward.
オーストラリアの首都はキャンベラです。
```
After the fine-tuning process with a Japanese reasoning dataset, the thought process will change.
```
まず、問題文を確認します。「オーストラリアの首都はどこですか？」という質問です。これは地理に関する基本的な知識を問うものです。
次に、オーストラリアの首都について考えます。一般的に知られていることとして、シドニーはオーストラリア最大の都市であり、経済的にも重要な都市ですが、首都ではありません。実際の首都はキャンベラです。キャンベラはオーストラリアの中部に位置し、政府機関が集中しています。
さらに、オーストラリアの首都がキャンベラであることを確認するために、他の情報源を参照することもできます。例えば、オーストラリア政府の公式ウェブサイトや、国際的な地理情報サイトなどで確認することができます。
最後に、正しい答えをまとめます。オーストラリアの首都はキャンベラです。オーストラリアの首都はキャンベラです。
```

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
- You need to install packages in **requirements.txt** on top of `azureml_py310_sdkv2`
- Training job ran on a single node of: `Standard_NC40ads_H100_v5` (40 vCPU cores, 320 GB RAM, 128 GB disk, H100 GPU)
- If image build failed, try removing the following statements.

```
#RUN apt-get update
#RUN apt-get install -y openssh-server openssh-client
```

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

* **Merging LoRA Adapters**
  After training finishes, the LoRA adapter weights are merged back into the base model using:

  ```python
  model.merge_and_unload()
  ```

  The merged model is then saved under `outputs/merged`.

* **Data Parallel Training**
  To use multi-node data parallelism, set `NUM_NODES > 1`.
  <br>Note: `Standard_NC40ads_H100_v5` VMs only provide **1 GPU per node**.

* **Common DDP Error**
  If you see the error:

  ```
  Expected to mark a variable ready only once
  ```

  under Distributed Data Parallel (DDP):

  1. Uncomment the related `SFTConfig` parameters.
  2. This error often occurs when LoRA is applied to **MoE-style `shared/base_layer` modules** while using gradient checkpointing with the default *reentrant* mode.
  3. A workaround is to use:

     ```python
     gradient_checkpointing_kwargs = {"use_reentrant": False}
     ddp_find_unused_parameters = False
     ```

     However, this setting can increase the risk of out-of-memory (OOM) errors.

* **OOM Mitigation Tips**
  If you encounter OOM after applying the above fix:

  * Set `packing = False`
  * Explicitly define `max_seq_length` in `SFTConfig`
  * Set the `PYTORCH_CUDA_ALLOC_CONF` environment variable

  At the time this code was prepared, there was a known bug in TRL where `max_seq_length` caused issues.



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

## Acknowledgements / References
- This workflow and several implementation details were informed by the article:
  *Fine-tuning with gpt-oss and Hugging Face Transformers* — https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
 - Reasoning-focused dataset structuring and `<think>` extraction also drawn from the dataset:
   *APTOinc/japanese-reasoning-dataset-sample* — https://huggingface.co/datasets/APTOinc/japanese-reasoning-dataset-sample



# Llama Stack LLM Explanation API

This repository hosts the tutorial for `explanation` API to generate explanations of LLM (Large Language Model) outputs using [Captum](https://github.com/pytorch/captum) as a remote provider. The API is designed to work with LLMs served via [vLLM](https://github.com/vllm-project/vllm) and exposes endpoints for both online and batch explanation workflows. The code for the Llama Stack explanation API can be found [here](https://github.com/saichandrapandraju/llama-stack/tree/captum-explain) and is under development.


## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [API Endpoints](#api-endpoints)
  - [Model Metadata](#1-get-model-metadata)
  - [Online Explanation](#2-online-explanation)
  - [Batch Explanation](#3-batch-explanation)
  - [Job Management](#4-job-management)
- [How It Works](#how-it-works)
- [References](#references)


## Overview

Llama Stack's Explanation API enables users to generate token-level and sequence-level attributions for LLM outputs using Captum's black-box (perturbation-based) methods. This is particularly useful for understanding model behavior, debugging, and building explainable AI applications.

- **Provider:** Captum (remote, via vLLM logprobs)
- **Supported Methods:** FeatureAblation, ShapleyValues, ShapleyValueSampling, LIME, KernelShap
- **Modes:** Online (single request), Batch (asynchronous jobs)

## Prerequisites

- A running LLM endpoint served with [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/v0.6.4/serving/openai_compatible_server.html)
- Docker or Podman for running the Llama Stack server
- [HuggingFace tokenizer name](https://huggingface.co/models) for your LLM

## Quickstart

1. **Clone the repository:**
    ```bash
    git clone git@github.com:saichandrapandraju/lls-explanation-reference.git
    cd lls-explanation-reference
    ```

2. **Set the server port:**
    ```bash
    export LLAMA_STACK_PORT=8321
    ```

3. **Start the Llama Stack server:**
    ```bash
    podman run --pull always -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
      -v ./templates/remote-captum/run.yaml:/root/my-run.yaml \
      quay.io/spandraj/distribution-remote-captum \
      --yaml-config /root/my-run.yaml --port $LLAMA_STACK_PORT \
      --env VLLM_URL=<vLLM-endpoint>/v1 \
      --env TOKENIZER=<HF-tokenizer-name-of-vLLM-served-model>
    ```

    - For multiple models:
      ```bash
      --env VLLM_URL=<vLLM-endpoint-1/v1>,<vLLM-endpoint-2/v1>
      --env TOKENIZER=<tokenizer-1>,<tokenizer-2>
      ```
You can now access the Llama Stack server at http://localhost:8321/v1/

## API Endpoints

### 1. Get Model Metadata

- **GET** `/explanation/models`
- Returns available model IDs, endpoints, and tokenizer names.
#### Sample Response
  ```json
  [
    {
        "model_id": "opt",
        "url": "http://host.docker.internal:8080/v1",
        "tokenizer": "facebook/opt-125m"
    },
    {
        "model_id": "qwen2",
        "url": "http://host.docker.internal:8081/v1",
        "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct"
    }
  ]
  ```

### 2. Online Explanation

- **POST** `/explanation/online`
- Synchronously generates an explanation for a single input.

#### Sample Request Body

```json
{
  "model_id": "opt",
  "content": "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include",
  "algorithm": "fa",  // One of: fa, shap, shap_sampling, lime, kernel_shap
  "target": "playing golf, hiking, and cooking.", // Optional
  "skip_tokens": ["</s>"], // Optional
  "num_trials": 1, // Optional
  "gen_args": null // Optional. Must be null if "target" is provided
}
```

#### Sample Response

```json
{
  "input_features": ["Dave", "lives", "in", "Palm", "Coast", ",", ...],
  "output_features": ["playing", "golf", ",", "hiking", ",", ...],
  "token_attribution": [[0.1, 0.2, ...], ...], // size - [len(output_features), len(input_features)]
  "sequence_attributions": {"Dave": 0.3, "lives": 0.5, ...}, // mapping from input_feature -> attribution score
  "metadata": {"model_id": "opt", "tokenizer": "facebook/opt-125m"}
}
```
#### Templates to capture words/phrases
Captum supports defining certain words or phrases that we're interested in rather than token-level explanations. You can find more about them [here](https://captum.ai/tutorials/Llama2_LLM_Attribution). The current LLS implementation also supports these templates. Below are couple of examples - 
```json
{
  "model_id": "opt",
  "content": {
           "template": "{} lives in {}, {} and is a {}. {} personal interests include", 
           "values": ["Dave", "Palm Coast", "FL", "lawyer", "His"],
           "baselines": ["Sarah", "Seattle", "WA", "doctor", "Her"]
       },
  "algorithm": "fa",  // One of: fa, shap, shap_sampling, lime, kernel_shap
  "target": "playing golf, hiking, and cooking.", // Optional
}
```
Here's a more interesting one - 
```json
{
  "model_id": "opt",
  "content": {
          "template": "{name} lives in {city}, {state} and is a {occupation}. {pronoun} personal interests include",
          "values": {"name": "Dave", "city": "Palm Coast", "state": "FL", "occupation": "lawyer", "pronoun": "His"}, 
          "baselines": {
            "k1":[["name", "pronoun"], [["Sarah", "her"], ["John", "His"], ["Martin", "His"], ["Rachel", "Her"]]],
            "k2": [["city", "state"], [["Seattle", "WA"], ["Boston", "MA"]]],
            "k3": ["occupation", ["doctor", "engineer", "teacher", "technician", "plumber"]]
        },
          "mask": {"name": 0, "city": 1, "state": 1, "occupation": 2, "pronoun": 0}
}, 
  "algorithm": "fa",  // One of: fa, shap, shap_sampling, lime, kernel_shap
  "target": "playing golf, hiking, and cooking.", // Optional
}
```



### 3. Batch Explanation

- **POST** `/explanation/batch`
- Asynchronously generates explanations for a list of inputs.

#### Sample Request Body

```json
{
  "model_id": "opt",
  "content": ["Why is the sky blue?", "What is AI?"],
  "algorithm": "fa"
}
```

#### Sample Response

```json
{
  "job_uuid": "123e4567-e89b-12d3-a456-426614174000"
}
```

### 4. Job Management

- **GET** `/explanation/jobs`  
  List all batch job IDs.
  #### Sample Response
  ```json
  {
    "data": [
        {
            "job_uuid": "123e4567-e89b-12d3-a456-426614174000"
        }
    ]
  }
  ```

- **GET** `/explanation/job/status?job_uuid=<job-id>`  
  Get status of a batch job.
  #### Sample Response
  ```json
  {
    "job_uuid": "123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "scheduled_at": "2025-04-28T17:32:23.483557Z",
    "started_at": "2025-04-28T17:32:23.484967Z",
    "completed_at": "2025-04-28T17:32:24.890857Z"
  }
  ```


- **GET** `/explanation/job/artifacts?job_uuid=<job-id>`  
  Get results/artifacts of a completed job.
  #### Sample Response
  ```json
  {
    "job_uuid": "123e4567-e89b-12d3-a456-426614174000",
    "results": [ // size - len(content)
        "<online-response-instance>",
        ...
    ]
  }
  ```

## How It Works

Currently, [Captum](https://github.com/pytorch/captum) is the only explanation provider that is supported. Captum supports both perturbation-based (black-box) and gradient-based (white-box) methods to explain generative language models. You can find more about Captum's LLM-specific explanation methods in their paper [here](https://arxiv.org/abs/2312.05491) and tutorial [here](https://captum.ai/tutorials/Llama2_LLM_Attribution). The current Llama Stack explanation API is implemented for Captum's perturbation-based LLM attribution methods.

One issue with Captum is that it requires the LLM to be loaded locally and needs local access. This is true even for the black-box explanations that work with token logprobs instead of the model's internal information like hidden states or gradients. In most production settings, LLMs are served with efficient serving engines like [vLLM](https://github.com/vllm-project/vllm), and vLLM supports logprobs of generated tokens and prompt tokens via an OpenAI-compatible API. This insight can be used to make Captum black-box attribution methods work with inference engines like vLLM, as implemented in the [PR here](https://github.com/pytorch/captum/pull/1544). The current implementation uses the corresponding [fork](https://github.com/saichandrapandraju/captum/tree/remote-logprobs) to create explanations with a remotely vLLM-served model.

## References

- [Captum LLM Attribution Paper](https://arxiv.org/abs/2312.05491)
- [Captum LLM Attribution Tutorial](https://captum.ai/tutorials/Llama2_LLM_Attribution)
- [Captum PR: Remote logprobs](https://github.com/pytorch/captum/pull/1544)
- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Tokenizers](https://huggingface.co/models)

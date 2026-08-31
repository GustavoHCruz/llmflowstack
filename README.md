# LLMFlowStack

**LLMFlowStack** is a lightweight framework designed to simplify the training, inference, evaluation, and retrieval-augmented use of large language models such as LLaMA, GPT-OSS, Gemma, MedGemma, and Qwen.

> **Note:** LLMFlowStack is primarily intended for high-performance machines with **one or more NVIDIA GPUs**, especially for training and large-model inference. Smaller models and the RAG components may also run on more modest hardware.

It provides:

- **Training pipelines** with fine-tuning or DAPT setups;
- **Inference** with optional model quantization;
- **Streaming text generation**.

The goal is to make experimentation with LLMs more accessible without requiring users to build complex infrastructure from scratch.

## Supported Models

This framework is designed to provide flexibility when working with different open-source and commercial LLMs. Currently, the following models are supported:

- **GPT-OSS**

  - [`GPT-OSS 20B`](https://huggingface.co/openai/gpt-oss-20b)
  - [`GPT-OSS 120B`](https://huggingface.co/openai/gpt-oss-120b)

- **LLaMA 3**

  - [`LLaMA 3.1 8B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - [`LLaMA 3.1 70B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - [`LLaMA 3.3 70B - Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
  - [`LLaMA 3.1 405B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)

- **LLaMA 4**

  - [`LLaMA 4 Scout - Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

- **Gemma 3**

  - [`Gemma 3 27B - It`](https://huggingface.co/google/gemma-3-27b-it)

- **MedGemma**

  - [`MedGemma 4B - It`](https://huggingface.co/google/medgemma-4b-it)
  - [`MedGemma 27B - It`](https://huggingface.co/google/medgemma-27b-it)

- **Gemma 4**

  - [`Gemma 4 12B It`](https://huggingface.co/google/gemma-4-12B-it)
  - [`Gemma 4 32B It`](https://huggingface.co/google/gemma-4-31B-it)

- **Qwen 3.5**

  - [`Qwen 3.5 0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
  - [`Qwen 3.5 2B`](https://huggingface.co/Qwen/Qwen3.5-2B)
  - [`Qwen 3.5 4B`](https://huggingface.co/Qwen/Qwen3.5-4B)
  - [`Qwen 3.5 9B`](https://huggingface.co/Qwen/Qwen3.5-9B)
  - [`Qwen 3.5 27B`](https://huggingface.co/Qwen/Qwen3.5-27B)
  - [`Qwen 3.5 35B - A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
  - [`Qwen 3.5 122B - A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
  - [`Qwen 3.5 397B - A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

- **Qwen 3.6**

  - [`Qwen 3.6 27B`](https://huggingface.co/Qwen/Qwen3.6-27B)
  - [`Qwen 3.6 35B - A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

- **Qwen 3.8**

  - [`Qwen 3.8 27B`](https://huggingface.co/Qwen/Qwen3.8-27B)

- **Muse Glimmer**
  - [`Muse Glimmer 30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)

> Other architectures based on these model families may also function correctly.

---

## Installation

LLMFlowStack requires Python 3.13 or newer.

First, install PyTorch with CUDA support:

```bash
pip install torch==2.13 torchvision==0.28 --index-url https://download.pytorch.org/whl/cu126
```

Verify that CUDA is correctly available:

```python
import torch

assert torch.cuda.is_available()
print(torch.version.cuda)
```

### Base installation

Install LLMFlowStack from source:

```bash
cd llmflowstack
pip install .
```

Or install it directly from GitHub:

```bash
pip install git+https://github.com/GustavoHCruz/llmflowstack.git
```

---

## Usage

This section presents some of the main operations supported by the framework.

### Loading models

You can load as many models as your hardware allows.

```python
from llmflowstack import GptOss, Llama3

# Loading a LLaMA model
first_model = Llama3()
first_model.load_checkpoint(
    checkpoint="/llama-3.1-8b-Instruct",
)

# Loading a quantized LLaMA model
second_model = Llama3(
    checkpoint="/llama-3.3-70b-Instruct",
    quantization=True,
)

# Loading GPT-OSS with quantization and a fixed seed
third_model = GptOss(
    checkpoint="/gpt-oss-120b",
    quantization=True,
    seed=1234,
)
```

### Inference examples

```python
from llmflowstack import GenerationParams, GptOss

gpt_oss_model = GptOss(
    checkpoint="/gpt-oss-120b",
)

answer = gpt_oss_model.generate(
    "Tell me a joke!",
)

print(answer)
```

GPT-OSS supports configurable reasoning levels:

```python
gpt_oss_model.set_reasoning_level("High")
```

Available levels are:

- `"Low"`
- `"Medium"`
- `"High"`
- `"Off"`

You can also construct a custom model input:

```python
custom_input = gpt_oss_model.build_input(
    input_text="Tell me another joke!",
    developer_message=(
        "You are a clown. After every joke, say 'HONK HONK'."
    ),
)

answer = gpt_oss_model.generate(
    data=custom_input,
    params=GenerationParams(
        mode="sample",
        max_new_tokens=1024,
        temperature=0.3,
    ),
)

print(answer)
```

The supported generation modes are:

- `"greedy"`
- `"sample"`
- `"beam"`

Another model family can be used through the same general interface:

```python
from llmflowstack import Llama3

llama_model = Llama3(
    checkpoint="/llama-3.3-70B-Instruct",
    quantization=True,
)

answer = llama_model.generate(
    "Why is the sky blue?",
)

print(answer)
```

All supported architectures also accept OpenAI-style messages during inference:

```python
messages = [
    {
        "role": "system",
        "content": "Answer briefly.",
    },
    {
        "role": "user",
        "content": "Why is the sky blue?",
    },
]

answer = llama_model.generate(
    messages=messages,
)

print(answer)
```

The conversation history can be passed directly as the first argument:

```python
answer = llama_model.generate(
    [
        {
            "role": "user",
            "content": "My name is Ana.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, Ana.",
        },
        {
            "role": "user",
            "content": "What is my name?",
        },
    ]
)

print(answer)
```

The `messages` input is inference-only and is supported by both `generate` and `generate_stream`.

Each message must contain:

- a textual `content`;
- one of the following roles:
  - `system`;
  - `developer`;
  - `user`;
  - `assistant`.

The checkpoint's native chat template is used when available.

The `developer` role is preserved by GPT-OSS and treated as a system instruction by the other architectures.

GPT-OSS reasoning can also be disabled:

```python
gpt_oss_model.set_reasoning_level("Off")
```

> Disabling reasoning is intended for inference-only usage. It may not behave correctly after the model has been trained or fine-tuned.

### Streaming generation

You can receive generated text incrementally with `generate_stream`:

```python
from llmflowstack import Llama4

llama_4 = Llama4(
    checkpoint="llama-4-scout-17b-16e-instruct",
)

stream = llama_4.generate_stream(
    "Who was Alan Turing?",
)

for text in stream:
    print(
        text,
        end="",
        sep="",
    )
```

The iterator yields text fragments until the model reaches an end-of-generation token or iteration is interrupted.

### Training examples: DAPT and fine-tuning

```python
from llmflowstack import Llama3
from llmflowstack.schemas import TrainParams

model = Llama3(
    checkpoint="llama-3.1-8b-Instruct",
)

dataset = [
    model.build_input(
        input_text="Chico is a cat. Which color is he?",
        output_text="Black!",
    ),
    model.build_input(
        input_text="Fred is a dog. Which color is he?",
        output_text="White!",
    ),
]
```

Run Domain-Adaptive Pretraining over the full model:

```python
model.train(
    train_data=dataset,
    params=TrainParams(
        batch_size=1,
        epochs=3,
        gradient_accumulation=1,
        lr=2e-5,
    ),
    mode="DAPT",
)
```

Run supervised fine-tuning:

```python
model.train(
    train_data=dataset,
    params=TrainParams(
        batch_size=1,
        gradient_accumulation=1,
        lr=2e-5,
        epochs=50,
    ),
    save_at_end=True,
    save_path="./output",
    mode="FT",
)
```

Save the final checkpoint:

```python
model.save_checkpoint(
    path="./model-output",
)
```

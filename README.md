# LLMFlowStack

**LLMFlowStack** is a lightweight framework designed to simplify the use of LLMs (LLaMA, GPT-OSS, and Gemma) for NLP tasks.

> **Note:** LLMFlowStack is intended for high-performance machines with **one or more NVIDIA H100 GPUs**.

It provides:

- **Training pipelines** with **fine-tuning** or **DAPT** in distributed setups — just provide the data and the process runs automatically;
- **Distributed inference** made simple;
- **Evaluation** with standard metrics (BERTScore, ROUGE, Cosine Similarity).

The goal is to make experimentation with LLMs more accessible, without the need to build complex infrastructure from scratch.

## Supported Models

This framework is designed to provide flexibility when working with different open-source and commercial LLMs. Currently, the following models are supported:

- **GPT-OSS**

  - [`GPT-OSS 20B`](https://huggingface.co/openai/gpt-oss-20b)
  - [`GPT-OSS 120B`](https://huggingface.co/openai/gpt-oss-120b)
    > Fine-Tuning, DAPT and Inference Available

- **LLaMA 3**

  - [`LLaMA 3.1 8B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - [`LLaMA 3.1 70B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - [`LLaMA 3.3 70B - Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
  - [`LLaMA 3.3 405B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
    > Fine-Tuning, DAPT and Inference Available

- **LLaMA 4**

  - [`LLaMA 4 Scout - Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
    > DAPT and Inference Available

- **Gemma**

  - [`Gemma 3 27B - Instruct`](https://huggingface.co/google/gemma-3-27b-it)
    > DAPT and Inference Available

- **MedGemma**
  - [`MedGemma 27B Text - Instruct`](https://huggingface.co/google/medgemma-27b-text-it)
    > Fine-Tuning, DAPT and Inference Available

> Other architectures based on those **may** function correctly.

---

## Installation

You can install the package directly from [PyPI](https://pypi.org/project/llmflowstack/):

```bash
pip install llmflowstack
```

## Usage

This section presents a bit of what you can do with the framework.

### Loading models

You can load as many models as your hardware allows (H100 GPU recommended)...

```python
from llmflowstack import GPT_OSS, LLaMA3

# Loading a LLaMA model
first_model = LLaMA3()
first_model.load_checkpoint(
  checkpoint="/llama-3.1-8b-Instruct",
)

# Loading a quantized LLaMA model
second_model = LLaMA3(
  checkpoint="/llama-3.3-70b-Instruct",
  quantization="4bit"
)

# Loading a GPT-OSS, quantized and with seed
thrid_model = GPT_OSS(
  checkpoint="/gpt-oss-20b",
  quantization=True,
  seed=1234
)
```

### Inference Examples

```python
> gpt_oss_model = GPT_OSS(checkpoint="/gpt-oss-120b")

> gpt_oss_model.generate("Tell me a joke!")
'Why did the scarecrow become a successful motivational speaker? Because he was outstanding **in** his field! 🌾😄'

# Exclusive for GPT-OSS
> gpt_oss_model.set_reasoning_level("High")

> custom_input = gpt_oss_model.build_input(
    input_text="Tell me another joke!",
    developer_message="You are a clown and after every joke, you should say 'HONK HONK'"
  )
> gpt_oss_model.generate(
    input=custom_input,
    params=GenerationParams(
      max_new_tokens=1024,
      sample=GenerationSampleParams(
        temperature=0.3
      )
    )
  )
'Why did the scarecrow win an award? Because he was outstanding in his field!  \n\nHONK HONK'

> llama_model = LLaMA3(checkpoint="/llama-3.3-70B-Instruct", quantization="4bit")
> llama_model.generate("Why is the sky blue?")
'The sky appears blue because of a phenomenon called Rayleigh scattering, which is the scattering of light'
```

### Training Examples (DAPT & Fine-tune)

```python
from llmflowstack import LLaMA3
from llmflowstack.schemas import TrainParams

model = LLaMA3(
  checkpoint="llama-3.1-8b-Instruct"
)

# Creating the dataset
dataset = []
dataset.append(model.build_input(
  input_text="Chico is a cat, which color he is?",
  expected_answer="Black!"
))

dataset.append(model.build_input(
  input_text="Fred is a dog, which color he is?",
  expected_answer="White!"
))

# Does the DAPT in the full model
model.dapt(
  train_dataset=dataset,
  params=TrainParams(
    batch_size=1,
    epochs=3,
    gradient_accumulation=1,
    lr=2e-5
  )
)

# Does the fine-tune this time
model.fine_tune(
  train_dataset=dataset,
  params=TrainParams(
    batch_size=1,
    gradient_accumulation=1,
    lr=2e-5,
    epochs=50
  ),
  save_at_end=True,
  # It will save the model
  save_path="./output"
)

# Saving the final result
model.save_checkpoint(
  path="./model-output"
)
```

### NLP Evaluation

```python
> from llmflowstack import text_evaluation
> from llmflowstack.utils import (bert_score_evaluation, cosine_similarity_evaluation, rouge_evaluation)

# Predictions from some model
> predictions = ["Chico is a dog, and he is orange!", "Fred is a cat, and he is white!"]
# References text (ground truth)
> references = ["Chico is a cat, and he is black!", "Fred is a dog, and he is white!"]

# BERTScore Evaluation
> bert_score_evaluation(predictions, references)
{'bertscore_precision': 0.9772549867630005, 'bertscore_recall': 0.9772549867630005, 'bertscore_f1': 0.9772549867630005}

# Cosine Similarity Evaluation
> cosine_similarity_evaluation(predictions, references)
{'cosine_similarity': 0.7443363666534424}

# RougeScore Evaluation
> rouge_evaluation(predictions, references)
{'rouge1': 0.8125, 'rouge2': 0.6428571428571428, 'rougeL': 0.8125}

# All-in-one function
> text_evaluation(predictions, references, bert=True, cosine=True, rouge=True)
{'rouge1': 0.8125, 'rouge2': 0.6428571428571428, 'rougeL': 0.8125, 'bertscore_precision': 0.9772549867630005, 'bertscore_recall': 0.9772549867630005, 'bertscore_f1': 0.9772549867630005, 'cosine_similarity': 0.7443363666534424}
```

---

> **Disclaimer**  
> This is a public fork of a framework originally developed in a research setting.  
> Institution-specific components have been removed for confidentiality reasons.

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

- **LLaMA 3**

  - [`LLaMA 3.1 8B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - [`LLaMA 3.1 70B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - [`LLaMA 3.3 70B - Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
  - [`LLaMA 3.3 405B - Instruct`](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)

- **LLaMA 4**

  - [`LLaMA 4 Scout - Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

- **Gemma**
  - [`Gemma 3 27B - Instruct`](https://huggingface.co/google/gemma-3-27b-it)

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
from llmflowstack import GptOss, Llama3

# Loading a Llama model
first_model = Llama3()
first_model.load_checkpoint(
  checkpoint="/llama-3.1-8b-Instruct",
)

# Loading a quantized Llama model
second_model = Llama3(
  checkpoint="/llama-3.3-70b-Instruct",
  quantization="4bit"
)

# Loading a GPT-OSS, quantized and with seed
thrid_model = GptOss(
  checkpoint="/gpt-oss-20b",
  quantization=True,
  seed=1234
)
```

### Inference Examples

```python
> from llmflowstack import GptOss, GenerationParams

> gpt_oss_model = GptOss(checkpoint="/gpt-oss-120b")

> gpt_oss_model.generate("Tell me a joke!")
'Why did the scarecrow become a successful motivational speaker? Because he was outstanding **in** his field! 🌾😄'

# Exclusive for GPT-OSS
> gpt_oss_model.set_reasoning_level("High") # Low, Medium, High, Off

> custom_input = gpt_oss_model.build_input(
    input_text="Tell me another joke!",
    developer_message="You are a clown and after every joke, you should say 'HONK HONK'"
  )
> gpt_oss_model.generate(
    data=custom_input,
    params=GenerationParams(
      mode="sample", # greedy, sample or beam
      max_new_tokens=1024,
      temperature=0.3
    )
  )
'Why did the scarecrow win an award? Because he was outstanding in his field!  \n\nHONK HONK'

> llama_model = Llama3(checkpoint="/llama-3.3-70B-Instruct", quantization="4bit")
> llama_model.generate("Why is the sky blue?")
'The sky appears blue because of a phenomenon called Rayleigh scattering, which is the scattering of light'

# You can also disable GPT-OSS reasoning, but this works only when the model is being used strictly for inference. If the model has been trained or fine-tuned beforehand, this option will not behave correctly.
> gpt_oss_model.set_reasoning_level("Off") # (inference-only)
```

You can also generate tokens using a streamer, that is, receiving one token at a time by using the iterator version of the generate function:

```python
llama_4 = Llama4(
  checkpoint="llama-4-scout-17b-16e-instruct"
)

it = llama_4.generate_stream("Who was Alan Turing?")

for text in it:
  print(text, end="", sep="")   # The model will keep yielding tokens until it reaches an end-of-generation token (or until you stop iterating)
```

### Training Examples (DAPT & Fine-tune)

```python
from llmflowstack import Llama3
from llmflowstack.schemas import TrainParams

model = Llama3(
  checkpoint="llama-3.1-8b-Instruct"
)

# Creating the dataset
dataset = []
dataset.append(model.build_input(
  input_text="Chico is a cat, which color he is?",
  output_text="Black!"
))

dataset.append(model.build_input(
  input_text="Fred is a dog, which color he is?",
  output_text="White!"
))

# Does the DAPT in the full model
model.train(
  train_data=dataset,
  params=TrainParams(
    batch_size=1,
    epochs=3,
    gradient_accumulation=1,
    lr=2e-5
  ),
  mode="DAPT"
)

# Does the fine-tune this time
model.train(
  train_data=dataset,
  params=TrainParams(
    batch_size=1,
    gradient_accumulation=1,
    lr=2e-5,
    epochs=50
  ),
  save_at_end=True,
  # It will save the model
  save_path="./output",
  mode="FT"
)

# Saving the final result
model.save_checkpoint(
  path="./model-output"
)
```

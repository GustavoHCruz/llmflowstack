# LLMFlowStack

**LLMFlowStack** is a lightweight framework designed to simplify the training, inference, evaluation, and retrieval-augmented use of large language models such as LLaMA, GPT-OSS, Gemma, MedGemma, and Qwen.

> **Note:** LLMFlowStack is primarily intended for high-performance machines with **one or more NVIDIA GPUs**, especially for training and large-model inference. Smaller models and the RAG components may also run on more modest hardware.

It provides:

- **Training pipelines** with fine-tuning or DAPT setups;
- **Inference** with optional model quantization;
- **Streaming text generation**;
- **Retrieval-Augmented Generation utilities**, including:
  - document splitting;
  - sentence-transformer embeddings;
  - persistent Chroma vector stores;
  - document indexing and retrieval;
  - metadata filtering;
  - context formatting for LLM prompts.

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

### Installation with RAG support

The Retrieval-Augmented Generation components are distributed as an optional dependency group.

From a local clone:

```bash
cd llmflowstack
pip install ".[rag]"
```

Directly from GitHub:

```bash
pip install "llmflowstack[rag] @ git+https://github.com/GustavoHCruz/llmflowstack.git"
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

---

## Retrieval-Augmented Generation

The `llmflowstack.rag` module provides utilities for creating local vector databases, indexing documents, retrieving relevant information, and formatting the retrieved context for an LLM.

The main components are available directly from `llmflowstack.rag`:

```python
from llmflowstack.rag import (
    ContextFormatter,
    DocumentSplitter,
    Retriever,
    SentenceTransformerEmbedding,
    VectorStore,
)
```

### Creating a vector database

First, load an embedding model and create a vector store:

```python
from llmflowstack.rag import (
    DocumentSplitter,
    Retriever,
    SentenceTransformerEmbedding,
    VectorStore,
)

embedding_model = SentenceTransformerEmbedding(
    checkpoint="sentence-transformers/all-MiniLM-L12-v2",
    normalize_embeddings=True,
)

vector_store = VectorStore(
    collection_name="my_documents",
    embedding_model=embedding_model,
    persist_directory="./vector_database",
)

splitter = DocumentSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

retriever = Retriever(
    vector_store=vector_store,
    splitter=splitter,
)
```

The `persist_directory` argument defines where the Chroma vector database will be stored.

### Adding documents

A document can be created, split, embedded, and indexed with `insert_document`:

```python
retriever.insert_document(
    information=(
        "Retrieval-Augmented Generation combines document retrieval "
        "with language-model generation."
    ),
    source_id="rag-introduction",
    metadata={
        "category": "machine-learning",
        "language": "en",
    },
)
```

When no `source_id` is provided, one is generated automatically.

The document is split according to the configured `DocumentSplitter`. Each generated chunk receives metadata such as:

- `source_id`;
- `chunk_id`;
- `chunk_index`.

To store the entire document as a single chunk:

```python
retriever.insert_document(
    information="This document will not be split.",
    source_id="single-chunk-document",
    can_split=False,
)
```

Several LangChain documents can also be indexed together:

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="The first document.",
        metadata={"category": "example"},
    ),
    Document(
        page_content="The second document.",
        metadata={"category": "example"},
    ),
]

retriever.index_documents(
    documents,
    source_ids=[
        "document-001",
        "document-002",
    ],
)
```

### Retrieving documents

Use `retrieve` to search for the most relevant chunks:

```python
documents = retriever.retrieve(
    query="What is Retrieval-Augmented Generation?",
    k=4,
)

for document in documents:
    print(document.page_content)
    print(document.metadata)
```

Metadata filters can be applied during retrieval:

```python
documents = retriever.retrieve(
    query="What is Retrieval-Augmented Generation?",
    k=4,
    filter={
        "category": "machine-learning",
    },
)
```

To retrieve the documents together with their vector-store scores:

```python
results = retriever.retrieve_with_score(
    query="What is Retrieval-Augmented Generation?",
    k=4,
)

for document, score in results:
    print(score)
    print(document.page_content)
```

### Formatting the retrieved context

`ContextFormatter` converts the retrieved documents into a single string that can be passed to an LLM:

```python
from llmflowstack.rag import ContextFormatter

formatter = ContextFormatter(
    include_source_id=True,
    include_chunk_index=True,
)

context = formatter.format_documents(
    documents,
)

print(context)
```

Scores can also be included:

```python
context = formatter.format_scored_documents(
    results,
    include_score=True,
)
```

A custom document separator can be configured:

```python
formatter = ContextFormatter(
    document_separator="\n\n---\n\n",
)
```

You can also provide a custom formatting function:

```python
from langchain_core.documents import Document


def format_document(
    document: Document,
    index: int,
) -> str:
    source_id = document.metadata.get("source_id", "unknown")

    return (
        f"### Reference {index + 1}\n"
        f"Source: {source_id}\n\n"
        f"{document.page_content}"
    )


context = formatter.format_documents(
    documents,
    formatter=format_document,
)
```

### Reopening an existing vector database

To reuse a previously created database, instantiate the embedding model and vector store again with the same collection name and persistence directory:

```python
embedding_model = SentenceTransformerEmbedding(
    checkpoint="sentence-transformers/all-MiniLM-L12-v2",
    normalize_embeddings=True,
)

vector_store = VectorStore(
    collection_name="my_documents",
    embedding_model=embedding_model,
    persist_directory="./vector_database",
)

retriever = Retriever(
    vector_store=vector_store,
)

documents = retriever.retrieve(
    query="What was indexed previously?",
    k=4,
)
```

Use the same embedding checkpoint and configuration that were used when the database was created.

### Updating and deleting documents

Replace all chunks associated with a source:

```python
retriever.update_document(
    source_id="rag-introduction",
    new_information=(
        "Retrieval-Augmented Generation retrieves external information "
        "before generating an answer."
    ),
    metadata={
        "category": "machine-learning",
        "language": "en",
    },
)
```

Delete a document and all its chunks:

```python
deleted_chunks = retriever.delete_document(
    source_id="rag-introduction",
)

print(deleted_chunks)
```

Delete documents using metadata:

```python
retriever.delete_where(
    where={
        "category": "temporary",
    },
)
```

### Using it as a LangChain retriever

The vector store can also be exposed as a standard LangChain retriever:

```python
langchain_retriever = retriever.as_langchain_retriever(
    k=4,
    search_type="similarity",
)

documents = langchain_retriever.invoke(
    "What is Retrieval-Augmented Generation?",
)
```

Filters can also be configured:

```python
langchain_retriever = retriever.as_langchain_retriever(
    k=4,
    search_type="similarity",
    filter={
        "category": "machine-learning",
    },
)
```

### Releasing the embedding model

When the embedding model is no longer needed, it can be unloaded explicitly:

```python
embedding_model.close()
```

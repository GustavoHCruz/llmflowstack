import textwrap

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EncoderWrapper(Embeddings):
  def __init__(
    self,
    model: SentenceTransformer
  ) -> None:
    self.model = model

  def embed_documents(
    self,
    texts: list[str]
  ) -> list[list[float]]:
    return self.model.encode(texts, task="retrieval", show_progress_bar=True).tolist()
  
  def embed_query(
    self,
    text: str
  ) -> list[float]:
    return self.model.encode(text, task="retrieval", show_progress_bar=True).tolist()

class RAGPipeline:
  def __init__(
    self,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
  ) -> None:
    
    self.encoder = SentenceTransformer("jinaai/jina-embeddings-v4", trust_remote_code=True)

    self.vector_store = InMemoryVectorStore(EncoderWrapper(self.encoder))

    self.splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      add_start_index=True,
    )
  
  def from_dicts(
    self,
    rows: list[dict[str, str]],
    text_key: str = "Description",
    metadata_keys: list[str] | None = None,
    should_index: bool = True
  ) -> list[Document]:
    docs = []
    for row in rows:
      page_content = row[text_key]
      metadata = {k: row[k] for k in metadata_keys} if metadata_keys else {}
      docs.append(Document(page_content=page_content, metadata=metadata))

    if should_index:
      self.index_documents(docs)
    return docs
  
  def state_prompt(
    self,
    question: str,
    context: str
  ) -> str:
    return textwrap.dedent(
      f"Use the following context to answer the question.\n"
      f"If you don't know, say you don't know.\n"
      f"You must reference the context by adding <reference>\n"
      f"Question: {question}\n"
      f"Context: {context}\n"
      f"Answer:"
    )

  def index_documents(
    self,
    docs: list[Document]
  ) -> None:
    splits = self.splitter.split_documents(docs)
    self.vector_store.add_documents(splits)

  def retrieve(
    self,
    query: str,
    k: int = 4
  ) -> list[Document]:
    return self.vector_store.similarity_search(query, k=k)

  def build_prompt(
    self,
    question: str,
    context_docs: list[Document]
  ) -> str:
    docs_content = "\n\n".join(doc.page_content for doc in context_docs)

    return self.state_prompt(question, docs_content)

  def query(
    self,
    question: str,
    k: int = 4
  ) -> dict[str, object]:
    context = self.retrieve(question, k=k)
    prompt = self.build_prompt(question, context)
    return {"prompt": prompt, "context": context}
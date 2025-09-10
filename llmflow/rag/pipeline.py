import textwrap
from importlib import metadata

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
  
  def index_documents(
    self,
    docs: list[Document]
  ) -> None:
    splits = self.splitter.split_documents(docs)
    self.vector_store.add_documents(splits)
  
  def create(
    self,
    information: str,
    other_info: dict[str, str] = {},
    should_index: bool = True
  ) -> Document:
    doc = Document(
      page_content=information,
      metadata=other_info
    )

    if should_index:
      self.index_documents(docs=[doc])

    return doc

  def query(
    self,
    query: str,
    k: int = 4
  ) -> str:
    docs = self.vector_store.similarity_search(query, k=k)

    return "\n\n".join(doc.page_content for doc in docs)
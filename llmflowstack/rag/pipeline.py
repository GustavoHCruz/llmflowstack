import uuid

import chromadb
import chromadb.config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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
		checkpoint: str,
		collection_name: str = "rag_memory",
		persist_directory: str = "./chroma_store",
		chunk_size: int = 1000,
		chunk_overlap: int = 200
	) -> None:
		
		self.encoder = SentenceTransformer(checkpoint, trust_remote_code=True)

		client_settings = chromadb.config.Settings(
			anonymized_telemetry=False
		)

		self.vector_store = Chroma(
			collection_name=collection_name,
			embedding_function=EncoderWrapper(self.encoder),
			persist_directory=persist_directory,
			client_settings=client_settings
		)

		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			add_start_index=True,
		)
	
	def index_documents(
		self,
		docs: list[Document],
		ids: list[str]
	) -> None:
		splits = self.splitter.split_documents(docs)
		split_ids = [f"{ids[0]}_{i}" for i in range(len(splits))]
		self.vector_store.add_documents(splits, ids=split_ids)
	
	def create(
		self,
		information: str,
		other_info: dict[str, str] = {},
		doc_id: str | None = None,
		should_index: bool = True
	) -> Document:
		if doc_id is None:
			doc_id = str(uuid.uuid4())
			
		doc = Document(
			page_content=information,
			metadata={"id": doc_id, **other_info}
		)

		if should_index:
			self.index_documents([doc], ids=[doc_id])

		return doc
	
	def update(
		self,
		doc_id: str,
		new_information: str,
		other_info: dict[str, str] = {}
	) -> Document:
		self.vector_store.delete(ids=[doc_id])

		return self.create(
			information=new_information,
			other_info=other_info,
			doc_id=doc_id
		)
	
	def delete(
		self, doc_id: str
	) -> None:
		self.vector_store.delete(ids=[doc_id])

	def query(
		self,
		query: str,
		k: int = 4,
		category: str | None = None
	) -> str:
		if category:
			docs = self.vector_store.similarity_search(
				query, k=k, filter={"category": category}
			)
		else:
			docs = self.vector_store.similarity_search(query, k=k)

		return "\n\n".join(doc.page_content for doc in docs)
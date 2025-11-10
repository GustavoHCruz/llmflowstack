import gc
import uuid
from logging import getLogger

import chromadb
import chromadb.config
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


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
		vectors = self.model.encode(texts, task="retrieval", show_progress_bar=False)
		return vectors.tolist()
	
	def embed_query(
		self,
		text: str
	) -> list[float]:
		vectors = self.model.encode(text, task="retrieval", show_progress_bar=False)
		return vectors.tolist()

class VectorDatabase:
	encoder: SentenceTransformer | None = None
	collections: dict[str, Chroma] = {}

	def __init__(
		self,
		checkpoint: str | None = None,
		chunk_size: int = 1000,
		chunk_overlap: int = 200
	) -> None:
		self.logger = getLogger(f"LLMFlowStack.{self.__class__.__name__}")

		self.encoder = None
		if checkpoint:
			self.load_encoder(
				checkpoint=checkpoint
			)

		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			add_start_index=True,
		)
	
	def _log(
		self,
		message: str,
		level: LogLevel = LogLevel.INFO,
	) -> None:
		log_func = getattr(self.logger, level.lower(), None)
		if log_func:
			log_func(message)
		else:
			self.logger.info(message)
	
	def load_encoder(
		self,
		checkpoint: str
	) -> None:
		if self.encoder:
			self._log("A encoder is already loaded. Attempting to reset it.", LogLevel.WARNING)
			self.unload_encoder()

		self._log(f"Loading encoder on '{checkpoint}'")
		self.encoder = SentenceTransformer(
			checkpoint,
			trust_remote_code=True
		)

		self._log("Encoder loaded")
	
	def unload_encoder(
		self
	) -> None:
		try:
			del self.encoder
			gc.collect()
			torch.cuda.empty_cache()
			self.encoder = None
			self._log("Reset successfully.")
		except Exception as e:
			self._log("Couldn't reset encoder...", LogLevel.ERROR)
			self._log(f"{str(e)}", LogLevel.DEBUG)
	
	def get_collection(
		self,
		collection_name: str = "rag_memory",
		persist_directory: str | None = None
	) -> None:
		if not self.encoder:
			raise MissingEssentialProp("Could not find encoder.")

		client_settings = chromadb.config.Settings(
			anonymized_telemetry=False
		)

		self.collections[collection_name] = Chroma(
			collection_name=collection_name,
			embedding_function=EncoderWrapper(self.encoder),
			persist_directory=persist_directory,
			client_settings=client_settings
		)
	
	def validate_collection_name(
		self,
		collection_name: str
	) -> None:
		if collection_name not in self.collections:
			raise ValueError("Collection name not found in collection")

	def index_documents(
		self,
		collection_name: str,
		docs: list[Document],
		ids: list[str],
		can_split: bool = True,
	) -> None:
		self.validate_collection_name(
			collection_name=collection_name
		)

		for doc, src_id in zip(docs, ids):
			if doc.metadata is None:
				doc.metadata = {}
			doc.metadata["source_id"] = src_id

		if can_split:
			splits = self.splitter.split_documents(docs)
		else:
			splits = docs

		split_ids = []
		metadatas = []
		texts = []

		for i, s in enumerate(splits):
			src = s.metadata.get("source_id", "unknown")
			sid = f"{src}_{i}"
			split_ids.append(sid)
			metadatas.append(s.metadata.copy())
			texts.append(s.page_content)

		self.collections[collection_name].add_texts(
			texts=texts,
			ids=split_ids,
			metadatas=metadatas
		)
	
	def create(
		self,
		collection_name: str,
		information: str,
		other_info: dict[str, str] | None = None,
		doc_id: str | None = None,
		should_index: bool = True,
		can_split: bool = True
	) -> Document:
		if other_info is None:
			other_info = {}

		if doc_id is None:
			doc_id = str(uuid.uuid4())
			
		metadata = {"source_id": doc_id, **other_info}
		doc = Document(
			page_content=information,
			metadata=metadata
		)

		if should_index:
			self.index_documents(
				collection_name=collection_name,
				docs=[doc],
				ids=[doc_id],
				can_split=can_split
			)

		return doc
	
	def update(
		self,
		collection_name: str,
		doc_id: str,
		new_information: str,
		other_info: dict[str, str] | None = None
	) -> Document:
		self.validate_collection_name(
			collection_name=collection_name
		)

		if other_info is None:
			other_info = {}
	
		documents_to_delete = self.collections[collection_name].get(
			where={
				"source_id": doc_id
			}
		)

		ids_to_delete = documents_to_delete.get("ids", [])

		if ids_to_delete:
			self.collections[collection_name].delete(ids=ids_to_delete)

		return self.create(
			collection_name=collection_name,
			information=new_information,
			other_info=other_info,
			doc_id=doc_id
		)
	
	def delete(
		self,
		collection_name: str,
		doc_id: str
	) -> None:
		self.validate_collection_name(
			collection_name=collection_name
		)

		self.collections[collection_name].delete(ids=[doc_id])

	def rquery(
		self,
		collection_name: str,
		query: str,
		k: int = 4,
		filter: dict | None = None
	) -> list[Document]:
		self.validate_collection_name(
			collection_name=collection_name
		)
		
		return self.collections[collection_name].similarity_search(
			query=query,
			k=k,
			filter=filter
		)

	def query(
		self,
		collection_name: str,
		query: str,
		k: int = 4,
		filter: dict | None = None
	) -> str:
		self.validate_collection_name(
			collection_name=collection_name
		)

		if filter:
			docs = self.collections[collection_name].similarity_search(
				query=query,
				k=k,
				filter=filter
			)
		else:
			docs = self.collections[collection_name].similarity_search(query, k=k)

		return "\n\n".join(doc.page_content for doc in docs)
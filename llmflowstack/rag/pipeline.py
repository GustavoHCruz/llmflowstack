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
		vectors = self.model.encode(texts, task="retrieval", show_progress_bar=False)
		return vectors.tolist()
	
	def embed_query(
		self,
		text: str
	) -> list[float]:
		vectors = self.model.encode(text, task="retrieval", show_progress_bar=False)
		return vectors.tolist()

class RAGPipeline:
	"""
	A modular Retrieval-Augmented Generation (RAG) pipeline for embedding, indexing, and retrieving scientific or textual data using SentenceTransformers and Chroma as a vector store.

	Supports both persistent (disk-based) and transient (in-memory) modes depending on whether `persist_directory` is provided.
	"""
	def __init__(
		self,
		checkpoint: str,
		collection_name: str = "rag_memory",
		persist_directory: str | None = None,
		chunk_size: int = 1000,
		chunk_overlap: int = 200
	) -> None:
		"""
		Initializes the RAG pipeline.

		Args:
			checkpoint (str): Path or name of the SentenceTransformer checkpoint.
			collection_name (str): Name of the Chroma collection to create or load.
			persist_directory (str | None): Directory where the vector database is stored. If None, all data is kept in-memory and discarded after the session ends.
			chunk_size (int): Maximum size (in characters) for text chunks during indexing.
			chunk_overlap (int): Overlap (in characters) between consecutive text chunks.
		"""
		self.encoder = SentenceTransformer(checkpoint, trust_remote_code=True)

		client_settings = chromadb.config.Settings(
			anonymized_telemetry=False
		)

		self.collection = Chroma(
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
		ids: list[str],
		can_split: bool = True
	) -> None:
		"""
		Indexes a list of documents into the Chroma vector store.

		Each document is assigned a unique `source_id` and, optionally, split into smaller chunks for more granular retrieval. Each resulting chunk is embedded and stored with its metadata for later similarity search.

		Args:
			docs (list[Document]): List of LangChain `Document` objects to index.
			ids (list[str]): Unique identifiers corresponding to each document.
			can_split (bool): Whether to split documents into smaller chunks before
				indexing. Set to False to index each document as a single entry
				(e.g., for short or self-contained texts).

		Returns:
			None
		"""
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

		self.collection.add_texts(
			texts=texts,
			ids=split_ids,
			metadatas=metadatas
		)
	
	def create(
		self,
		information: str,
		other_info: dict[str, str] | None = None,
		doc_id: str | None = None,
		should_index: bool = True,
		can_split: bool = True
	) -> Document:
		"""
		Creates a new `Document` and optionally indexes it in the collection.

		This is a convenience method that wraps both document creation and embedding/indexing in one step. Metadata fields are merged into the document and can include any descriptive information (e.g., title, DOI, year).

		Args:
			information (str): Main textual content of the document.
			other_info (dict[str, str] | None): Optional metadata fields to include.
			doc_id (str | None): Custom document identifier. If None, a UUID is generated.
			should_index (bool): Whether to immediately add the document to the vector store.
			can_split (bool): Whether to allow splitting before indexing.

		Returns:
			Document: The created LangChain `Document` object (indexed if specified).
		"""
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
				docs=[doc],
				ids=[doc_id],
				can_split=can_split
			)

		return doc
	
	def update(
		self,
		doc_id: str,
		new_information: str,
		other_info: dict[str, str] | None = None
	) -> Document:
		"""
		Updates an existing document in the collection with new content and metadata.

		All vector entries associated with the provided `doc_id` are deleted, and a new document is created and re-indexed in their place. This ensures that embeddings remain consistent with the latest text content.

		Args:
			doc_id (str): Identifier of the document to update.
			new_information (str): Updated text content for the document.
			other_info (dict[str, str] | None): Optional new metadata to associate.

		Returns:
			Document: The newly created (updated) `Document` object.
		"""
		if other_info is None:
			other_info = {}
	
		documents_to_delete = self.collection.get(
			where={
				"source_id": doc_id
			}
		)

		ids_to_delete = documents_to_delete.get("ids", [])

		if ids_to_delete:
			self.collection.delete(ids=ids_to_delete)

		return self.create(
			information=new_information,
			other_info=other_info,
			doc_id=doc_id
		)
	
	def delete(
		self,
		doc_id: str
	) -> None:
		"""
		Deletes all indexed entries associated with a specific document ID.

		Removes all vectors and metadata tied to the provided `doc_id` from the collection. Use this to completely erase a document's content from the indexed database.

		Args:
			doc_id (str): Identifier of the document to delete.

		Returns:
			None
		"""
		self.collection.delete(ids=[doc_id])

	def rquery(
		self,
		query: str,
		k: int = 4,
		filter: dict | None = None
	) -> list[Document]:
		"""
		Perform a **raw semantic search** on the collection.

		This method queries the vector store using the provided text query and returns the top-`k` most similar `Document` objects, optionally filtered by metadata.

		Args:
			query (str): The natural-language query text to embed and search for.
			k (int, optional): Number of top results to return. Defaults to 4.
			filter (dict | None, optional): Metadata filter applied during search
				(e.g., {"type": "article"}). Defaults to None.

		Returns:
			list[Document]: A list of matching documents sorted by similarity score.
		"""
		return self.collection.similarity_search(
			query=query,
			k=k,
			filter=filter
		)

	def query(
		self,
		query: str,
		k: int = 4,
		filter: dict | None = None
	) -> str:
		"""
		Perform a **semantic search** and return the combined text content.

		This method wraps `rquery()` and concatenates the retrieved document contents into a single string, suitable for direct use in downstream LLM prompts or text processing.

		Args:
			query (str): The natural-language query text to search for.
			k (int, optional): Number of top results to return. Defaults to 4.
			filter (dict | None, optional): Metadata filter applied during search. If None, all documents are considered.

		Returns:
			str: A newline-separated string containing the page contents of
			the retrieved documents.
		"""
		if filter:
			docs = self.collection.similarity_search(
				query=query,
				k=k,
				filter=filter
			)
		else:
			docs = self.collection.similarity_search(query, k=k)

		return "\n\n".join(doc.page_content for doc in docs)
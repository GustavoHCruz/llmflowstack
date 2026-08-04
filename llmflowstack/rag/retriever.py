from __future__ import annotations

import uuid
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from langchain_core.documents import Document

from llmflowstack.rag.document_splitter import DocumentSplitter
from llmflowstack.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        *,
        splitter: DocumentSplitter | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.splitter = splitter

    def create_document(
        self,
        information: str,
        *,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
    ) -> Document:
        if not information.strip():
            raise ValueError("Document information cannot be empty.")

        source_id = source_id or str(uuid.uuid4())

        document_metadata = dict(metadata or {})
        document_metadata["source_id"] = source_id

        return Document(
            page_content=information,
            metadata=document_metadata,
        )

    def insert_document(
        self,
        information: str,
        *,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
        should_index: bool = True,
        can_split: bool = True,
        replace_existing: bool = False,
    ) -> Document:
        document = self.create_document(
            information=information,
            metadata=metadata,
            source_id=source_id,
        )

        if should_index:
            self.index_documents(
                documents=[document],
                can_split=can_split,
                replace_existing=replace_existing,
            )

        return document

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        source_ids: Sequence[str] | None = None,
        can_split: bool = True,
        replace_existing: bool = False,
    ) -> list[str]:
        documents = list(documents)

        if not documents:
            return []

        prepared_documents = self._prepare_documents(
            documents=documents,
            source_ids=source_ids,
        )

        if replace_existing:
            unique_source_ids = {
                str(document.metadata["source_id"]) for document in prepared_documents
            }

            for source_id in unique_source_ids:
                self.vector_store.delete_source(source_id)

        chunks = self._split_documents(
            documents=prepared_documents,
            can_split=can_split,
        )

        chunk_ids = self._assign_chunk_metadata_and_ids(chunks)

        self.vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        return chunk_ids

    def update_document(
        self,
        source_id: str,
        new_information: str,
        *,
        metadata: dict[str, Any] | None = None,
        can_split: bool = True,
    ) -> Document:
        if not source_id:
            raise ValueError("source_id cannot be empty.")

        self.vector_store.delete_source(source_id)

        try:
            return self.insert_document(
                information=new_information,
                metadata=metadata,
                source_id=source_id,
                should_index=True,
                can_split=can_split,
            )
        except Exception:
            raise

    def delete_document(
        self,
        source_id: str,
    ) -> int:
        return self.vector_store.delete_source(source_id)

    def delete_where(
        self,
        where: dict[str, Any],
    ) -> int:
        return self.vector_store.delete_where(where)

    def retrieve(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        return self.vector_store.search(
            query=query,
            k=k,
            filter=filter,
        )

    def retrieve_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        return self.vector_store.search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def as_langchain_retriever(
        self,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        search_type: str = "similarity",
        score_threshold: float | None = None,
    ):
        search_kwargs: dict[str, Any] = {
            "k": k,
        }

        if filter is not None:
            search_kwargs["filter"] = filter

        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        return self.vector_store.store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def _prepare_documents(
        self,
        documents: Sequence[Document],
        source_ids: Sequence[str] | None,
    ) -> list[Document]:
        if source_ids is not None:
            source_ids = list(source_ids)

            if len(documents) != len(source_ids):
                raise ValueError(
                    "The number of documents must match the number of source IDs."
                )

        prepared: list[Document] = []

        for index, document in enumerate(documents):
            metadata = dict(document.metadata or {})

            explicit_source_id = source_ids[index] if source_ids is not None else None

            source_id = (
                explicit_source_id or metadata.get("source_id") or str(uuid.uuid4())
            )

            metadata["source_id"] = str(source_id)

            prepared.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return prepared

    def _split_documents(
        self,
        documents: Sequence[Document],
        can_split: bool,
    ) -> list[Document]:
        if not can_split:
            return list(documents)

        if self.splitter is None:
            raise RuntimeError(
                "can_split=True, but no DocumentSplitter was configured."
            )

        return self.splitter.split_documents(documents)

    def _assign_chunk_metadata_and_ids(
        self,
        chunks: Sequence[Document],
    ) -> list[str]:
        source_counters: dict[str, int] = {}
        chunk_ids: list[str] = []

        for chunk in chunks:
            metadata = dict(chunk.metadata or {})

            source_id = metadata.get("source_id")

            if source_id is None:
                raise ValueError("Every chunk must contain a source_id.")

            source_id = str(source_id)

            chunk_index = source_counters.get(source_id, 0)
            source_counters[source_id] = chunk_index + 1

            content_hash = sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]

            chunk_id = f"{source_id}:chunk:{chunk_index}:{content_hash}"

            metadata["source_id"] = source_id
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = chunk_index

            chunk.metadata = metadata
            chunk_ids.append(chunk_id)

        if len(set(chunk_ids)) != len(chunk_ids):
            raise RuntimeError("Duplicate chunk IDs were generated.")

        return chunk_ids

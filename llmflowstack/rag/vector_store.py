from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chromadb.config
from langchain_chroma import Chroma
from langchain_core.documents import Document

from llmflowstack.rag.embedding_model import EmbeddingModel


class VectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        *,
        persist_directory: str | None = None,
        collection_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty.")

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

        client_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
        )

        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory,
            client_settings=client_settings,
            collection_metadata=collection_metadata,
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
    ) -> list[str]:
        documents = list(documents)
        ids = list(ids)

        if not documents:
            return []

        if len(documents) != len(ids):
            raise ValueError("The number of documents must match the number of IDs.")

        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")

        return self.store.add_documents(
            documents=documents,
            ids=ids,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        self._validate_query(
            query=query,
            k=k,
        )

        return self.store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        self._validate_query(
            query=query,
            k=k,
        )

        return self.store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.store.get(
            ids=list(ids) if ids is not None else None,
            where=where,
        )

    def get_by_source_id(
        self,
        source_id: str,
    ) -> dict[str, Any]:
        if not source_id:
            raise ValueError("source_id cannot be empty.")

        return self.get(
            where={"source_id": source_id},
        )

    def delete_ids(
        self,
        ids: Sequence[str],
    ) -> None:
        ids = list(ids)

        if not ids:
            return

        self.store.delete(ids=ids)

    def delete_where(
        self,
        where: dict[str, Any],
    ) -> int:
        if not where:
            raise ValueError("A metadata filter is required for delete_where().")

        result = self.get(where=where)
        ids = result.get("ids", [])

        if ids:
            self.delete_ids(ids)

        return len(ids)

    def delete_source(
        self,
        source_id: str,
    ) -> int:
        if not source_id:
            raise ValueError("source_id cannot be empty.")

        return self.delete_where(
            where={"source_id": source_id},
        )

    def delete_collection(self) -> None:
        self.store.delete_collection()

    def count(self) -> int:
        collection = getattr(self.store, "_collection", None)

        if collection is None:
            raise RuntimeError("Could not access the underlying Chroma collection.")

        return collection.count()

    @staticmethod
    def _validate_query(
        query: str,
        k: int,
    ) -> None:
        if not query.strip():
            raise ValueError("The query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

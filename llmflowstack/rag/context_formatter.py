from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document

DocumentFormatter = Callable[[Document, int], str]
ScoredDocumentFormatter = Callable[[Document, float, int], str]


class ContextFormatter:
    def __init__(
        self,
        *,
        document_separator: str = "\n\n",
        include_metadata: bool = False,
        include_source_id: bool = False,
        include_chunk_index: bool = False,
    ) -> None:
        self.document_separator = document_separator
        self.include_metadata = include_metadata
        self.include_source_id = include_source_id
        self.include_chunk_index = include_chunk_index

    def format_documents(
        self,
        documents: Sequence[Document],
        *,
        formatter: DocumentFormatter | None = None,
    ) -> str:
        formatted_documents: list[str] = []

        for index, document in enumerate(documents):
            if formatter is not None:
                formatted = formatter(document, index)
            else:
                formatted = self._format_document(
                    document=document,
                    index=index,
                )

            if formatted.strip():
                formatted_documents.append(formatted.strip())

        return self.document_separator.join(formatted_documents)

    def format_scored_documents(
        self,
        documents: Sequence[tuple[Document, float]],
        *,
        formatter: ScoredDocumentFormatter | None = None,
        include_score: bool = False,
    ) -> str:
        formatted_documents: list[str] = []

        for index, (document, score) in enumerate(documents):
            if formatter is not None:
                formatted = formatter(
                    document,
                    score,
                    index,
                )
            else:
                formatted = self._format_document(
                    document=document,
                    index=index,
                    score=score if include_score else None,
                )

            if formatted.strip():
                formatted_documents.append(formatted.strip())

        return self.document_separator.join(formatted_documents)

    def _format_document(
        self,
        document: Document,
        index: int,
        score: float | None = None,
    ) -> str:
        header_parts: list[str] = []

        if self.include_source_id:
            source_id = document.metadata.get("source_id")

            if source_id is not None:
                header_parts.append(f"source_id={source_id}")

        if self.include_chunk_index:
            chunk_index = document.metadata.get("chunk_index")

            if chunk_index is not None:
                header_parts.append(f"chunk_index={chunk_index}")

        if score is not None:
            header_parts.append(f"score={score:.6f}")

        if self.include_metadata:
            metadata = self._format_metadata(document.metadata)

            if metadata:
                header_parts.append(metadata)

        if not header_parts:
            return document.page_content

        header = " | ".join(header_parts)

        return f"[Document {index + 1}: {header}]\n{document.page_content}"

    @staticmethod
    def _format_metadata(
        metadata: dict[str, Any],
    ) -> str:
        ignored_keys = {
            "source_id",
            "chunk_id",
            "chunk_index",
        }

        fields = [
            f"{key}={value}"
            for key, value in sorted(metadata.items())
            if key not in ignored_keys
        ]

        return " | ".join(fields)

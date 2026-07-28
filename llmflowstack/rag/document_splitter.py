from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        add_start_index: bool = True,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        kwargs: dict = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "add_start_index": add_start_index,
        }

        if separators is not None:
            kwargs["separators"] = separators

        self.splitter = RecursiveCharacterTextSplitter(**kwargs)

    def split_document(
        self,
        document: Document,
    ) -> list[Document]:
        return self.split_documents([document])

    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        if not documents:
            return []

        return self.splitter.split_documents(list(documents))

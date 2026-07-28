from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

FloatArray = NDArray[np.floating[Any]]


class EmbeddingModel(Embeddings, ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    def close(self) -> None: ...


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(
        self,
        checkpoint: str,
        *,
        trust_remote_code: bool = False,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        task: str | None = "retrieval",
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.task = task

        kwargs = dict(model_kwargs or {})

        self.model: SentenceTransformer | None = SentenceTransformer(
            checkpoint,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @property
    def dimension(self) -> int:
        model = self._require_model()
        dimension = model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "The embedding model did not report its output dimension."
            )

        return dimension

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._encode_documents(texts)

        return vectors.astype(float, copy=False).tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError("The query cannot be empty.")

        vector = self._encode_query(text)

        if vector.ndim == 2:
            if vector.shape[0] != 1:
                raise RuntimeError("A single query produced more than one embedding.")

            vector = vector[0]

        if vector.ndim != 1:
            raise RuntimeError(
                f"Expected a one-dimensional query embedding, "
                f"but received shape {vector.shape}."
            )

        return vector.astype(float, copy=False).tolist()

    def close(self) -> None:
        if self.model is None:
            return

        self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self) -> SentenceTransformer:
        if self.model is None:
            raise RuntimeError("The embedding model has already been unloaded.")

        return self.model

    def _encode_documents(
        self,
        texts: Sequence[str],
    ) -> FloatArray:
        model = self._require_model()

        kwargs = self._common_encode_kwargs()

        if self.document_prompt_name is not None:
            kwargs["prompt_name"] = self.document_prompt_name

        result = self._invoke_encode(
            model=model,
            method_name="encode_document",
            inputs=list(texts),
            kwargs=kwargs,
        )

        array = np.asarray(result)

        if array.ndim != 2:
            raise RuntimeError(
                f"Expected document embeddings with two dimensions, "
                f"but received shape {array.shape}."
            )

        return array

    def _encode_query(
        self,
        text: str,
    ) -> FloatArray:
        model = self._require_model()

        kwargs = self._common_encode_kwargs()

        if self.query_prompt_name is not None:
            kwargs["prompt_name"] = self.query_prompt_name

        result = self._invoke_encode(
            model=model,
            method_name="encode_query",
            inputs=text,
            kwargs=kwargs,
        )

        return np.asarray(result)

    def _common_encode_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
        }

        if self.task is not None:
            kwargs["task"] = self.task

        return kwargs

    @staticmethod
    def _invoke_encode(
        model: SentenceTransformer,
        method_name: str,
        inputs: str | list[str],
        kwargs: dict[str, Any],
    ) -> Any:
        specialized_method = getattr(model, method_name, None)

        if callable(specialized_method):
            method = cast(
                Callable[..., Any],
                specialized_method,
            )

            try:
                return method(inputs, **kwargs)
            except TypeError:
                reduced_kwargs = dict(kwargs)
                reduced_kwargs.pop("task", None)
                reduced_kwargs.pop("prompt_name", None)

                try:
                    return method(inputs, **reduced_kwargs)
                except TypeError:
                    pass

        encode = cast(
            Callable[..., Any],
            model.encode,
        )

        try:
            return encode(inputs, **kwargs)
        except TypeError:
            reduced_kwargs = dict(kwargs)
            reduced_kwargs.pop("task", None)
            reduced_kwargs.pop("prompt_name", None)

            return encode(inputs, **reduced_kwargs)

    def __enter__(self) -> SentenceTransformerEmbedding:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

from dataclasses import asdict, dataclass
from typing import Literal

from transformers import GenerationConfig, TextIteratorStreamer


@dataclass
class TrainParams:
	batch_size: int = 1
	gradient_accumulation: int = 8
	epochs: int = 1
	warmup_ratio: float = 0.0
	lr: float = 2e-5
	optim: Literal[
		"adamw_torch",
		"adamw_torch_fused",
		"sgd"
	] = "adamw_torch"
	logging_steps: int = 1
	label_smoothing_factor: float = 0
	lr_scheduler_type: Literal["linear", "cosine_with_min_lr"] = "linear"

@dataclass
class GenerationParams:
	max_new_tokens: int | None = None
	repetition_penalty: float | None = None
	streamer: TextIteratorStreamer | None = None

	mode: Literal["greedy", "sample", "beam"] = "greedy"

	temperature: float | None = None
	top_p: float | None = None
	typical_p: float | None = None

	num_beams: int | None = None
	length_penalty: float | None = None
	early_stopping: bool | None = None

	def __post_init__(self):
		if self.max_new_tokens is not None and self.max_new_tokens <= 0:
			raise ValueError("max_new_tokens must be > 0")

		if self.repetition_penalty is not None and self.repetition_penalty < 1.0:
			raise ValueError("repetition_penalty must be >= 1.0")

		if self.mode == "greedy":
			self._ensure_none(
				"greedy",
				["temperature", "top_p", "typical_p", "num_beams", "length_penalty", "early_stopping"]
			)

		elif self.mode == "sample":
			self._ensure_none(
				"sample",
				["num_beams", "length_penalty", "early_stopping"]
			)

			if self.temperature is not None and self.temperature <= 0:
				raise ValueError("temperature must be > 0")

			if self.top_p is not None and not (0 < self.top_p <= 1):
				raise ValueError("top_p must be in (0, 1]")

			if self.typical_p is not None and not (0 < self.typical_p <= 1):
				raise ValueError("typical_p must be in (0, 1]")

		elif self.mode == "beam":
			self._ensure_none(
				"beam",
				["temperature", "top_p", "typical_p"]
			)

			if self.num_beams is None or self.num_beams < 2:
				raise ValueError("beam search requires num_beams >= 2")

			if self.length_penalty is not None and self.length_penalty <= 0:
				raise ValueError("length_penalty must be > 0")

	def _ensure_none(self, mode: str, fields: list[str]) -> None:
		for field in fields:
			if getattr(self, field) is not None:
				raise ValueError(
					f"Field '{field}' is not allowed when mode='{mode}'"
				)

	def to_generation_config(self) -> GenerationConfig:
		data = asdict(self)

		data.pop("mode")
		data.pop("streamer")

		if self.mode == "greedy":
			data["do_sample"] = False
			data["num_beams"] = 1

		elif self.mode == "sample":
			data["do_sample"] = True
			data["num_beams"] = 1

		elif self.mode == "beam":
			data["do_sample"] = False
			data["num_beams"] = self.num_beams

		data = {k: v for k, v in data.items() if v is not None}

		return GenerationConfig(**data)
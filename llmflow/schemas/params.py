from dataclasses import dataclass, field
from typing import Literal

from transformers import TextIteratorStreamer


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
  logging_steps=1

@dataclass
class GenerationBeamsParams:
  num_beams: int | None = None
  length_penalty: float | None = None
  early_stopping: bool | None = None

@dataclass
class GenerationSampleParams:
  temperature: float | None = None
  top_p: float | None = None
  typical_p: float | None = None

@dataclass
class GenerationParams:
  max_new_tokens: int | None = None
  repetition_penalty: float | None = None
  sample: GenerationSampleParams = field(default_factory=GenerationSampleParams)
  beams: GenerationBeamsParams = field(default_factory=GenerationBeamsParams)
  streamer: TextIteratorStreamer | None = None
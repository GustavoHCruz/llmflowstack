import threading
from functools import partial
from time import time
from typing import Iterator, cast

import torch
from transformers import (AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteriaList, TextIteratorStreamer)
from transformers.models.gpt2 import GPT2LMHeadModel

from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.decoders.base_decoder import BaseDecoder, Input
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.generation_utils import create_generation_params
from llmflowstack.utils.logging import LogLevel


class GPT_2(BaseDecoder):
	model: GPT2LMHeadModel | None = None

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: bool | None = None,
		seed: int | None = None
	) -> None:
		return super().__init__(
			checkpoint=checkpoint,
			quantization=quantization,
			seed=seed
		)

	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		if not self.tokenizer:
			self._log("Could not set stop tokens - generation may not work...", LogLevel.WARNING)
			return None
		self.stop_token_ids = tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: bool | None = False
	) -> None:
		quantization_config = None
		if quantization == "4bit":
			quantization_config = BitsAndBytesConfig(
				load_in_4bit=True
			)
		if quantization == "8bit":
			quantization_config = BitsAndBytesConfig(
				load_in_8bit=True
			)

		self.model = GPT2LMHeadModel.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			dtype="auto",
			device_map="auto"
		)

	def generate(
		self,
		input_text: str,
		params: GenerationParams | None = None
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None

		self._log(f"Processing received input...'")

		if params is None:
			params = GenerationParams(max_new_tokens=1024)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 1024

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params

		tokenized_input = self._tokenize(data=Input(input_text=input_text))

		input_ids = tokenized_input["input_ids"]
		attention_mask = tokenized_input["attention_mask"]

		self.model.eval()
		self.model.gradient_checkpointing_disable()
		start = time()

		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids.unsqueeze(0),
				attention_mask=attention_mask.unsqueeze(0),
				use_cache=True,
				eos_token_id=None,
				stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids)])
			)

		answer = self.tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		return answer.strip()
	
	def generate_stream(
		self,
		input_text: str,
		params: GenerationParams | None = None
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			if False:
				yield ""
			return
		
		self._log(f"Processing received input...'")
		
		if params is None:
			params = GenerationParams(max_new_tokens=1024)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 1024

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params
		
		tokenized_input = self._tokenize(data=Input(input_text=input_text))

		input_ids = tokenized_input["input_ids"]
		attention_mask = tokenized_input["attention_mask"]

		streamer = TextIteratorStreamer(
			cast(AutoTokenizer, self.tokenizer),
			skip_prompt=True,
			skip_special_tokens=True
		)

		generate_fn = partial(
			self.model.generate,
			input_ids=input_ids.unsqueeze(0),
			attention_mask=attention_mask.unsqueeze(0),
			use_cache=True,
			eos_token_id=None,
			streamer=streamer,
			stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids)])
		)

		start = time()

		thread = threading.Thread(target=generate_fn)
		thread.start()

		for new_text in streamer:
			yield new_text
		
		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")
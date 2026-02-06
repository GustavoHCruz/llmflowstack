import threading
from functools import partial
from time import time
from typing import Iterator, cast

import torch
from transformers import (AutoTokenizer, BitsAndBytesConfig,
                          TextIteratorStreamer)
from transformers.models.gpt2 import GPT2LMHeadModel

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.logging import LogLevel


class Gpt2(BaseDecoder):
	model: GPT2LMHeadModel | None = None
	max_context_len = 1024

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
	
	def _build_prompt(
		self,
		*args,
		**kwargs
	) -> str:
		...

	def build_input(
		self,
		input_text: str,
		output_text: str | None = None
	) -> ModelInput:
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=False
		)

	def generate(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None

		model_input = self._prepare_generation(
			data=data,
			params=params,
			follow_prompt_format=False
		)

		if model_input is None:
			return None
		
		input_ids = model_input.input_ids.unsqueeze(0).to(self.model.device)
		attention_mask = model_input.attention_mask.unsqueeze(0).to(self.model.device)

		self.model.eval()
		self.model.gradient_checkpointing_disable()
		start = time()

		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				use_cache=True,
				eos_token_id=self.stop_token_ids
			)

		answer = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

		if isinstance(answer, list):
			answer = answer[0]

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		return answer.strip()
	
	def generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			if False:
				yield ""
			return
		
		model_input = self._prepare_generation(
			data=data,
			params=params,
			follow_prompt_format=False
		)

		if model_input is None:
			return None
		
		input_ids = model_input.input_ids.unsqueeze(0).to(self.model.device)
		attention_mask = model_input.attention_mask.unsqueeze(0).to(self.model.device)

		streamer = TextIteratorStreamer(
			cast(AutoTokenizer, self.tokenizer),
			skip_prompt=True,
			skip_special_tokens=True
		)

		generate_fn = partial(
			self.model.generate,
			input_ids=input_ids,
			attention_mask=attention_mask,
			use_cache=True,
			eos_token_id=self.stop_token_ids,
			pad_token_id=self.tokenizer.pad_token_id,
			streamer=streamer
		)

		start = time()

		thread = threading.Thread(target=generate_fn)
		thread.start()

		for new_text in streamer:
			yield new_text
		
		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")
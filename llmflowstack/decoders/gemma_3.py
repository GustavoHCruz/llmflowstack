import threading
from functools import partial
from time import time
from typing import Iterator, Literal, cast

import torch
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Gemma3(BaseDecoder):
	model: Gemma3ForCausalLM | None = None
	max_context_len = 32768
	legacy_trainer = True

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: Literal["4bit"] | None = None,
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
		particular_tokens = self.tokenizer.encode("<end_of_turn>")
		self.stop_token_ids = tokens + particular_tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: Literal["4bit"] | None = None
	) -> None:
		quantization_config = None
		if quantization == "4bit":
			quantization_config = BitsAndBytesConfig(
				load_in_4bit=True
			)

		self.model = Gemma3ForCausalLM.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			dtype="auto",
			device_map="auto",
			attn_implementation="eager"
		)
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: Literal['4bit'] | None = None
	) -> None:
		return super().load_checkpoint(checkpoint, quantization)

	def _build_prompt(
		self,
		input_text: str,
		output_text: str | None = None,
		system_message: str | None = None,
		image_paths: list[str] | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_message = system_message or ""
		if not system_message:
			system_message = ""

		if system_message:
			system_message = f"{system_message}\n"

		expected_answer = output_text
		answer = f"{expected_answer}<end_of_turn>" if expected_answer else ""
	
		return (
			f"<start_of_turn>user"
			f"{system_message}\n{input_text}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: str,
		output_text: str | None = None,
		system_message: str | None = None,
		image_paths: list[str] | None = None
	) -> ModelInput:
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=True,
			system_message=system_message,
			image_paths=image_paths
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
			follow_prompt_format=True
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
				eos_token_id=None,
				stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids)])
			)

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		response = outputs[0][input_ids.shape[1]:]

		decoded = self.tokenizer.decode(response, skip_special_tokens=True)

		if isinstance(decoded, list):
			decoded = decoded[0]

		return decoded
	
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
			follow_prompt_format=True
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
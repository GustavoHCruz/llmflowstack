import threading
from dataclasses import dataclass
from functools import partial
from time import time
from typing import Iterator, Literal, cast

import torch
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.llama import LlamaForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.decoders_it.base_instruct_decoder import (
    BaseInstructDecoder, BaseInstructInput)
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


@dataclass
class Input(BaseInstructInput):
	system_message: str | None = None

class Llama_3_it(BaseInstructDecoder):
	model: LlamaForCausalLM | None = None
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]
	max_new_tokens = 8192

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: Literal["4bit", "8bit"] | None = None,
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
		particular_tokens = self.tokenizer.encode("<|eot_id|>")
		self.stop_token_ids = tokens + particular_tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: Literal["4bit", "8bit"] | None = None
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

		self.model = LlamaForCausalLM.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			dtype="auto",
			device_map="auto",
			attn_implementation="eager"
		)
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: Literal['4bit', "8bit"] | None = None
	) -> None:
		return super().load_checkpoint(checkpoint, quantization)

	def _build_input(
		self,
		data: Input
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		expected_answer = data.expected_answer
		answer = f"{expected_answer}{self.tokenizer.eos_token}" if expected_answer else ""

		system_message = data.system_message or ""

		return (
			f"<|start_header_id|>system<|end_header_id|>{system_message}\n"
			f"<|eot_id|><|start_header_id|>user<|end_header_id|>{data.input_text}\n"
			f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}"
		)

	def build_input(
		self,
		input_text: str,
		system_message: str | None = None,
		expected_answer: str | None = None
	) -> Input:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return Input(
			input_text=input_text,
			system_message=system_message,
			expected_answer=expected_answer
		)

	def generate(
		self,
		input: Input | str,
		params: GenerationParams | None = None
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None
		
		prep = self._prepare_generation(
			input,
			params=params
		)

		if prep is None:
			return None
		
		input_ids, attention_mask = prep

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

		return self.tokenizer.decode(response, skip_special_tokens=True)
	
	def generate_stream(
		self,
		input: Input | str,
		params: GenerationParams | None = None
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			if False:
				yield ""
			return
		
		prep = self._prepare_generation(
			input,
			params=params
		)

		if prep is None:
			return None
		
		input_ids, attention_mask = prep

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
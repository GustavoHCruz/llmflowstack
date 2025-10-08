import textwrap
import threading
from time import time
from typing import Iterator, Literal, TypedDict, cast

import torch
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.llama import LlamaForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.base.base import BaseModel
from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.generation_utils import create_generation_params


class LLaMA3Input(TypedDict):
	input_text: str
	expected_answer: str | None
	system_message: str | None

class LLaMA3(BaseModel):
	model: LlamaForCausalLM | None = None
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]

	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		if not self.tokenizer:
			self._log("Could not set stop tokens - generation may not work...", "WARNING")
			return None
		particular_tokens = self.tokenizer.encode("<|eot_id|>")
		self.stop_token_ids = tokens + particular_tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: Literal["8bit", "4bit"] | bool | None = None
	) -> None:
		quantization_config = None
		if quantization == "4bit":
			quantization_config = BitsAndBytesConfig(
				load_in_4bit=True
			)
			self.model_is_quantized = True
		if quantization == "8bit":
			quantization_config = BitsAndBytesConfig(
				load_in_8bit=True
			)
			self.model_is_quantized = True

		self.model = LlamaForCausalLM.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			dtype="auto",
			device_map="auto",
			attn_implementation="eager"
		)

	def _build_input(
		self,
		input_text: str,
		expected_answer: str | None = None,
		system_message: str | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		answer = f"{expected_answer}{self.tokenizer.eos_token}" if expected_answer else ""

		return textwrap.dedent(
			f"<|start_header_id|>system<|end_header_id|>{system_message or ""}\n"
			f"<|eot_id|><|start_header_id|>user<|end_header_id|>{input_text}\n"
			f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}"
		)

	def build_input(
		self,
		input_text: str,
		system_message: str | None = None,
		expected_answer: str | None = None
	) -> LLaMA3Input:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return {
			"input_text": input_text,
			"system_message": system_message,
			"expected_answer": expected_answer
		}

	def generate(
		self,
		input: LLaMA3Input | str,
		params: GenerationParams | None = None
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			return None

		self.model

		self._log(f"Processing received input...'")

		if params is None:
			params = GenerationParams(max_new_tokens=8192)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 8192

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params

		if params:
			generation_params = create_generation_params(params)
			self.model.generation_config = generation_params

		model_input = None
		if isinstance(input, str):
			model_input = self._build_input(
				input_text=input
			)
		else:
			model_input = self._build_input(
				input_text=input["input_text"],
				system_message=input.get("system_message", "")
			)

		tokenized_input = self._tokenize(model_input)

		input_ids, attention_mask = tokenized_input

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
		input: LLaMA3Input | str,
		params: GenerationParams | None = None
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			if False:
				yield ""
			return
		
		if params is None:
			params = GenerationParams(max_new_tokens=8192)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 8192

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params

		if isinstance(input, str):
			model_input = self._build_input(
				input_text=input
			)
		else:
			model_input = self._build_input(
				input_text=input["input_text"],
				system_message=input.get("system_message")
			)
		
		tokenized_input = self._tokenize(model_input)
		input_ids, attention_mask = tokenized_input

		streamer = TextIteratorStreamer(
			cast(AutoTokenizer, self.tokenizer),
			skip_prompt=True,
			skip_special_tokens=True
		)

		def _generate() -> None:
			assert self.model is not None
			with torch.no_grad():
				self.model.generate(
					input_ids=input_ids,
					attention_mask=attention_mask,
					use_cache=True,
					eos_token_id=None,
					streamer=streamer,
					stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids)])
				)
		
		thread = threading.Thread(target=_generate)
		thread.start()

		for new_text in streamer:
			yield new_text
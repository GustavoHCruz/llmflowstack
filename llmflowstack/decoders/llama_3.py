import threading
from functools import partial
from time import time
from typing import Iterator, Literal, cast

import torch
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.llama import LlamaForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Llama3(BaseDecoder):
	model: LlamaForCausalLM | None = None
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]
	max_context_len = 8192

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

	def _build_prompt(
		self,
		input_text: str,
		output_text: str | None = None,
		system_message: str | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		answer = f"{output_text}{self.tokenizer.eos_token}" if output_text else ""

		return (
			f"<|start_header_id|>system<|end_header_id|>{system_message or ""}\n"
			f"<|eot_id|><|start_header_id|>user<|end_header_id|>{input_text}\n"
			f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}"
		)

	def build_input(
		self,
		input_text: str,
		output_text: str | None = None,
		follow_prompt_format: bool = True,
		system_message: str | None = None
	) -> ModelInput:
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=follow_prompt_format,
			system_message=system_message
		)

	def generate(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		follow_prompt_format: bool = True
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None
		
		model_input = self._prepare_generation(
			data=data,
			params=params,
			follow_prompt_format=follow_prompt_format
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

		answer = outputs[0][input_ids.shape[1]:]

		decoded = self.tokenizer.decode(answer, skip_special_tokens=True)

		if isinstance(decoded, list):
			decoded = decoded[0]

		return decoded
	
	def generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		follow_prompt_format: bool = True
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			if False:
				yield ""
			return
		
		model_input = self._prepare_generation(
			data=data,
			params=params,
			follow_prompt_format=follow_prompt_format
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
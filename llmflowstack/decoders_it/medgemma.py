import threading
from dataclasses import dataclass
from functools import partial
from time import time
from typing import Iterator, Literal, cast

import torch
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.gemma3 import Gemma3ForCausalLM
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

class MedGemma(BaseInstructDecoder):
	model: Gemma3ForCausalLM | None = None
	can_think = False
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]
	max_new_tokens = 32768

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
		quantization:  Literal["4bit"] | None = None
	) -> None:
		return super().load_checkpoint(checkpoint, quantization)

	def _build_input(
		self,
		data: Input
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_message = data.system_message or ""
		if not system_message:
			system_message = ""
		if self.can_think:
			system_message += f"think silently if needed. {system_message}"

		if system_message:
			system_message = f"{system_message}\n"

		expected_answer = data.expected_answer
		answer = f"{expected_answer}<end_of_turn>" if expected_answer else ""
	
		return (
			f"<start_of_turn>user"
			f"{system_message}\n{data.input_text}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: str,
		expected_answer: str | None = None,
		system_message: str | None = None
	) -> Input:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return Input(
			input_text=input_text,
			expected_answer=expected_answer,
			system_message=system_message
		)

	def set_can_think(self, value: bool) -> None:
		self.can_think = value

	def generate(
		self,
		input: Input | str,
		params: GenerationParams | None = None,
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

		answer = self.tokenizer.decode(outputs[0])

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		start = answer.rfind("<unused95>")
		if start == -1:
			start = answer.rfind("<start_of_turn>model")
			start = start + len("<start_of_turn>model")
		else:
			start = start + len("<unused95>")

		end = answer.find("<end_of_turn>", start)
		if end == -1:
			end = len(answer)

		return answer[start:end].strip().replace("<eos>", "")
	
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

		buffer = ""
		is_thinking = None
		
		for new_text in streamer:
			buffer += new_text

			if is_thinking is None:
				if len(buffer.split()) > 5:
					is_thinking = False
					continue

				lower_buffer = buffer.lower()
				if lower_buffer.find("thought") != -1 or lower_buffer.find("<unused94>") != -1:
					is_thinking = True
					continue
			elif not is_thinking:
				yield buffer
				buffer = "" 
			else:
				if buffer.find("<unused95>") != -1:
					is_thinking = False
					buffer = buffer.split("<unused95>", 1)[1]
		
		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")
import textwrap
import threading
from functools import partial
from time import time
from typing import Iterator, Literal, TypedDict, cast

import torch
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.base.base import BaseModel
from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.generation_utils import create_generation_params


class MedGemmaInput(TypedDict):
	input_text: str
	expected_answer: str | None
	system_message: str | None

class MedGemma(BaseModel):
	model: Gemma3ForCausalLM | None = None
	can_think = False
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: Literal["4bit"] | None = None,
		seed: int | None = None,
		log_level: Literal["INFO", "DEBUG", "WARNING"] = "INFO",
	) -> None:
		return super().__init__(
			checkpoint=checkpoint,
			quantization=quantization,
			seed=seed,
			log_level=log_level
		)

	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		if not self.tokenizer:
			self._log("Could not set stop tokens - generation may not work...", "WARNING")
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
		data: MedGemmaInput
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_message = data.get("system_message", "")
		if not system_message:
			system_message = ""
		if self.can_think:
			system_message += f"think silently if needed. {system_message}"

		if system_message:
			system_message = f"{system_message}\n"

		expected_answer = data.get("expected_answer")
		answer = f"{expected_answer}<end_of_turn>" if expected_answer else ""
	
		return (
			f"<start_of_turn>user"
			f"{system_message}\n{data["input_text"]}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: str,
		expected_answer: str | None = None,
		system_message: str | None = None
	) -> MedGemmaInput:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return {
			"input_text": input_text,
			"expected_answer": expected_answer,
			"system_message": system_message
		}

	def set_can_think(self, value: bool) -> None:
		self.can_think = value

	def generate(
		self,
		input: MedGemmaInput | str,
		params: GenerationParams | None = None,
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			return None

		self._log(f"Processing received input...'")

		if params is None:
			params = GenerationParams(max_new_tokens=32768)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 32768

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params

		model_input = None
		if isinstance(input, str):
			model_input = self.build_input(
				input_text=input
			)
			model_input = self._build_input(
				data=model_input
			)
		else:
			model_input = self._build_input(
				data=input
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
		input: MedGemmaInput | str,
		params: GenerationParams | None = None
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			if False:
				yield ""
			return

		if params is None:
			params = GenerationParams(max_new_tokens=32768)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 32768

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params

		model_input = None
		if isinstance(input, str):
			model_input = self.build_input(
				input_text=input
			)
			model_input = self._build_input(
				data=model_input
			)
		else:
			model_input = self._build_input(
				data=input
			)
		
		tokenized_input = self._tokenize(model_input)
		input_ids, attention_mask = tokenized_input

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
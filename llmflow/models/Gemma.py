import textwrap
from time import time
from typing import Literal, TypedDict

import torch
from transformers import StoppingCriteriaList
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflow.base.base import BaseModel
from llmflow.callbacks.stop_on_token import StopOnToken
from llmflow.schemas.params import GenerationParams
from llmflow.utils.exceptions import MissingEssentialProp
from llmflow.utils.generation_utils import create_generation_params


class GemmaInput(TypedDict):
	input_text: str
	expected_answer: str | None
	system_message: str | None

class Gemma(BaseModel):
	model: Gemma3ForCausalLM | None = None
	can_think = False
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]

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
		quantization: Literal["8bit", "4bit"] | bool | None = None
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

		self.model = Gemma3ForCausalLM.from_pretrained(
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

		if not system_message:
			system_message = ""
		if self.can_think:
			system_message += f"think silently if needed. {system_message}"

		if system_message:
			system_message = f"SYSTEM INSTRUCTION: {system_message}."

		answer = f"{expected_answer}{self.tokenizer.eos_token}<eos><end_of_turn>" if expected_answer else ""
	
		return textwrap.dedent(
			f"<bos><start_of_turn>user\n"
			f"{system_message}\n{input_text}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: str,
		expected_answer: str | None = None,
		system_message: str | None = None
	) -> GemmaInput:
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
		input: GemmaInput | str,
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
			model_input = self._build_input(
				input_text=input
			)
		else:
			model_input = self._build_input(
				input_text=input["input_text"],
				system_message=input["system_message"]
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
				streamer=params.streamer,
				stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids)])
			)

		answer = self.tokenizer.decode(outputs[0])

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		thought = start = answer.find("<unused95>")
		if self.can_think or thought != -1:
			start = answer.find("thought")
			start = answer.find("<unused95>", start)

			if start == -1:
				start = answer.find("<start_of_turn>model")
				if start == -1:
					return ""
			start = start + len("<unused95>")
		else:
			start = answer.find("<start_of_turn>model")

			if start == -1:
				return ""
			start = start + len("<start_of_turn>model")

		end = answer.find("<end_of_turn>", start)
		if end == -1:
			end = len(answer)

		return answer[start:end].strip().replace("<eos>", "")
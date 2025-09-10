import textwrap
from time import time
from typing import Literal, TypedDict

import torch
from peft import PeftModelForCausalLM
from transformers import StoppingCriteriaList
from transformers.models.llama import LlamaForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflow.base.base import BaseModel
from llmflow.callbacks.stop_on_token import StopOnToken
from llmflow.schemas.params import GenerationParams
from llmflow.utils.exceptions import MissingEssentialProp
from llmflow.utils.generation_utils import create_generation_params


class LLaMA3Input(TypedDict):
	input_text: str
	expected_answer: str | None
	system_message: str | None

class LLaMA3(BaseModel):
	model: LlamaForCausalLM | None = None

	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		if not self.tokenizer:
			self._log("Could not set stop tokens - generation may not work...", "WARNING")
			return None
		particular_tokens = self.tokenizer.encode("<|eot_id|>")
		self.stop_token_ids = tokens + particular_tokens

	def load_checkpoint(
		self,
		checkpoint: str,
		adapter_path: str | None = None,
		quantization: Literal["8bit", "4bit"] | bool | None = None
	) -> None:
		self._log(f"Loading model on '{checkpoint}'")
		self._load_tokenizer(checkpoint)

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

		self._log("Model & Tokenizer loaded")

		if adapter_path:
			self._log(f"Loading adapter on '{adapter_path}'")
			self.adapter = PeftModelForCausalLM.from_pretrained(self.model, adapter_path)
			self._log(f"Adapter loaded")
		else:
			if self.adapter is not None:
				self._log("LoRA adapter found. Using the adpter with a different base model could lead to errors.", "WARNING")

		if not self._model_id:
			self._create_model_id()

		stop_tokens = []
		pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
		if pad_token_id:
			stop_tokens.append(pad_token_id)
		eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
		if eos_token_id:
			stop_tokens.append(eos_token_id)

		self._set_generation_stopping_tokens(stop_tokens)
		self.stop_token_ids = list(set(self.stop_token_ids))

	def _build_input(
		self,
		input_text: str,
		expected_answer: str | None = None,
		system_message: str | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		answer = f"{expected_answer}{self.tokenizer.eos_token}" if expected_answer else ""

		return textwrap.dedent(f"""<|start_header_id|>system<|end_header_id|>{system_message or ""}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{input_text}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}""")

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

	def _build_input_for_fine_tune(
		self,
		input: LLaMA3Input
	) -> dict[Literal["partial", "complete"], str]:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		partial = self._build_input(
			input_text=input["input_text"],
			system_message=input["system_message"]
		)
		complete = self._build_input(
			input_text=input["input_text"],
			expected_answer=input["expected_answer"],
			system_message=input["system_message"]
		)

		return {
			"partial": partial,
			"complete": complete
		}

	def _promptfy_dataset_for_dapt(
		self,
		dataset: list[LLaMA3Input]
	) -> list[str]:
		output = []
		for data in dataset:
			complete_input = self._build_input(
				input_text=data["input_text"],
				expected_answer=data.get("expected_answer", None),
				system_message=data.get("system_message", None)
			)
			output.append(complete_input)

		return output

	def generate(
		self,
		input: LLaMA3Input | str,
		params: GenerationParams | None = None,
		target: Literal["model", "adapter"] = "model",
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			return None

		if target == "model":
			model = self.model
		else:
			if self.adapter is None:
				self._log("Adapter missing. Defaulting to model.", "WARNING")
				model = self.model
			else:
				model = self.adapter

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

		model.eval()
		model.gradient_checkpointing_disable()

		start = time()

		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				use_cache=True,
				eos_token_id=None,
				stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids, self.tokenizer, self.log_level == "DEBUG")])
			)

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		response = outputs[0][input_ids.shape[1]:]

		return self.tokenizer.decode(response, skip_special_tokens=True)
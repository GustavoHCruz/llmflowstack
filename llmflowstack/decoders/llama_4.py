import threading
from functools import partial
from time import time
from typing import Iterator, cast

import torch
from torchao.quantization import Int4WeightOnlyConfig
from transformers import AutoTokenizer, TextIteratorStreamer, TorchAoConfig
from transformers.models.llama4 import Llama4ForCausalLM

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Llama4(BaseDecoder):
	model: Llama4ForCausalLM | None = None
	max_context_len = 32768
	legacy_trainer = True

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
		particular_tokens = self.tokenizer.encode("<|eot|>")
		self.stop_token_ids = tokens + particular_tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: bool | None = None
	) -> None:
		quantization_config = None
		if quantization:
			quant_config = Int4WeightOnlyConfig(group_size=128)
			quantization_config = TorchAoConfig(quant_type=quant_config)

		self.model = Llama4ForCausalLM.from_pretrained(
			checkpoint,
			dtype="auto",
			device_map="auto",
			quantization_config=quantization_config
		)
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: bool | None = None
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

		if system_message is not None:
			system_message = f"<|header_start|>system<|header_end|>\n\n{system_message}<|eot|>"

		answer = "<|header_start|>assistant<|header_end|>\n\n"
		answer += f"{output_text}<|eot|>" if output_text else ""

		return (
			"<|begin_of_text|>"
			f"{system_message}"
			"<|header_start|>user<|header_end|>\n\n"
			f"{input_text}<|eot|>"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: str,
		output_text: str | None = None,
		system_message: str | None = None
	) -> ModelInput:
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=True,
			system_message=system_message
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
				eos_token_id=self.stop_token_ids,
				pad_token_id=self.tokenizer.pad_token_id
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
from typing import Iterator

from torchao.quantization import Int4WeightOnlyConfig
from transformers import TorchAoConfig
from transformers.models.llama4 import Llama4ForConditionalGeneration

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Llama4(BaseDecoder):
	model: Llama4ForConditionalGeneration | None = None
	max_context_len = 32768
	can_handle_image_processing = True

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
		quantization: bool | None = None,
		max_memory: dict | None = None
	) -> None:
		quantization_config = None
		if quantization:
			quant_config = Int4WeightOnlyConfig(group_size=128)
			quantization_config = TorchAoConfig(quant_type=quant_config)

		self.model = Llama4ForConditionalGeneration.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			attn_implementation="sdpa",
			dtype="auto",
			device_map="auto",
			max_memory=max_memory
		)

	def _build_prompt(
		self,
		input_text: str,
		output_text: str | None = None,
		system_message: str | None = None,
		image_paths: list[str] | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		if system_message is not None:
			system_message = f"<|header_start|>system<|header_end|>\n\n{system_message}<|eot|>"

		answer = "<|header_start|>assistant<|header_end|>\n\n"
		answer += f"{output_text}<|eot|>" if output_text else ""

		image_text = len(image_paths or []) * "<|image_start|><|image|><|image_end|>"

		return (
			"<|begin_of_text|>"
			f"{system_message}"
			"<|header_start|>user<|header_end|>\n\n"
			f"{image_text}"
			f"{input_text}<|eot|>"
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
		params: GenerationParams | None = None,
		force_json: bool = False
	) -> str | None:
		if self.tokenizer is None:
			self._log("Tokenizer missing", LogLevel.WARNING)
			return None
		
		generation_outputs = self._generate(
			data=data,
			params=params,
			force_json=force_json,
			follow_prompt_format=True
		)

		if generation_outputs is None:
			return None
		
		start_index, outputs = generation_outputs

		answer = outputs[0][start_index:]

		decoded = self.tokenizer.decode(answer, skip_special_tokens=True)
		
		if isinstance(decoded, list):
			decoded = decoded[0]

		return decoded.strip()
	
	def generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False,
		follow_prompt_format: bool = True
	) -> Iterator[str]:
		return self._generate_stream(
			data=data,
			params=params,
			force_json=force_json,
			follow_prompt_format=follow_prompt_format
		)
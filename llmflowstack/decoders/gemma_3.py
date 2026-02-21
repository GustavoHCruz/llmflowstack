from typing import Iterator

from torchao.quantization import Int4WeightOnlyConfig
from transformers import TorchAoConfig
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Gemma3(BaseDecoder):
	model: Gemma3ForConditionalGeneration | None = None
	max_context_len = 32768
	can_handle_image_processing = True

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
		quantization: bool | None = None,
		max_memory: dict | None = None
	) -> None:
		quantization_config = None
		if quantization:
			quant_config = Int4WeightOnlyConfig(group_size=128)
			quantization_config = TorchAoConfig(quant_type=quant_config)

		self.model = Gemma3ForConditionalGeneration.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			attn_implementation="sdpa",
			dtype="auto",
			device_map="auto",
			max_memory=max_memory
		)

	def _build_prompt(
		self,
		input_text: list[str] | str,
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

		input_final_text = ""
		if image_paths is not None and isinstance(input_text, list) and len(image_paths) == len(input_text):
			for text in input_text:
				input_final_text += f"<start_of_image>{text}"
		elif image_paths is not None and isinstance(input_text, str):
			for _ in image_paths:
				input_final_text += f"<start_of_image>{input_text}"
		else:
			input_final_text = str(input_text)

		expected_answer = output_text
		answer = f"{expected_answer}<end_of_turn>" if expected_answer else ""
	
		return (
			f"<bos><start_of_turn>user\n"
			f"{system_message}\n{input_final_text}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: list[str] | str,
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
		force_json: bool = False
	) -> Iterator[str]:
		return self._generate_stream(
			data=data,
			params=params,
			force_json=force_json,
			follow_prompt_format=True
		)
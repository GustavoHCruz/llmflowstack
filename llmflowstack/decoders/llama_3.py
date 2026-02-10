from typing import Iterator

from torchao.quantization import Int4WeightOnlyConfig
from transformers import TorchAoConfig
from transformers.models.llama import LlamaForCausalLM

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Llama3(BaseDecoder):
	model: LlamaForCausalLM | None = None
	max_context_len = 8192

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
		particular_tokens = self.tokenizer.encode("<|eot_id|>")
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

		self.model = LlamaForCausalLM.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			attn_implementation="sdpa",
			dtype="auto",
			device_map="auto"
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
		force_json: bool = False,
		follow_prompt_format: bool = True
	) -> str | None:
		if self.tokenizer is None:
			self._log("Tokenizer missing", LogLevel.WARNING)
			return None
		
		generation_outputs = self._generate(
			data=data,
			params=params,
			force_json=force_json,
			follow_prompt_format=follow_prompt_format
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
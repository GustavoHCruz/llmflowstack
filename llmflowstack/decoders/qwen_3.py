from pathlib import Path
from typing import Iterator

from torchao.quantization import Float8WeightOnlyConfig
from transformers import AutoConfig, TorchAoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import \
    Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import \
    Qwen3_5MoeForConditionalGeneration

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Qwen3(BaseDecoder):
	model: Qwen3_5ForConditionalGeneration | Qwen3_5MoeForConditionalGeneration | None = None
	max_context_len = 32768
	can_handle_image_processing = True

	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		if not self.tokenizer:
			self._log("Could not set stop tokens - generation may not work...", LogLevel.WARNING)
			return None
		particular_tokens = self.tokenizer.encode("<|im_end|>")
		self.stop_token_ids = particular_tokens + tokens

	def _load_model(
		self,
		checkpoint: str | Path,
		quantization: bool | None = False,
		max_memory: dict | None = None
	) -> None:
		quantization_config = None
		if quantization:
			quant_config = Float8WeightOnlyConfig()
			quantization_config = TorchAoConfig(quant_type=quant_config)

		config = AutoConfig.from_pretrained(
			checkpoint
		)

		if "model_type" in config and "moe" in config.model_type:
			self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
				checkpoint,
				quantization_config=quantization_config,
				dtype="auto",
				device_map="auto",
				max_memory=max_memory
			)
		else:
			self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
				checkpoint,
				quantization_config=quantization_config,
				dtype="auto",
				device_map="auto",
				max_memory=max_memory
			)

	def _build_prompt(
		self,
		input_text: str,
		output_text: str | None = None,
		system_text: str | None = None,
		reasoning_text: str | None = None,
		enable_reasoning: bool = True,
		image_paths: list[str] | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_content = ""
		if system_text:
			system_content = f"<|im_start|>system\n{system_text}<|im_end|>\n"

		user_content = input_text
		if image_paths is not None:
			for _ in image_paths:
				user_content += "<|vision_start|><|image_pad|><|vision_end|>"

		assistant_content = "<|im_start|>assistant\n"

		# reasoning block (sempre presente para manter template consistente)
		if enable_reasoning:
			reasoning = reasoning_text or ""
			assistant_content += f"<think>\n{reasoning}\n</think>\n\n"
		else:
			assistant_content += "<think>\n</think>\n\n"

		if output_text:
			assistant_content += f"{output_text}<|im_end|>"

		return (
			f"{system_content}"
			f"<|im_start|>user\n{user_content}<|im_end|>\n"
			f"{assistant_content}"
		)
		
	def build_input(
		self,
		input_text: str,
		output_text: str | None = None,
		system_text: str | None = None,
		reasoning_text: str | None = None,
		enable_reasoning: bool = True,
		image_paths: list[str] | None = None
	) -> ModelInput:		
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=True,
			system_text=system_text,
			reasoning_text=reasoning_text,
			enable_reasoning=enable_reasoning,
			image_paths=image_paths
		)

	def generate(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False,
		follow_prompt_format: bool = True
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
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

		decoded = self.tokenizer.decode(outputs[0])

		if isinstance(decoded, list):
			decoded = decoded[0]

		start = decoded.find("</think>", start_index)
		if start == -1:
			start = start_index
		else:
			start += len("</think>")

		end = decoded.find("<|im_end|>", start)

		return decoded[start:end].strip()
	
	def generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False,
		follow_prompt_format: bool = True
	) -> Iterator[str]:
		streamer = self._generate_stream(
			data=data,
			params=params,
			force_json=force_json,
			follow_prompt_format=follow_prompt_format
		)

		thinking = False
		buffer = ""

		for new_text in streamer:
			buffer += new_text
			
			if not thinking and "<think>" in buffer:
				thinking = True
				buffer = buffer.split("<think>", 1)[1]

			if thinking:
				if "</think>" in buffer:
					buffer = buffer.split("</think>", 1)[1]
					buffer = buffer.strip()
					thinking = False
				else:
					continue

			buffer = buffer.replace("<|im_end|>", "")
			yield buffer
			buffer = ""
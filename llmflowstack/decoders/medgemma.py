from pathlib import Path
from typing import Iterator

from torchao.quantization import Int4WeightOnlyConfig
from transformers import Gemma3ForConditionalGeneration, TorchAoConfig

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class MedGemma(BaseDecoder):
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
		checkpoint: str | Path,
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
		system_text: str | None = None,
		image_paths: list[str] | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_content = f"Think silently if needed. {system_text or ""}\n"

		user_content = ""
		if image_paths is not None and isinstance(input_text, list) and len(image_paths) == len(input_text):
			for text in input_text:
				user_content += f"<start_of_image>{text}"
		elif image_paths is not None and isinstance(input_text, str):
			for _ in image_paths:
				user_content += f"<start_of_image>{input_text}"
		else:
			user_content = str(input_text)

		assistant_content = f"{output_text}<end_of_turn>" if output_text else ""
	
		return (
			f"<bos><start_of_turn>user\n"
			f"{system_content}\n{user_content}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{assistant_content}"
		)

	def build_input(
		self,
		input_text: list[str] | str,
		output_text: str | None = None,
		system_text: str | None = None,
		image_paths: list[str] | None = None
	) -> ModelInput:
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=True,
			system_text=system_text,
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

		answer = self.tokenizer.decode(outputs[0])

		if isinstance(answer, list):
			answer = answer[0]

		start = answer.rfind("<unused95>", start_index)
		if start == -1:
			start = answer.rfind("<start_of_turn>model")
			start = start + len("<start_of_turn>model")
		else:
			start = start + len("<unused95>")

		end = answer.find("<end_of_turn>", start)
		if end == -1:
			end = len(answer)

		return answer[start:end].strip()
	
	def generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False
	) -> Iterator[str]:
		streamer = self._generate_stream(
			data=data,
			params=params,
			force_json=force_json,
			follow_prompt_format=True
		)

		done_thinking = False
		buffer = ""
		
		for new_text in streamer:
			buffer += new_text

			if done_thinking is None:
				if len(buffer.split()) > 5:
					done_thinking = False
					continue

				lower_buffer = buffer.lower()
				if lower_buffer.find("thought") != -1 or lower_buffer.find("<unused94>") != -1:
					done_thinking = True
					continue
			elif not done_thinking:
				yield buffer
				buffer = "" 
			else:
				if buffer.find("<unused95>") != -1:
					done_thinking = False
					buffer = buffer.split("<unused95>", 1)[1]
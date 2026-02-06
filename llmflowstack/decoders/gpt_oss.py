import threading
from functools import partial
from time import time
from typing import Iterator, Literal, cast

import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers.models.gpt_oss import GptOssForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

from llmflowstack.decoders.base_decoder import BaseDecoder, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class GptOss(BaseDecoder):
	model: GptOssForCausalLM | None = None
	reasoning_level: Literal["Low", "Medium", "High", "Off"] = "Low"
	max_context_len = 32768

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
		particular_tokens = [200012, 200002]
		self.stop_token_ids = particular_tokens + tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: bool | None = False
	) -> None:
		if quantization:
			quantization_config = Mxfp4Config(dequantize=False)
		else:
			quantization_config = Mxfp4Config(dequantize=True)

		try:
			self.model = GptOssForCausalLM.from_pretrained(
				checkpoint,
				quantization_config=quantization_config,
				dtype="auto",
				device_map="auto"
			)
		except Exception as _:
			self._log("Error trying to load the model. Defaulting to load without quantization...", LogLevel.WARNING)
			self.model = GptOssForCausalLM.from_pretrained(
				checkpoint,
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
		system_message: str | None = None,
		developer_message: str | None = None,
		reasoning_message: str | None = None
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: {self.reasoning_level}\n\n{system_message or ''}# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
		if self.reasoning_level == "Off":
			system_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\n{system_message}# Valid channels: final. Channel must be included for every message.<|end|>"

		developer_text = ""
		if developer_message:
			developer_text = f"<|start|>developer<|message|># Instructions\n\n{developer_message or ''}<|end|>"

		assistant_text = ""
		if reasoning_message:
			assistant_text += f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_message or ''}<|end|>"

		if output_text:
			assistant_text += f"<|start|>assistant<|channel|>final<|message|>{output_text or ''}<|return|>"

		if not output_text and self.reasoning_level == "Off":
			assistant_text = "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"

		return (
			f"{system_text}{developer_text}"
			f"<|start|>user<|message|>{input_text}<|end|>"
			f"{assistant_text}"
		)
		
	def build_input(
		self,
		input_text: str,
		output_text: str | None = None,
		system_message: str | None = None,
		developer_message: str | None = None,
		reasoning_message: str | None = None
	) -> ModelInput:		
		return self._tokenize(
			input_text=input_text,
			output_text=output_text,
			follow_prompt_format=True,
			system_message=system_message,
			developer_message=developer_message,
			reasoning_message=reasoning_message
		)

	def set_reasoning_level(
		self,
		level: Literal["Low", "Medium", "High", "Off"]
	) -> None:
		self.reasoning_level = level

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

		answer = self.tokenizer.decode(outputs[0])

		if isinstance(answer, list):
			answer = answer[0]

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		start = answer.rfind("<|message|>")
		if start == -1:
			return ""

		start += len("<|message|>")

		end = answer.find("<|return|>", start)
		if end == -1:
			end = len(answer)

		return answer[start:end].strip()
	
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

		done_thinking = self.reasoning_level == "Off"
		buffer = ""

		for new_text in streamer:
			buffer += new_text

			if "final" in buffer and not done_thinking:
				done_thinking = True
				buffer = buffer.split("final", 1)[1]
			
			if done_thinking:
				yield buffer
				buffer = ""
		
		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")
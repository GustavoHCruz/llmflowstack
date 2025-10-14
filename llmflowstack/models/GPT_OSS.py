import textwrap
import threading
from functools import partial
from time import time
from typing import Iterator, Literal, TypedDict, cast

import torch
from openai_harmony import HarmonyEncodingName, load_harmony_encoding
from transformers import (AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.gpt_oss import GptOssForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

from llmflowstack.base.base import BaseModel
from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.generation_utils import create_generation_params


class GPTOSSInput(TypedDict):
	input_text: str
	system_message: str | None
	developer_message: str | None
	expected_answer: str | None
	reasoning_message: str | None
	reasoning_level: Literal["Low", "Medium", "High"] | None

class GPT_OSS(BaseModel):
	model: GptOssForCausalLM | None = None
	reasoning_level: Literal["Low", "Medium", "High"] = "Low"
	question_fields = ["input_text", "developer_message", "system_message"]
	answer_fields = ["expected_answer", "reasoning_message"]

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: bool | None = None,
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
		encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
		particular_tokens = encoding.stop_tokens_for_assistant_actions()
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
				device_map="auto",
				attn_implementation="eager",
			)
		except Exception as _:
			self._log("Error trying to load the model. Defaulting to load without quantization...", "WARNING")
			self.model = GptOssForCausalLM.from_pretrained(
				checkpoint,
				dtype="auto",
				device_map="auto",
				attn_implementation="eager"
			)
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: bool | None = None
	) -> None:
		return super().load_checkpoint(checkpoint, quantization)

	def _build_input(
		self,
		data: GPTOSSInput
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		reasoning = data.get("reasoning_level")
		if reasoning is None:
			reasoning = self.reasoning_level

		system_message = data.get("system_message", "")
		system_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: {reasoning}\n\n{system_message}# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"

		developer_text = ""
		developer_message = data.get("developer_message", "")
		if developer_message:
			developer_text = f"<|start|>developer<|message|># Instructions\n\n{developer_message}<|end|>"

		assistant_text = ""
		reasoning_message = data.get("reasoning_message", "")
		if reasoning_message:
			assistant_text += f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_message}<|end|>"

		expected_answer = data.get("expected_answer", "")
		if expected_answer:
			assistant_text += f"<|start|>assistant<|channel|>final<|message|>{expected_answer}<|return|>"

		return textwrap.dedent(f"""{system_text}{developer_text}<|start|>user<|message|>{data["input_text"]}<|end|>{assistant_text}""")

	def build_input(
		self,
		input_text: str,
		system_message: str | None = None,
		developer_message: str | None = None,
		expected_answer: str | None = None,
		reasoning_message: str | None = None,
		reasoning_level: Literal["Low", "Medium", "High"] | None = None
	) -> GPTOSSInput:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return {
			"input_text": input_text,
			"developer_message": developer_message,
			"system_message": system_message,
			"reasoning_level": reasoning_level,
			"expected_answer": expected_answer,
			"reasoning_message": reasoning_message
		}

	def set_reasoning_level(
		self,
		level: Literal["Low", "Medium", "High"]
	) -> None:
		self.reasoning_level = level

	def generate(
		self,
		input: GPTOSSInput | str,
		params: GenerationParams | None = None
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
		input: GPTOSSInput | str,
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

		done_thinking = False
		buffer = ""

		for new_text in streamer:
			buffer += new_text

			if "final" in buffer:
				done_thinking = True
				buffer = buffer.split("final", 1)[1]
			
			if done_thinking:
				yield buffer
				buffer = ""
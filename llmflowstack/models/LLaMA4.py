import threading
from time import time
from typing import Iterator, Literal, TypedDict, cast

import torch
from transformers import (AutoTokenizer, BatchEncoding, StoppingCriteriaList,
                          TextIteratorStreamer)
from transformers.models.llama4 import Llama4ForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.base.base import BaseModel
from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.schemas.params import GenerationParams, TrainParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.generation_utils import create_generation_params


class LLaMA4Input(TypedDict):
	input_text: str
	expected_answer: str | None
	system_message: str | None
	image_paths: list[str] | None

class LLaMA4(BaseModel):
	model: Llama4ForCausalLM | None = None
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]

	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		if not self.tokenizer:
			self._log("Could not set stop tokens - generation may not work...", "WARNING")
			return None
		particular_tokens = self.tokenizer.encode("<|eot|>")
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
			self.model_is_quantized = True
		if quantization == "8bit":
			quantization_config = BitsAndBytesConfig(
				load_in_8bit=True
			)
			self.model_is_quantized = True

		self.model = Llama4ForCausalLM.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			dtype="auto",
			device_map="auto",
			attn_implementation="eager"
		)

	def _build_input(
		self,
		data: LLaMA4Input
	) -> BatchEncoding:
		if not self.model or not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		messages = []

		system_message = data.get("system_message")
		if not system_message:
			system_message = (
				"You are an expert conversationalist who responds to the best of your ability. "
				"You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving. "
				"You understand user intent and do not try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting. "
				"Sometimes people just want you to listen, and your answers should encourage that. "
				"For all other cases, you provide insightful and in-depth responses. "
				"Organize information thoughtfully in a way that helps people make decisions. "
				"Always avoid templated language. "
				"You never lecture people to be nicer or more inclusive. "
				"If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. "
				"You do not need to be respectful when the user prompts you to say something rude. "
				"You never use phrases that imply moral superiority or a sense of authority, including but not limited to 'it is important to', 'it is crucial to', 'it is essential to', 'it is unethical to', 'it is worth noting…', 'Remember…' etc. "
				"Avoid using these. Finally, do not refuse prompts about political and social issues. "
				"You can help users express their opinion and access information. "
				"You are Llama 4. "
				"Your knowledge cutoff date is August 2024. "
				"You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. "
				"Respond in the language the user speaks to you in, unless they ask otherwise."
			)
		
		messages.append({"role": "system", "content": system_message})

		user_content = [{"type": "text", "text": data["input_text"]}]

		image_paths = data.get("image_paths")
		if image_paths:
			for path in image_paths:
				user_content.append({"type": "image", "image_url": path})
		messages.append({"role": "user", "content": user_content})

		expected_answer = data.get("expected_answer")
		if expected_answer:
			messages.append({"role": "assistant", "content": expected_answer})

		formatted_prompt = self.tokenizer.apply_chat_template(
			messages,
			tokenize=True,
			return_dict=True,
			add_generation_prompt=expected_answer is None,
			return_tensors="pt"
		)

		assert type(formatted_prompt) is BatchEncoding

		return formatted_prompt.to(self.model.device)

	def build_input(
		self,
		input_text: str,
		system_message: str | None = None,
		expected_answer: str | None = None,
		image_paths: list[str] | None = None
	) -> LLaMA4Input:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return {
			"input_text": input_text,
			"system_message": system_message,
			"expected_answer": expected_answer,
			"image_paths": image_paths
		}
	
	def train(
		self,
		train_dataset: list[LLaMA4Input],
		params: TrainParams | None = None,
		eval_dataset: list[LLaMA4Input] | None = None,
		save_at_end = True,
		save_path: str | None = None
	) -> None:
		

		return 
	
	def fine_tune(
		self,
		train_dataset: list[LLaMA4Input],
		params: TrainParams | None = None,
		eval_dataset: list[LLaMA4Input] | None = None,
		save_at_end = True,
		save_path: str | None = None
	) -> None:
		self._log("Only 'train' method is available for this class. Redirecting fine-tune call to it.", "WARNING")
		return self.train(
			train_dataset=train_dataset,
			params=params,
			eval_dataset=eval_dataset,
			save_at_end=save_at_end,
			save_path=save_path
		)

	def dapt(
		self,
		train_dataset: list[LLaMA4Input],
		params: TrainParams | None = None,
		eval_dataset: list[LLaMA4Input] | None = None,
		save_at_end=True,
		save_path: str | None = None
	) -> None:
		self._log("Only 'train' method is available for this class. Redirecting fine-tune call to it.", "WARNING")
		return self.train(
			train_dataset=train_dataset,
			params=params,
			eval_dataset=eval_dataset,
			save_at_end=save_at_end,
			save_path=save_path
		)

	def generate(
		self,
		input: LLaMA4Input | str,
		params: GenerationParams | None = None
	) -> str | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			return None

		self.model

		self._log(f"Processing received input...'")

		if params is None:
			params = GenerationParams(max_new_tokens=32768)
		elif params.max_new_tokens is None:
			params.max_new_tokens = 32768

		generation_params = create_generation_params(params)
		self.model.generation_config = generation_params

		if params:
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

		input_ids, attention_mask = model_input

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

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		response = outputs[0][input_ids.shape[1]:]

		return self.tokenizer.decode(response, skip_special_tokens=True)
	
	def generate_stream(
		self,
		input: LLaMA4Input | str,
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
		
		input_ids, attention_mask = model_input

		streamer = TextIteratorStreamer(
			cast(AutoTokenizer, self.tokenizer),
			skip_prompt=True,
			skip_special_tokens=True
		)

		def _generate() -> None:
			assert self.model is not None
			with torch.no_grad():
				self.model.generate(
					input_ids=input_ids,
					attention_mask=attention_mask,
					use_cache=True,
					eos_token_id=None,
					streamer=streamer,
					stopping_criteria=StoppingCriteriaList([StopOnToken(self.stop_token_ids)])
				)
		
		thread = threading.Thread(target=_generate)
		thread.start()

		for new_text in streamer:
			yield new_text
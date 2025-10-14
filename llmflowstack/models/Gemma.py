import threading
from functools import partial
from time import time
from typing import Iterator, Literal, TypedDict, cast

import torch
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          StoppingCriteriaList, TextIteratorStreamer, Trainer,
                          TrainingArguments)
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from llmflowstack.base.base import BaseModel
from llmflowstack.callbacks.log_collector import LogCollectorCallback
from llmflowstack.callbacks.stop_on_token import StopOnToken
from llmflowstack.schemas.params import GenerationParams, TrainParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.generation_utils import create_generation_params


class Gemma3Input(TypedDict):
	input_text: str
	expected_answer: str | None
	system_message: str | None
	image_paths: list[str] | None

class Gemma3(BaseModel):
	model: Gemma3ForCausalLM | None = None
	question_fields = ["input_text", "system_message"]
	answer_fields = ["expected_answer"]

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: Literal["4bit"] | None = None,
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
		particular_tokens = self.tokenizer.encode("<end_of_turn>")
		self.stop_token_ids = tokens + particular_tokens

	def _load_model(
		self,
		checkpoint: str,
		quantization: Literal["4bit"] | None = None
	) -> None:
		quantization_config = None
		if quantization == "4bit":
			quantization_config = BitsAndBytesConfig(
				load_in_4bit=True
			)

		self.model = Gemma3ForCausalLM.from_pretrained(
			checkpoint,
			quantization_config=quantization_config,
			dtype="auto",
			device_map="auto",
			attn_implementation="eager"
		)
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: Literal['4bit'] | None = None
	) -> None:
		return super().load_checkpoint(checkpoint, quantization)

	def _build_input(
		self,
		data: Gemma3Input
	) -> str:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		system_message = data.get("system_message", "")
		if not system_message:
			system_message = ""

		if system_message:
			system_message = f"{system_message}\n"

		expected_answer = data.get("expected_answer")
		answer = f"{expected_answer}<end_of_turn>" if expected_answer else ""
	
		return (
			f"<start_of_turn>user"
			f"{system_message}\n{data["input_text"]}<end_of_turn>\n"
			f"<start_of_turn>model\n"
			f"{answer}"
		)

	def build_input(
		self,
		input_text: str,
		system_message: str | None = None,
		expected_answer: str | None = None,
		image_paths: list[str] | None = None
	) -> Gemma3Input:
		if not self.tokenizer:
			raise MissingEssentialProp("Could not find tokenizer.")

		return {
			"input_text": input_text,
			"system_message": system_message,
			"expected_answer": expected_answer,
			"image_paths": image_paths
		}
	
	def dapt(
		self,
		train_dataset: list,
		params: TrainParams | None = None,
		eval_dataset: list | None = None,
		save_at_end = True,
		save_path: str | None = None
	) -> None:
		if not self.model:
			self._log("Could not find a model loaded. Try loading a model first.", "WARNING")
			return None
		if not self.tokenizer:
			self._log("Could not find a tokenizer loaded. Try loading a tokenizer first.", "WARNING")
			return None

		self._log("Starting Training")

		if self.model_is_quantized:
			self._log("Cannot traub a quantized model.", "WARNING")
			return None
		
		if params is None:
			params = TrainParams()

		training_arguments = TrainingArguments(
			num_train_epochs=params.epochs,
			learning_rate=params.lr,
			gradient_accumulation_steps=params.gradient_accumulation,
			warmup_ratio=params.warmup_ratio,
			lr_scheduler_type="cosine_with_min_lr",
			lr_scheduler_kwargs={"min_lr_rate": 0.1},
			output_dir=None,
			save_strategy="no",
			logging_steps=params.logging_steps
		)

		if self.seed is not None:
			training_arguments.seed = self.seed

		processed_train_dataset = self._promptfy_dataset_for_dapt(train_dataset)
		tokenized_train_dataset = self._tokenize_dataset_for_dapt(processed_train_dataset)

		tokenized_eval_dataset = None
		if eval_dataset:
			processed_eval_dataset = self._promptfy_dataset_for_dapt(eval_dataset)
			tokenized_eval_dataset = self._tokenize_dataset_for_dapt(processed_eval_dataset)

		log_callback = LogCollectorCallback()

		trainer = Trainer(
			model=self.model,
			train_dataset=tokenized_train_dataset,
			eval_dataset=tokenized_eval_dataset,
			args=training_arguments,
			callbacks=[log_callback],
			data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
		)

		trainer.train()

		if save_at_end and save_path:
			self.save_checkpoint(
				path=save_path
			)

		self._log("Finished Training")
	
	def fine_tune(
		self,
		train_dataset: list,
		params: TrainParams | None = None,
		eval_dataset: list | None = None,
		save_at_end = True,
		save_path: str | None = None
	) -> None:
		self._log("Only 'dapt' method is available for this class. Redirecting call to it.", "WARNING")
		return self.dapt(
			train_dataset=train_dataset,
			params=params,
			eval_dataset=eval_dataset,
			save_at_end=save_at_end,
			save_path=save_path
		)

	def generate(
		self,
		input: Gemma3Input | str,
		params: GenerationParams | None = None,
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

		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		response = outputs[0][input_ids.shape[1]:]

		return self.tokenizer.decode(response, skip_special_tokens=True)
	
	def generate_stream(
		self,
		input: Gemma3Input | str,
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

		for new_text in streamer:
			yield new_text
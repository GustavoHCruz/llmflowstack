import gc
import os
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from time import time
from typing import Any, Iterator, Literal, cast

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from torch import Tensor, tensor
from transformers import (AutoProcessor, AutoTokenizer, LogitsProcessorList,
                          PreTrainedTokenizerBase, StoppingCriteriaList,
                          TextIteratorStreamer, Trainer)
from transformers.tokenization_utils_base import BatchEncoding
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from llmflowstack.callbacks.force_json import (ForceJsonLogitsProcessor,
                                               StopOnJsonComplete)
from llmflowstack.callbacks.log_collector import LogCollectorCallback
from llmflowstack.schemas.params import GenerationParams, TrainParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


@dataclass
class ModelInput:
	input_ids: Tensor
	attention_mask: Tensor
	ft_labels: Tensor
	dapt_labels: Tensor
	token_type_ids: Tensor | None
	pixel_values: Tensor | None

class BaseDecoder(ABC):
	model = None
	tokenizer = None
	processor = None
	model_is_quantized = None
	seed = None
	stop_token_ids = []
	question_fields = []
	answer_fields = []
	max_context_len: int = 1024
	can_handle_image_processing = False

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: bool | None = None,
		seed: int | None = None
	) -> None:
		if seed:
			self._set_seed(seed)

		self._base_model = checkpoint

		self.logger = getLogger(f"LLMFlowStack.{self.__class__.__name__}")

		self.tokenizer: PreTrainedTokenizerBase | None = None

		if checkpoint:
			self._checkpoint = checkpoint
			self.load_checkpoint(
				checkpoint=checkpoint,
				quantization=quantization
			)

			if quantization:
				self.model_is_quantized = True
	
	def _log(
		self,
		message: str,
		level: LogLevel = LogLevel.INFO,
	) -> None:
		log_func = getattr(self.logger, level.lower(), None)
		if log_func:
			log_func(message)
		else:
			self.logger.info(message)
	
	@abstractmethod
	def _load_model(
		self,
		checkpoint: str,
		*args: Any,
		**kwargs: Any
	) -> None:
		pass

	def _load_tokenizer(
		self,
		checkpoint: str
	) -> None:
		tokenizer = AutoTokenizer.from_pretrained(checkpoint)
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.add_eos_token = True
		tokenizer.padding_side = "right"

		self.tokenizer = tokenizer
	
	def _load_processor(
		self,
		checkpoint: str
	) -> None:
		processor = AutoProcessor.from_pretrained(checkpoint)
		
		self.processor = processor
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: Any
	) -> None:
		if self.model:
			self._log("A model is already loaded. Attempting to reset it.", LogLevel.WARNING)
			self.unload_model()

		self._log(f"Loading model on '{checkpoint}'")

		self._load_tokenizer(
			checkpoint=checkpoint
		)
		self._load_model(
			checkpoint=checkpoint,
			quantization=quantization
		)
		if self.can_handle_image_processing:
			self._load_processor(
				checkpoint=checkpoint
			)

		self._log("Model & Tokenizer loaded")
		
		if quantization:
			self.model_is_quantized = True
		
		stop_tokens = []
		pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
		if pad_token_id:
			stop_tokens.append(pad_token_id)
		eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
		if eos_token_id:
			stop_tokens.append(eos_token_id)

		self._set_generation_stopping_tokens(stop_tokens)
		self.stop_token_ids = list(set(self.stop_token_ids))

	def _set_seed(
		self,
		seed: int
	) -> None:
		self.seed = seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True)

		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
		os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

	def save_checkpoint(
		self,
		path: str
	) -> None:
		if not self.model:
			self._log("No model to save.", LogLevel.WARNING)
			return None
		if not self.tokenizer:
			self._log("No tokenizer to save.", LogLevel.WARNING)
			return None

		os.makedirs(path, exist_ok=True)

		self._log("Saving model...")
		model_to_save = self.model

		model_to_save.save_pretrained(path)
		self.tokenizer.save_pretrained(path)

		self._log(f"Model and Tokenizer saved at {path}")

		self._log(f"Model custom information saved at {path}")

	@abstractmethod
	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		pass

	def _build_generic_input(
		self,
		input_text: str,
		output_text: str | None = None
	) -> str:
		if self.tokenizer is None:
			raise MissingEssentialProp("Tokenizer missing")
		eos = self.tokenizer.eos_token

		output_text = (output_text or "") + str(eos or "") if output_text else ""

		return input_text + output_text
	
	@abstractmethod
	def _build_prompt(
		self,
		input_text: str,
		output_text: str | None = None,
		*args: Any,
		**kwargs: Any
	) -> str:
		pass
		
	def _tokenize(
		self,
		input_text: str,
		output_text: str | None = None,
		follow_prompt_format: bool = True,
		image_paths: list[str] | None = None,
		*args: Any,
		**kwargs: Any
	) -> ModelInput:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing")

		if follow_prompt_format:
			promptfied_input = self._build_prompt(
				input_text=input_text,
				output_text=output_text,
				*args,
				**kwargs
			)
		else:
			promptfied_input = self._build_generic_input(
				input_text=input_text,
				output_text=output_text
			)
		
		if image_paths and self.processor is not None:
			images = []
			for path in image_paths:
				images.append(
					Image.open(path).convert("RGB")
				)
			
			tokenized_input = self.processor(
				text=promptfied_input,
				images=images
			)
		else:		
			tokenized_input: BatchEncoding = self.tokenizer(
				text=promptfied_input,
				add_special_tokens=False
			)

		input_ids = tensor(tokenized_input["input_ids"], dtype=torch.long)
		attention_mask = tensor(tokenized_input["attention_mask"], dtype=torch.bool)
		token_type_ids = None
		if tokenized_input.get("token_type_ids"):
			token_type_ids = tensor(tokenized_input["token_type_ids"], dtype=torch.long)
		pixel_values = None
		if tokenized_input.get("pixel_values"):
			pixel_values = tensor(tokenized_input["pixel_values"], dtype=torch.float)
	
		if follow_prompt_format:
			promptfied_partial_input = self._build_prompt(
				input_text=input_text,
				output_text=None,
				*args,
				**kwargs
			)
		else:
			promptfied_partial_input = self._build_generic_input(
				input_text=input_text,
				output_text=None
			)

		tokenized_partial_input: BatchEncoding = self.tokenizer(
			text=promptfied_partial_input,
			add_special_tokens=False
		)

		partial_input_ids = tensor(tokenized_partial_input["input_ids"], dtype=torch.long)

		start = int(partial_input_ids.shape[0])

		ft_labels = torch.full_like(input_ids, -100)
		ft_labels[start:] = input_ids[start:]
		ft_labels = ft_labels.masked_fill(attention_mask == 0, -100)

		dapt_labels = input_ids.clone().masked_fill(attention_mask == 0, -100)

		return ModelInput(
			input_ids=input_ids,
			attention_mask=attention_mask,
			ft_labels=ft_labels,
			dapt_labels=dapt_labels,
			token_type_ids=token_type_ids,
			pixel_values=pixel_values
		)
	
	def _build_dataset(
		self,
		dataset: list[ModelInput],
		mode: Literal["FT", "DAPT"]
	) -> Dataset:
		if mode == "DAPT":
			return Dataset.from_list([{
				"input_ids": data.input_ids,
				"attention_mask": data.attention_mask,
				"labels": data.dapt_labels
			} for data in dataset])
		
		return Dataset.from_list([{
			"input_ids": data.input_ids,
			"attention_mask": data.attention_mask,
			"labels": data.ft_labels
		} for data in dataset])
	
	def train(
		self,
		train_data: list[ModelInput],
		params: TrainParams | None = None,
		eval_data: list[ModelInput] | None = None,
		save_at_end = False,
		save_path: str | None = None,
		mode: Literal["FT", "DAPT"] = "FT"
	) -> None:
		if not self.model:
			self._log("Could not find a model loaded. Try loading a model first.", LogLevel.WARNING)
			return None
		if not self.tokenizer:
			self._log("Could not find a tokenizer loaded. Try loading a tokenizer first.", LogLevel.WARNING)
			return None
		
		if self.model_is_quantized:
			self._log("Cannot train a quantized model.", LogLevel.WARNING)
			return None

		self._log("Starting training...")

		if params is None:
			params = TrainParams()
		
		training_arguments = SFTConfig(
			num_train_epochs=params.epochs,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			per_device_eval_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			gradient_checkpointing=True,
			warmup_ratio=params.warmup_ratio,
			lr_scheduler_type="cosine_with_min_lr",
			lr_scheduler_kwargs = {"min_lr_rate": 0.1},
			label_smoothing_factor=params.label_smoothing_factor,
			output_dir=None,
			save_strategy="no",
			logging_steps=params.logging_steps
		)

		if self.seed is not None:
			training_arguments.seed = self.seed

		train_dataset = self._build_dataset(
			dataset=train_data,
			mode=mode
		)
		eval_dataset = None
		if eval_data:
			eval_dataset = self._build_dataset(
				dataset=eval_data,
				mode=mode
			)
		
		log_callback = LogCollectorCallback()

		trainer = SFTTrainer
		if self.can_handle_image_processing:
			trainer = Trainer
		
		trainer = trainer(
			model=self.model,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			args=training_arguments,
			callbacks=[log_callback]
		)

		trainer.train()

		if save_at_end and save_path:
			self.save_checkpoint(
				path=save_path
			)
		
		self._log("Finished training")

	def _prepare_generation(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		follow_prompt_format: bool = True
	) -> ModelInput | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None

		self._log(f"Processing received input...'")

		if isinstance(data, str):
			model_input = self._tokenize(
				input_text=data,
				follow_prompt_format=follow_prompt_format
			)
		else:
			model_input = data

		input_tokens_len = model_input.input_ids.shape[-1]

		if params is None:
			params = GenerationParams()
		
		requested_tokens = params.max_new_tokens
		available_tokens = self.max_context_len - input_tokens_len

		if available_tokens <= 0:
			raise ValueError(f"Could not generate new tokens, input of {input_tokens_len} tokens exceeds or euqals max model context window ({self.max_context_len})")

		if requested_tokens is None:
			max_new_tokens = max(0, available_tokens)
		else:
			max_new_tokens = min(
				requested_tokens,
				max(0, available_tokens)
			)
		
		params.max_new_tokens = max_new_tokens
		self.model.generation_config = params.to_generation_config()

		return model_input

	def _generate(
		self,
		data: ModelInput | str,
		params: GenerationParams | None,
		force_json: bool,
		follow_prompt_format: bool
	) -> None | tuple[int, Tensor]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None

		model_input = self._prepare_generation(
			data=data,
			params=params,
			follow_prompt_format=follow_prompt_format
		)

		if model_input is None:
			return None
		
		input_dict = {}
		input_dict["input_ids"] = model_input.input_ids.unsqueeze(0).to(self.model.device)
		input_dict["attention_mask"] = model_input.attention_mask.unsqueeze(0).to(self.model.device)
		if model_input.token_type_ids:
			input_dict["token_type_ids"] = model_input.token_type_ids.unsqueeze(0).to(self.model.device)
		if model_input.pixel_values:
			input_dict["pixel_values"] = model_input.pixel_values.unsqueeze(0).to(self.model.device)

		self.model.eval()
		self.model.gradient_checkpointing_disable()

		stopping = []

		logits_processor = None
		if force_json:
			forcer = ForceJsonLogitsProcessor(
				tokenizer=self.tokenizer,
				top_k=256
			)
			logits_processor = LogitsProcessorList([forcer])
			stopping.append(StopOnJsonComplete(forcer))

		start = time()
		with torch.no_grad():
			outputs = self.model.generate(
				**input_dict,
				use_cache=True,
				eos_token_id=self.stop_token_ids,
				pad_token_id=self.tokenizer.pad_token_id,
				logits_processor=logits_processor,
				stopping_criteria=StoppingCriteriaList(stopping)
			)
		
		end = time()
		total_time = end - start

		self._log(f"Response generated in {total_time:.4f} seconds")

		return input_dict["input_ids"].shape[1], outputs
	
	def _generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False,
		follow_prompt_format: bool = True
	) -> Iterator[str]:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", LogLevel.WARNING)
			return None
		
		model_input = self._prepare_generation(
			data=data,
			params=params,
			follow_prompt_format=follow_prompt_format
		)

		if model_input is None:
			return None
		
		input_dict = {}
		input_dict["input_ids"] = model_input.input_ids.unsqueeze(0).to(self.model.device)
		input_dict["attention_mask"] = model_input.attention_mask.unsqueeze(0).to(self.model.device)
		if model_input.token_type_ids:
			input_dict["token_type_ids"] = model_input.token_type_ids.unsqueeze(0).to(self.model.device)
		if model_input.pixel_values:
			input_dict["pixel_values"] = model_input.pixel_values.unsqueeze(0).to(self.model.device)

		streamer = TextIteratorStreamer(
			cast(AutoTokenizer, self.tokenizer),
			skip_prompt=True,
			skip_special_tokens=True
		)

		stopping = []

		logits_processor = None
		if force_json:
			forcer = ForceJsonLogitsProcessor(
				tokenizer=self.tokenizer,
				top_k=256
			)
			logits_processor = LogitsProcessorList([forcer])
			stopping.append(StopOnJsonComplete(forcer))

		generate_fn = partial(
			self.model.generate,
			**input_dict,
			use_cache=True,
			eos_token_id=self.stop_token_ids,
			pad_token_id=self.tokenizer.pad_token_id,
			logits_processor=logits_processor,
			stopping_criteria=StoppingCriteriaList(stopping),
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

	@abstractmethod
	def generate(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False,
		*args: Any,
		**kwargs: Any
	) -> str | None:
		pass

	@abstractmethod
	def generate_stream(
		self,
		data: ModelInput | str,
		params: GenerationParams | None = None,
		force_json: bool = False,
		*args: Any,
		**kwargs: Any
	) -> Iterator[str]:
		pass

	def unload_model(self) -> None:
		try:
			self._log("Trying to reset model...")
			del self.model
			gc.collect()
			torch.cuda.empty_cache()
			self.model = None
			self.model_is_quantized = None
			self._log("Reset successfully.")
		except Exception as e:
			self._log("Couldn't reset model...", LogLevel.ERROR)
			self._log(f"{str(e)}", LogLevel.DEBUG)

	def set_seed(self, seed: int) -> None:
		self._log(f"Setting seed value {seed}")
		self._set_seed(seed)
		self._log(f"Seed setted")

	def __del__(self) -> None:
		self.unload_model()
		del self.tokenizer
		if self.can_handle_image_processing:
			del self.processor
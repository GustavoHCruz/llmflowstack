import gc
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal, cast

import numpy as np
import torch
from datasets import Dataset
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from llmflowstack.callbacks.log_collector import LogCollectorCallback
from llmflowstack.schemas.params import TrainParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


@dataclass
class Input:
	input_text: str
	expected_text: str | None = None

class BaseDecoder(ABC):
	model = None
	tokenizer = None
	model_is_quantized = None
	seed = None
	stop_token_ids = []
	question_fields = []
	answer_fields = []

	def __init__(
		self,
		checkpoint: str | None = None,
		quantization: Literal["4bit", "8bit"] | bool | None = None,
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

	def _load_tokenizer(self, checkpoint: str) -> None:
		tokenizer = AutoTokenizer.from_pretrained(checkpoint)
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.add_eos_token = True
		tokenizer.padding_side = "right"

		self.tokenizer = tokenizer
	
	def load_checkpoint(
		self,
		checkpoint: str,
		quantization: Any
	) -> None:
		if self.model:
			self._log("A model is already loaded. Attempting to reset it.", LogLevel.WARNING)
			self.unload_model()

		self._log(f"Loading model on '{checkpoint}'")

		self._load_tokenizer(checkpoint)
		self._load_model(
			checkpoint=checkpoint,
			quantization=quantization
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

	def from_pretrained(
		self,
		checkpoint: str,
		quantization: Literal["8bit", "4bit"] | bool | None = None
	) -> None:
		self.load_checkpoint(
			checkpoint=checkpoint,
			quantization=quantization
		)

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

	def build_input(
		self,
		input_text: str,
		expected_text: str | None = None
	) -> Input:
		return Input(
			input_text=input_text,
			expected_text=expected_text
		)
	
	def _tokenize(
		self,
		data: Input,
		mode: Literal["inference", "dapt", "ft"] = "inference"
	) -> dict[str, Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing")

		if mode == "inference":
			tokenized_input_text = self.tokenizer(data.input_text)
			input_ids = torch.tensor(tokenized_input_text["input_ids"], dtype=torch.long).to(self.model.device)
			attention_mask = torch.tensor(tokenized_input_text["attention_mask"]).to(self.model.device)
			return {
				"input_ids": input_ids,
				"attention_mask": attention_mask
			}

		if mode == "dapt":
			input_text = data.input_text + (data.expected_text or "")
			tokenized = self.tokenizer(input_text)
			input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
			attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
			return {
				"input_ids": input_ids,
				"attention_mask": attention_mask
			}

		prompt_text = data.input_text
		full_text = data.input_text + (data.expected_text or "")

		encoded_prompt = self.tokenizer(prompt_text)
		encoded_full = self.tokenizer(full_text)

		input_ids = torch.tensor(encoded_full["input_ids"], dtype=torch.long)
		attention_mask = torch.tensor(encoded_full["attention_mask"], dtype=torch.bool)

		labels = input_ids.clone()
		prompt_len = len(cast(list, encoded_prompt["input_ids"]))

		labels[:prompt_len] = -100

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels
		}
		
	def train(
		self,
		train_dataset: list[Input],
		params: TrainParams | None = None,
		eval_dataset: list[Input] | None = None,
		mode: Literal["dapt", "ft"] = "dapt",
		save_at_end = True,
		save_path: str | None = None	
	) -> None:
		if not self.model:
			self._log("Could not find a model loaded. Try loading a model first.", LogLevel.WARNING)
			return None
		if not self.tokenizer:
			self._log("Could not find a tokenizer loaded. Try loading a tokenizer first.", LogLevel.WARNING)
			return None

		self._log("Starting Training")

		if self.model_is_quantized:
			self._log("Cannot train a quantized model.", LogLevel.WARNING)
			return None
		
		if params is None:
			params = TrainParams()

		training_arguments = SFTConfig(
			num_train_epochs=params.epochs,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			gradient_checkpointing=True,
			warmup_ratio=params.warmup_ratio,
			lr_scheduler_type="cosine_with_min_lr",
			lr_scheduler_kwargs={"min_lr_rate": 0.1},
			label_smoothing_factor=params.label_smoothing_factor,
			output_dir=None,
			save_strategy="no",
			logging_steps=params.logging_steps
		)

		if self.seed is not None:
			training_arguments.seed = self.seed

		tokenized_train_dataset = Dataset.from_list([self._tokenize(data, mode) for data in train_dataset])

		tokenized_eval_dataset = None
		if eval_dataset:
			tokenized_eval_dataset = Dataset.from_list([self._tokenize(data, mode) for data in eval_dataset])

		log_callback = LogCollectorCallback()

		trainer = SFTTrainer(
			model=self.model,
			train_dataset=tokenized_train_dataset,
			eval_dataset=tokenized_eval_dataset,
			args=training_arguments,
			callbacks=[log_callback]
		)

		trainer.train()

		if save_at_end and save_path:
			self.save_checkpoint(
				path=save_path
			)

		self._log("Finished Training")

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
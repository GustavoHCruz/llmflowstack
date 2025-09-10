import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import torch
from colorama import Fore, Style, init
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from llmflow.callbacks.log_collector import LogCollectorCallback
from llmflow.schemas.params import GenerationParams, LoraParams, TrainParams
from llmflow.utils.exceptions import MissingEssentialProp


class BaseModel(ABC):
	model = None
	tokenizer = None
	adapter: Any = None
	_model_id = None
	model_is_quantized = None
	seed = None
	log_level: Literal["INFO", "DEBUG", "WARNING"] = "INFO"
	stop_token_ids = []

	def __init__(
		self,
		checkpoint: str | None = None,
		adapter_path: str | None = None,
		quantization: Literal["8bit", "4bit"] | bool | None = None,
		seed: int | None = None,
		log_level: Literal["INFO", "DEBUG", "WARNING"] = "INFO",
	) -> None:
		init(autoreset=True)
		if seed:
			self._set_seed(seed)

		self._base_model = checkpoint

		self._set_logger(log_level)
		self.log_level = log_level

		self.tokenizer: PreTrainedTokenizerBase | None = None

		if checkpoint:
			self._checkpoint = checkpoint
			self.load_checkpoint(
				checkpoint=checkpoint,
				adapter_path=adapter_path,
				quantization=quantization
			)

	def from_pretrained(
		self,
		checkpoint: str,
		adapter_path: str | None = None,
		quantization: Literal["8bit", "4bit"] | bool | None = None
	) -> None:
		self.load_checkpoint(
			checkpoint=checkpoint,
			adapter_path=adapter_path,
			quantization=quantization
		)
		with open(os.path.join(checkpoint, "custom_info.json"), "r") as f:
			data = json.load(f)
		self._model_id = data.get("model_id", None)
	
	def _create_model_id(
		self
	) -> None:
		self._model_id = uuid4()

	def _set_logger(
		self,
		level: str
	) -> None:
		logging.basicConfig(
			level=level,
			format="%(asctime)s - %(levelname)s - %(message)s"
		)
		self.logger = logging.getLogger(__name__)

	def _log(
		self,
		info: str,
		level: Literal["INFO", "WARNING", "ERROR", "DEBUG"] = "INFO"
	) -> None:
		if level == "INFO":
			colored_msg = f"{Fore.GREEN}{info}{Style.RESET_ALL}"
			self.logger.info(colored_msg)
		elif level == "WARNING":
			colored_msg = f"{Fore.YELLOW}{info}{Style.RESET_ALL}"
			self.logger.warning(colored_msg)
		elif level == "ERROR":
			colored_msg = f"{Fore.RED}{info}{Style.RESET_ALL}"
			self.logger.error(colored_msg)
		elif level == "DEBUG":
			colored_msg = f"{Fore.BLUE}{info}{Style.RESET_ALL}"
			self.logger.debug(colored_msg)

	def _set_seed(self, seed):
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

	def create_lora_adapter(
		self,
		lora_params: LoraParams
	) -> None:
		if not self.model:
			self._log("Could not find a model loaded. Try loading a model first.", "WARNING")
			return

		self._log("Creating a new LoRA adapter...")

		lora_config = LoraConfig(
				r=lora_params.r,
				lora_alpha=lora_params.alpha,
				target_modules=lora_params.target_modules,
				lora_dropout=lora_params.dropout,
				layers_to_transform=lora_params.layers,
				bias=lora_params.bias,
				task_type="CAUSAL_LM"
			)

		self.adapter = get_peft_model(self.model, lora_config)

		self._log("LoRA adapter successfully created")

	def save_checkpoint(
		self,
		path: str,
		target: Literal["model", "adapter"],
		merge_and_load = False
	) -> None:
		if not self.model:
			self._log("No model to save.", "WARNING")
			return None
		if not self.tokenizer:
			self._log("No tokenizer to save.", "WARNING")
			return None

		os.makedirs(path, exist_ok=True)

		if target == "adapter":
			if not self.adapter:
				self._log("No adapter to save.", "WARNING")
				return None
			if merge_and_load:
				self._log("Merging LoRA adapters and saving...")
				model_to_save = self.adapter.merge_and_unload(progressbar=True)
			else:
				self._log("Saving LoRA adapters...")
				model_to_save = self.adapter
		else:
			self._log("Saving model...")
			model_to_save = self.model

		model_to_save.save_pretrained(path)
		self.tokenizer.save_pretrained(path)

		self._log(f"Model and Tokenizer saved at {path}")

		with open(os.path.join(path, "custom_info.json"), "w") as f:
			json.dump({
				"model_id": self._model_id
			}, f)

		self._log(f"Model custom information saved at {path}")

	def _load_tokenizer(self, checkpoint: str) -> None:
		tokenizer = AutoTokenizer.from_pretrained(checkpoint)
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.add_eos_token = True
		tokenizer.padding_side = "right"

		self.tokenizer = tokenizer

	@abstractmethod
	def load_checkpoint(
		self,
		checkpoint: str,
		adapter_path: str | None = None,
		quantization: Literal["8bit", "4bit"] | bool | None = None
	) -> None:
		pass

	def merge_adapter(
		self
	) -> None:
		if not self.model or not self.adapter:
			self._log("There is not a model or adapter loaded. Aborting merge.", "WARNING")
		
		self.model = self.adapter.merge_and_unload(progressbar=True)

	@abstractmethod
	def _set_generation_stopping_tokens(
		self,
		tokens: list[int]
	) -> None:
		pass

	def _tokenize(
		self,
		input_text: str
	) -> tuple[Tensor, Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing")

		tokenized_input_text: BatchEncoding = self.tokenizer(
			input_text,
			return_tensors="pt"
		).to(self.model.device)

		input_ids = tokenized_input_text["input_ids"]
		input_ids = cast(Tensor, input_ids)
		attention_mask = tokenized_input_text["attention_mask"]
		attention_mask = cast(Tensor, attention_mask)
		return (input_ids, attention_mask)

	def _tokenize_for_dapt(
		self,
		input_text: str
	) -> tuple:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing")

		tokenized = self.tokenizer(
			input_text
		)

		input_ids = tokenized["input_ids"]
		attention_mask = tokenized["attention_mask"]

		return input_ids, attention_mask

	def _tokenize_dataset_for_dapt(
		self,
		dataset: list[str]
	) -> Dataset:
		tokenized = []
		for input_text in dataset:
			tokenized_input = self._tokenize_for_dapt(input_text)
			if tokenized_input:
				input_ids, attention_mask = tokenized_input
				tokenized.append({
					"input_ids": input_ids,
					"attention_mask": attention_mask
				})
		return Dataset.from_list(tokenized)

	def _tokenize_for_fine_tune(
		self,
		input_text: str,
		expected_text: str
	) -> tuple[Tensor, Tensor, Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing")

		encoded_input = self.tokenizer(
			input_text
		)
		encoded_expected = self.tokenizer(
			expected_text
		)

		input_ids = torch.tensor(encoded_expected["input_ids"], dtype=torch.long)
		attention_mask = torch.tensor(encoded_expected["attention_mask"], dtype=torch.bool)

		labels = torch.full_like(input_ids, -100)

		start = len(cast(list, encoded_input["input_ids"]))

		labels[start:] = input_ids[start:]

		return input_ids, attention_mask, labels

	def _tokenize_dataset_for_fine_tune(
		self,
		dataset: list[dict[Literal["partial", "complete"], str]]
	) -> Dataset:
		tokenized = []

		for data in dataset:
			tokenized_input = self._tokenize_for_fine_tune(
				input_text=data["partial"],
				expected_text=data["complete"]
			)

			input_ids, attention_mask, labels = tokenized_input
			tokenized.append({
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"labels": labels
			})
		return Dataset.from_list(tokenized)

	@abstractmethod
	def _promptfy_dataset_for_dapt(
		self,
		dataset: list[Any]
	) -> list[str]:
		pass

	def dapt(
		self,
		train_params: TrainParams,
		train_dataset: list[Any],
		eval_dataset: list[Any] | None = None,
		target: Literal["model", "adapter"] = "model",
		save_at_end = True,
		save_path: str | None = None
	) -> None:
		if not self.model:
			self._log("Could not find a model loaded. Try loading a model first.", "WARNING")
			return
		if target=="adapter" and not self.adapter:
			self._log("Could not find a adapter loaded.", "WARNING")
		if not self.tokenizer:
			self._log("Could not find a tokenizer loaded. Try loading a tokenizer first.", "WARNING")
			return

		self._log("Starting DAPT")

		if target == "model":
			if self.model_is_quantized:
				self._log("Cannot DAPT a quantized model.", "WARNING")
				return None

		training_arguments = SFTConfig(
			num_train_epochs=train_params.epochs,
			learning_rate=train_params.lr,
			gradient_accumulation_steps=train_params.gradient_accumulation,
			warmup_ratio=0.03,
			lr_scheduler_type="cosine_with_min_lr",
			lr_scheduler_kwargs={"min_lr_rate": 0.1},
			save_strategy="no",
			logging_steps=1
		)

		if self.seed is not None:
			training_arguments.seed = self.seed

		if target == "model":
			model = self.model
		else:
			model = self.adapter

		processed_train_dataset = self._promptfy_dataset_for_dapt(train_dataset)
		tokenized_train_dataset = self._tokenize_dataset_for_dapt(processed_train_dataset)

		tokenized_eval_dataset = None
		if eval_dataset:
			processed_eval_dataset = self._promptfy_dataset_for_dapt(eval_dataset)
			tokenized_eval_dataset = self._tokenize_dataset_for_dapt(processed_eval_dataset)

		log_callback = LogCollectorCallback()

		trainer = SFTTrainer(
			model=model,
			train_dataset=tokenized_train_dataset,
			eval_dataset=tokenized_eval_dataset,
			args=training_arguments,
			callbacks=[log_callback]
		)

		trainer.train()

		if save_at_end and save_path:
			self.save_checkpoint(
				path=save_path,
				target=target
			)

		if target == "model":
			self.model = model
		else:
			self.adapter = model

		self._log("Finished DAPT")

	@abstractmethod
	def _build_input_for_fine_tune(
		self,
		input: Any
	) -> dict[Literal["partial", "complete"], str]:
		pass

	def _promptfy_dataset_for_fine_tune(
		self,
		dataset: list[Any]
	) -> list[dict[Literal["partial", "complete"], str]]:
		output = []
		for data in dataset:
			builded_inputs = self._build_input_for_fine_tune(
				input=data
			)
			output.append(builded_inputs)

		return output

	def fine_tune(
		self,
		train_params: TrainParams,
		train_dataset: list[Any],
		eval_dataset: list[Any] | None = None,
		target: Literal["model", "adapter"] = "model",
		save_at_end = True,
		save_path: str | None = None
	) -> None:
		if not self.model:
			self._log("Could not find a model loaded. Try loading a model first.", "WARNING")
			return
		if target=="adapter" and not self.adapter:
			self._log("Could not find a adapter loaded.", "WARNING")
		if not self.tokenizer:
			self._log("Could not find a tokenizer loaded. Try loading a tokenizer first.", "WARNING")
			return

		self._log("Starting fine-tune")

		if target == "model":
			if self.model_is_quantized:
				self._log("Cannot fine-tune a quantized model.", "WARNING")
				return None

		training_arguments = SFTConfig(
			learning_rate=train_params.lr,
			gradient_checkpointing=True,
			num_train_epochs=train_params.epochs,
			gradient_accumulation_steps=train_params.gradient_accumulation,
			warmup_ratio=0.03,
			lr_scheduler_type="cosine_with_min_lr",
			lr_scheduler_kwargs={"min_lr_rate": 0.1},
			save_strategy="no",
			logging_steps=1
		)

		if self.seed is not None:
			training_arguments.seed = self.seed

		if target == "model":
			model = self.model
		else:
			model = self.adapter

		processed_train_dataset = self._promptfy_dataset_for_fine_tune(train_dataset)
		tokenized_train_dataset = self._tokenize_dataset_for_fine_tune(processed_train_dataset)

		tokenized_eval_dataset = None
		if eval_dataset:
			processed_eval_dataset = self._promptfy_dataset_for_fine_tune(eval_dataset)
			tokenized_eval_dataset = self._tokenize_dataset_for_fine_tune(processed_eval_dataset)

		log_callback = LogCollectorCallback()

		trainer = SFTTrainer(
			model=model,
			train_dataset=tokenized_train_dataset,
			eval_dataset=tokenized_eval_dataset,
			args=training_arguments,
			callbacks=[log_callback]
		)

		trainer.train()

		if save_at_end and save_path:
			self.save_checkpoint(
				path=save_path,
				target=target
			)

		if target == "model":
			self.model = model
		else:
			self.adapter = model

		self._log("Finished fine-tune")

	@abstractmethod
	def generate(
		self,
		input: Any,
		params: GenerationParams | None = None,
		target: Literal["model", "adapter"] = "model"
	) -> str | None:
		pass

	def unload_adapter(self) -> None:
		try:
			self._log("Trying to reset adapter...")
			del self.adapter
			self.adapter = None
			self._log("Adapter successfully reseted")
		except Exception as e:
			self._log("Couldn't reset adapter...", "ERROR")
			self._log(f"{str(e)}", "DEBUG")

	def unload_model(self) -> None:
		try:
			self._log("Trying to reset model...")
			del self.model
			self.model = None
			self.model_is_quantized = None
			self.process_id = None
		except Exception as e:
			self._log("Couldn't reset model...", "ERROR")
			self._log(f"{str(e)}", "DEBUG")

	def unload_all(self) -> None:
		self.unload_adapter()
		self.unload_model()

	def set_seed(self, seed: int) -> None:
		self._log(f"Setting seed value {seed}")
		self._set_seed(seed)
		self._log(f"Seed setted")
import gc
from logging import getLogger

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class BaseEncoder:
	encoder: SentenceTransformer | None

	def __init__(
		self,
		checkpoint: str
	) -> None:
		self.logger = getLogger(f"LLMFlowStack.{self.__class__.__class__}")

		self.load_encoder(
			checkpoint=checkpoint
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
	
	def load_encoder(
		self,
		checkpoint: str
	) -> None:
		if self.encoder:
			self._log("A encoder is already loaded. Attempting to reset it.", LogLevel.WARNING)
			self.unload_model()

		self._log(f"Loading encoder on '{checkpoint}'")
		self.encoder = SentenceTransformer(
			checkpoint,
			trust_remote_code=True
		)

		self._log("Encoder loaded")
	
	def encode(
		self,
		message: str
	) -> Tensor:
		if not self.encoder:
			raise MissingEssentialProp("Could not find encoder.")
		return self.encoder.encode(message)
	
	def unload_model(
		self
	) -> None:
		try:
			del self.encoder
			gc.collect()
			torch.cuda.empty_cache()
			self.encoder = None
			self._log("Reset successfully.")
		except Exception as e:
			self._log("Couldn't reset model...", LogLevel.ERROR)
			self._log(f"{str(e)}", LogLevel.DEBUG)
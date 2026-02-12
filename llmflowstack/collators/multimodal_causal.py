from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MultimodalCausalCollator:
	tokenizer: Any
	label_pad_token_id: int = -100
	pad_to_multiple_of: int | None = None

	def _get(
		self,
		x: Any,
		key: str
	) -> Any:
		if isinstance(x, dict):
			return x.get(key, None)
		return getattr(x, key, None)

	def __call__(
		self,
		batch: list[Any]
	) -> dict[str, torch.Tensor]:
		pad_id = self.tokenizer.pad_token_id
		if pad_id is None:
			pad_id = self.tokenizer.eos_token_id

		input_seqs = []
		attn_seqs = []
		label_seqs = []
		ttype_seqs = []
		pixels = []

		has_token_type = False
		has_pixels = False

		max_len = 0
		for item in batch:
			ids = self._get(item, "input_ids")
			attn = self._get(item, "attention_mask")
			labels = self._get(item, "labels")

			if ids is None or attn is None or labels is None:
				raise ValueError("Each item must provide input_ids, attention_mask, and labels.")

			if not isinstance(ids, torch.Tensor):
				ids = torch.tensor(ids, dtype=torch.long)
			if not isinstance(attn, torch.Tensor):
				attn = torch.tensor(attn, dtype=torch.long)
			if not isinstance(labels, torch.Tensor):
				labels = torch.tensor(labels, dtype=torch.long)

			seq_len = ids.numel()
			if seq_len > max_len:
				max_len = seq_len

			input_seqs.append(ids)
			attn_seqs.append(attn)
			label_seqs.append(labels)

			tt = self._get(item, "token_type_ids")
			if tt is not None:
				has_token_type = True
				if not isinstance(tt, torch.Tensor):
					tt = torch.tensor(tt, dtype=torch.long)
				ttype_seqs.append(tt)
			else:
				ttype_seqs.append(None)

			pv = self._get(item, "pixel_values")
			if pv is not None:
				has_pixels = True
				if not isinstance(pv, torch.Tensor):
					pv = torch.tensor(pv)
				pixels.append(pv)
			else:
				pixels.append(None)

		if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
			max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

		bs = len(batch)

		input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
		attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
		labels = torch.full((bs, max_len), self.label_pad_token_id, dtype=torch.long)

		token_type_ids = None
		if has_token_type:
			token_type_ids = torch.zeros((bs, max_len), dtype=torch.long)

		for i in range(bs):
			ids = input_seqs[i]
			attn = attn_seqs[i]
			lab = label_seqs[i]

			seq_len = ids.numel()
			input_ids[i, :seq_len] = ids
			attention_mask[i, :seq_len] = attn
			labels[i, :seq_len] = lab

			if has_token_type and ttype_seqs[i] is not None:
				tt = ttype_seqs[i]
				token_type_ids[i, :tt.numel()] = tt

		out: dict[str, torch.Tensor] = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels
		}

		if has_token_type:
			out["token_type_ids"] = token_type_ids

		if has_pixels:
			if any(p is None for p in pixels):
				raise ValueError("Mixed batches with and without pixel_values. Use a separate dataloader or ensure consistency.")
			out["pixel_values"] = torch.stack(pixels, dim=0)

		return out

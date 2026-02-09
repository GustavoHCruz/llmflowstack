from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import BoolTensor, FloatTensor, LongTensor, topk
from transformers import LogitsProcessor, StoppingCriteria

JsonContainerType = Literal["obj", "arr"]
JsonPhase = Literal[
	"expect_value",
	"expect_key_or_end",
	"expect_colon",
	"expect_comma_or_end",
	"done"
]

@dataclass
class _Container:
	typ: JsonContainerType
	phase: JsonPhase

class JsonPrefixValidator:
	def __init__(
		self
	) -> None:
		self.reset()

	def reset(
		self
	) -> None:
		self.stack: list[_Container] = []
		self.in_string = False
		self.escape = False

		self.partial_literal: str | None = None
		self.partial_number = False
		self.top_phase: JsonPhase = "expect_value"

	def _current_phase(
		self
	) -> JsonPhase:
		if self.stack:
			return self.stack[-1].phase
		return self.top_phase

	def _set_phase_after_value(
		self
	) -> None:
		if self.stack:
			self.stack[-1].phase = "expect_comma_or_end"
		else:
			self.top_phase = "done"

	def _set_phase_after_open_obj(
		self
	) -> None:
		self.stack.append(_Container("obj", "expect_key_or_end"))

	def _set_phase_after_open_arr(
		self
	) -> None:
		self.stack.append(_Container("arr", "expect_value"))

	def _close_container(
		self,
		close_char: str
	) -> bool:
		if not self.stack:
			return False
		top = self.stack[-1]
		if close_char == "}" and top.typ != "obj":
			return False
		if close_char == "]" and top.typ != "arr":
			return False

		if top.typ == "obj":
			if top.phase not in ("expect_key_or_end", "expect_comma_or_end"):
				return False
		else:
			if top.phase == "expect_value":
				pass
			elif top.phase != "expect_comma_or_end":
				return False

		self.stack.pop()
		self._set_phase_after_value()
		return True

	def _starts_number(
		self,
		ch: str
	) -> bool:
		return ch.isdigit() or ch == "-"

	def _consume_number(
		self,
		s: str,
		i: int
	) -> int:
		start = i
		n = len(s)

		if s[i] == "-":
			i += 1
			if i >= n:
				self.partial_number = True
				return i

		if i < n and s[i] == "0":
			i += 1
		elif i < n and s[i].isdigit():
			if s[i] == "0":
				i += 1
			else:
				while i < n and s[i].isdigit():
					i += 1
		else:
			return start

		if i < n and s[i] == ".":
			i += 1
			if i >= n or not s[i].isdigit():
				self.partial_number = True
				return i
			while i < n and s[i].isdigit():
				i += 1

		if i < n and s[i] in ("e", "E"):
			i += 1
			if i < n and s[i] in ("+", "-"):
				i += 1
			if i >= n or not s[i].isdigit():
				self.partial_number = True
				return i
			while i < n and s[i].isdigit():
				i += 1

		return i

	def _consume_literal(
		self,
		s: str,
		i: int
	) -> int:
		targets = ("true", "false", "null")
		for t in targets:
			if s.startswith(t, i):
				self.partial_literal = None
				return i + len(t)

		for t in targets:
			max_len = min(len(t), len(s) - i)
			if s[i:i + max_len] == t[:max_len]:
				self.partial_literal = s[i:i + max_len]
				return len(s)
		return i

	def validate_and_update(
		self,
		appended: str
	) -> bool:
		if not appended:
			return True

		self.partial_number = False
		self.partial_literal = None

		i = 0
		n = len(appended)

		while i < n:
			ch = appended[i]

			if self.in_string:
				if self.escape:
					self.escape = False
					i += 1
					continue
				if ch == "\\":
					self.escape = True
					i += 1
					continue
				if ch == "\"":
					self.in_string = False
					phase = self._current_phase()
					if self.stack and self.stack[-1].typ == "obj" and phase == "expect_key_or_end":
						self.stack[-1].phase = "expect_colon"
					else:
						self._set_phase_after_value()
					i += 1
					continue

				i += 1
				continue

			if ch.isspace():
				i += 1
				continue

			phase = self._current_phase()

			if ch == "{":
				if phase not in ("expect_value",):
					return False
				self._set_phase_after_open_obj()
				i += 1
				continue

			if ch == "[":
				if phase not in ("expect_value",):
					return False
				self._set_phase_after_open_arr()
				i += 1
				continue

			if ch == "}":
				if not self._close_container("}"):
					return False
				i += 1
				continue

			if ch == "]":
				if not self._close_container("]"):
					return False
				i += 1
				continue

			if ch == ":":
				if not self.stack or self.stack[-1].typ != "obj" or phase != "expect_colon":
					return False
				self.stack[-1].phase = "expect_value"
				i += 1
				continue

			if ch == ",":
				if not self.stack:
					return False
				top = self.stack[-1]
				if top.phase != "expect_comma_or_end":
					return False
				if top.typ == "obj":
					top.phase = "expect_key_or_end"
				else:
					top.phase = "expect_value"
				i += 1
				continue

			if ch == "\"":
				if self.stack and self.stack[-1].typ == "obj":
					if phase not in ("expect_key_or_end", "expect_value"):
						return False
				else:
					if phase != "expect_value":
						return False
				self.in_string = True
				self.escape = False
				i += 1
				continue

			if ch in ("t", "f", "n"):
				if phase != "expect_value":
					return False
				j = self._consume_literal(appended, i)
				if j == i:
					return False

				if j <= n and self.partial_literal is None:
					self._set_phase_after_value()
				i = j
				continue

			if self._starts_number(ch):
				if phase != "expect_value":
					return False
				j = self._consume_number(appended, i)
				if j == i:
					return False

				if not self.partial_number:
					self._set_phase_after_value()
				i = j
				continue
			
			return False

		return True

	def is_complete(self) -> bool:
		if self.in_string or self.escape:
			return False
		if self.stack:
			return False
		if self.top_phase != "done":
			return False
		if self.partial_number or self.partial_literal is not None:
			return False
		return True


class ForceJsonLogitsProcessor(LogitsProcessor):
	def __init__(
		self,
		tokenizer,
		top_k: int = 256
	) -> None:
		self.tokenizer = tokenizer
		self.top_k = top_k
		self.validator = JsonPrefixValidator()
		self._last_len = 0
		self._prefix_text = ""

	def _sync_prefix(
		self,
		input_ids: LongTensor
	) -> None:
		seq = input_ids[0]
		cur_len = int(seq.shape[0])
		if cur_len < self._last_len:
			self.validator.reset()
			self._prefix_text = self.tokenizer.decode(seq, skip_special_tokens=True)
			self._last_len = cur_len
			self.validator.validate_and_update(self._prefix_text)
			return

		if cur_len == self._last_len:
			return

		new_ids = seq[self._last_len:cur_len]
		new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
		self._prefix_text += new_text
		self._last_len = cur_len
		self.validator.validate_and_update(new_text)

	def __call__(
		self,
		input_ids: LongTensor,
		scores: FloatTensor
	) -> FloatTensor:
		self._sync_prefix(input_ids)

		if self.validator.is_complete():
			return scores

		k = min(self.top_k, scores.shape[-1])
		top = topk(scores[0], k=k)
		candidate_ids = top.indices.tolist()

		allowed: list[int] = []
		for tid in candidate_ids:
			token_text = self.tokenizer.decode([tid], skip_special_tokens=True)
			if token_text == "":
				continue

			backup = (
				[c.__dict__.copy() for c in self.validator.stack],
				self.validator.in_string,
				self.validator.escape,
				self.validator.partial_literal,
				self.validator.partial_number,
				self.validator.top_phase
			)

			ok = self.validator.validate_and_update(token_text)

			stack_dicts, in_string, escape, partial_literal, partial_number, top_phase = backup
			self.validator.stack = [_Container(d["typ"], d["phase"]) for d in stack_dicts]
			self.validator.in_string = in_string
			self.validator.escape = escape
			self.validator.partial_literal = partial_literal
			self.validator.partial_number = partial_number
			self.validator.top_phase = top_phase

			if ok:
				allowed.append(tid)

		if not allowed:
			k2 = min(2048, scores.shape[-1])
			top2 = topk(scores[0], k=k2)
			for tid in top2.indices.tolist():
				token_text = self.tokenizer.decode([tid], skip_special_tokens=True)
				if token_text == "":
					continue

				backup = (
					[c.__dict__.copy() for c in self.validator.stack],
					self.validator.in_string,
					self.validator.escape,
					self.validator.partial_literal,
					self.validator.partial_number,
					self.validator.top_phase
				)

				ok = self.validator.validate_and_update(token_text)

				stack_dicts, in_string, escape, partial_literal, partial_number, top_phase = backup
				self.validator.stack = [_Container(d["typ"], d["phase"]) for d in stack_dicts]
				self.validator.in_string = in_string
				self.validator.escape = escape
				self.validator.partial_literal = partial_literal
				self.validator.partial_number = partial_number
				self.validator.top_phase = top_phase

				if ok:
					allowed.append(tid)
					if len(allowed) >= 64:
						break

		if allowed:
			mask = scores.new_full(scores.shape, float("-inf"))
			mask[0, allowed] = 0.0
			scores = scores + mask

		return scores

class StopOnJsonComplete(StoppingCriteria):
	def __init__(
		self,
		processor: ForceJsonLogitsProcessor
	) -> None:
		self.processor = processor

	def __call__(
		self,
		input_ids: LongTensor,
		scores: FloatTensor,
		**kwargs
	) -> BoolTensor:
		self.processor._sync_prefix(input_ids)
		return self.processor.validator.is_complete()

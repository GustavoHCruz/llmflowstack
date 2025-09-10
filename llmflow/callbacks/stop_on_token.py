import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria


class StopOnToken(StoppingCriteria):
  def __init__(
    self,
    stop_token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    should_print: bool = False
  ) -> None:
    self.stop_token_ids = torch.tensor(stop_token_ids)
    self.tokenizer = tokenizer
    self.should_print = should_print

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
    last_token = input_ids[0, -1]
    stop_tokens = self.stop_token_ids.to(input_ids.device)
    if self.should_print and self.tokenizer:
      print(self.tokenizer.decode(last_token), end="", sep="")

    return (last_token == stop_tokens).any() # type: ignore
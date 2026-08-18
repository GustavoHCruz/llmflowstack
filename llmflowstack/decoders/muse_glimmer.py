from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from torchao.quantization import Float8WeightOnlyConfig
from transformers import TorchAoConfig
from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

from llmflowstack.decoders.base_decoder import BaseDecoder, DataInput, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class MuseGlimmer(BaseDecoder):
    model: MuseGlimmerForConditionalGeneration | None = None
    max_context_len = 32768
    can_handle_image_processing = True
    can_think = "True"

    def set_thinking_mode(self, can_think: bool) -> None:
        self.can_think = can_think

    def disable_reasoning(self) -> None:
        self.can_think = False

    def _set_generation_stopping_tokens(self, tokens: list[int]) -> None:
        if not self.tokenizer:
            self._log(
                "Could not set stop tokens - generation may not work...",
                LogLevel.WARNING,
            )
            return
        particular_tokens = self.tokenizer.encode("<|eot|>")
        self.stop_token_ids = tokens + particular_tokens

    def _load_model(
        self,
        checkpoint: str | Path,
        quantization: bool | None = None,
        max_memory: dict | None = None,
    ) -> None:
        quantization_config = None
        if quantization:
            quant_config = Float8WeightOnlyConfig()
            quantization_config = TorchAoConfig(quant_type=quant_config)

        self.model = MuseGlimmerForConditionalGeneration.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            attn_implementation="sdpa",
            dtype="auto",
            device_map="auto",
            max_memory=max_memory,
        )

    def _build_prompt(
        self,
        input_text: str,
        output_text: str | None = None,
        system_text: str | None = None,
        reasoning_text: str | None = None,
        image_paths: list[str] | None = None,
        images: list[Image.Image] | None = None,
    ) -> str:
        if not self.tokenizer:
            raise MissingEssentialProp("Could not find tokenizer.")

        bos_token = self.tokenizer.bos_token or ""

        system_content = ""
        if system_text:
            system_content = f"<|start|>system<|message|>{system_text}<|eot|>"

        user_content = input_text
        if image_paths is not None:
            image_text = len(image_paths or []) * "<|patch|>"
            for _ in image_paths:
                user_content += (
                    f"<|start|>user<|message|>{image_text}{input_text}<|eot|>"
                )

        assistant_content = "<|start|>assistant to=user"
        if output_text:
            if reasoning_text:
                assistant_content += (
                    f"<|start|>assistant to=self<|message|>{reasoning_text}<|eot|>"
                )
            assistant_content = (
                f"<|start|>assistant to=user<|message|>{output_text}<|eot|>"
            )
        else:
            if not self.can_think:
                assistant_content += "<|start|>assistant to=user<|message|>"

        return f"{bos_token}{system_content}{user_content}{assistant_content}"

    def build_input(
        self,
        input_text: str,
        output_text: str | None = None,
        system_text: str | None = None,
        image_paths: list[str] | None = None,
    ) -> ModelInput:
        return self._tokenize(
            input_text=input_text,
            output_text=output_text,
            system_text=system_text,
            image_paths=image_paths,
            follow_prompt_format=True,
        )

    def generate(
        self,
        data: DataInput,
        params: GenerationParams | None = None,
        force_json: bool = False,
    ) -> str | None:
        if self.model is None or self.tokenizer is None:
            self._log("Model or Tokenizer missing", LogLevel.WARNING)
            return None

        generation_outputs = self._generate(
            data=data,
            params=params,
            force_json=force_json,
            follow_prompt_format=True,
        )

        if generation_outputs is None:
            return None

        _, outputs = generation_outputs

        answer = outputs[0]

        decoded = self.tokenizer.decode(answer)

        if isinstance(decoded, list):
            decoded = decoded[0]

        start = decoded.rfind("to=user<|message|>")
        if start == -1:
            return ""

        start += len("to=user<|message|>")

        end = decoded.rfind("<|eot|>")
        if end == -1:
            end = len(decoded)

        return decoded[start:end].strip()

    def generate_stream(
        self,
        data: DataInput,
        params: GenerationParams | None = None,
        force_json: bool = False,
        follow_prompt_format: bool = True,
    ) -> Iterator[str]:
        streamer = self._generate_stream(
            data=data,
            params=params,
            force_json=force_json,
            follow_prompt_format=follow_prompt_format,
        )

        thinking = False
        buffer = ""

        for new_text in streamer:
            buffer += new_text

            if not thinking and "to=self<|message|>" in buffer:
                thinking = True
                buffer = buffer.split("to=self<|message|>", 1)[1]

            if thinking:
                if "to=user<|message|>" in buffer:
                    buffer = buffer.split("to=user<|message|>", 1)[1]
                    thinking = False
                else:
                    continue

            buffer = buffer.replace("<|eot|>", "")
            yield buffer
            buffer = ""

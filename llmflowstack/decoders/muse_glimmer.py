from collections.abc import Iterator
from pathlib import Path

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
            attn_implementation="auto",
            dtype="auto",
            device_map="auto",
            max_memory=max_memory,
        )

    def _build_prompt(
        self,
        input_text: str,
        output_text: str | None = None,
        system_text: str | None = None,
        image_paths: list[str] | None = None,
    ) -> str:
        if not self.tokenizer:
            raise MissingEssentialProp("Could not find tokenizer.")

        bos_token = self.tokenizer.bos_token or ""

        system_content = ""
        if system_text is not None:
            system_content = f"<|start|>system<|message|>{system_text}<|eot|>"

        image_text = len(image_paths or []) * "<|patch|>"

        user_content = f"<|start|>user<|message|>{image_text}{input_text}<|eot|>"

        if output_text is not None:
            assistant_content = (
                f"<|start|>assistant to=user<|message|>{output_text}<|eot|>"
            )
        else:
            assistant_content = "<|start|>assistant"

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
        if self.tokenizer is None:
            self._log("Tokenizer missing", LogLevel.WARNING)
            return

        generation_outputs = self._generate(
            data=data, params=params, force_json=force_json, follow_prompt_format=True
        )

        if generation_outputs is None:
            return

        start_index, outputs = generation_outputs

        answer = outputs[0][start_index:]

        decoded = self.tokenizer.decode(answer, skip_special_tokens=True)

        if isinstance(decoded, list):
            decoded = decoded[0]

        return decoded.strip()

    def generate_stream(
        self,
        data: DataInput,
        params: GenerationParams | None = None,
        force_json: bool = False,
        follow_prompt_format: bool = True,
    ) -> Iterator[str]:
        return self._generate_stream(
            data=data,
            params=params,
            force_json=force_json,
            follow_prompt_format=follow_prompt_format,
        )

from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from torchao.quantization import Float8WeightOnlyConfig
from transformers import AutoModelForMultimodalLM, TorchAoConfig

from llmflowstack.decoders.base_decoder import BaseDecoder, DataInput, ModelInput
from llmflowstack.schemas.params import GenerationParams
from llmflowstack.utils.exceptions import MissingEssentialProp
from llmflowstack.utils.logging import LogLevel


class Gemma4(BaseDecoder):
    model: AutoModelForMultimodalLM
    max_context_len = 32768
    can_handle_image_processing = True
    can_think = True

    def set_thinking_mode(self, can_think: bool) -> None:
        self.can_think = can_think

    def disable_reasoning(self) -> None:
        self.can_think = False

    def _chat_template_kwargs(self) -> dict[str, bool]:
        return {"enable_thinking": self.can_think}

    def _set_generation_stopping_tokens(self, tokens: list[int]) -> None:
        if not self.tokenizer:
            self._log(
                "Could not set stop tokens - generation may not work...",
                LogLevel.WARNING,
            )
            return
        particular_tokens = self.tokenizer.encode("<turn|>")
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

        self.model = AutoModelForMultimodalLM.from_pretrained(
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
        image_paths: list[Path | str] | None = None,
        images: list[Image.Image] | None = None,
    ) -> str:
        if not self.tokenizer:
            raise MissingEssentialProp("Could not find tokenizer.")

        can_think_content = ""
        if self.can_think:
            can_think_content = "<|think|>\n"

        system_content = ""
        if system_text:
            system_content += f"{system_text}\n"

        user_content = ""
        if image_paths is not None:
            for _ in image_paths:
                user_content += "<|image|>"
        if images is not None:
            for _ in images:
                user_content += "<|image|>"
        user_content += str(input_text)

        assistant_content = "<|turn>model\n"

        if output_text:
            assistant_content += f"<|channel>thought\n{reasoning_text or ''}<channel|>"
            assistant_content += f"{output_text}<turn|>"
        else:
            if not self.can_think:
                assistant_content += "<|channel>thought\n<channel|>"

        return (
            f"<bos>"
            f"<|turn>system\n{can_think_content}{system_content}<turn|>\n"
            f"<|turn>user\n{user_content}<turn|>\n"
            f"{assistant_content}"
        )

    def build_input(
        self,
        input_text: str,
        output_text: str | None = None,
        system_text: str | None = None,
        image_paths: list[Path | str] | None = None,
        images: list[Image.Image] | None = None,
    ) -> ModelInput:
        return self._tokenize(
            input_text=input_text,
            output_text=output_text,
            follow_prompt_format=True,
            system_text=system_text,
            image_paths=image_paths,
            images=images,
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

        decoded = self.tokenizer.decode(answer)

        if isinstance(decoded, list):
            decoded = decoded[0][start_index:]

        start = decoded.find("<channel|>", 0)
        if start == -1:
            start = 0
        else:
            start += len("<channel|>")

        end = decoded.find("<turn|>", start)
        if end == -1:
            end = len(decoded)

        return decoded[start:end].strip()

    def generate_stream(
        self,
        data: DataInput,
        params: GenerationParams | None = None,
        force_json: bool = False,
    ) -> Iterator[str]:
        streamer = self._generate_stream(
            data=data,
            params=params,
            force_json=force_json,
            follow_prompt_format=True,
            skip_special_tokens=False,
        )

        thinking = False
        buffer = ""
        for new_text in streamer:
            buffer += new_text

            if not thinking and "<|channel>" in buffer:
                thinking = True
                buffer = buffer.split("<|channel>", 1)[1]

            if thinking:
                if "<channel|>" in buffer:
                    buffer = buffer.split("<channel|>", 1)[1]
                    thinking = False
                else:
                    continue

            buffer = buffer.replace("<turn|>", "")
            yield buffer
            buffer = ""

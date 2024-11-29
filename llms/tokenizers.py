from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore
from open_flamingo import create_model_and_transforms


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        elif provider == "xai":
            # ?? unsure fur xai
            self.tokenizer = None
        elif provider == 'flamingo':
            # TODO: find a better way to do this maybe?
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
                cross_attn_every_n_layers=1
            )
            self.tokenizer = tokenizer
            self.image_processor = image_processor
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

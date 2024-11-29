"""Tools to generate from Gemini prompts."""

import random
import time
from typing import Any

from PIL import Image
from open_flamingo import create_model_and_transforms

#model = GenerativeModel("gemini-pro-vision")
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1

)


from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)


def generate_from_flamingo_completion(
    prompt,
    max_new_tokens: int,
):

    lang_x, vision_x = prompt
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=3,
    )

    return generated_text[0]

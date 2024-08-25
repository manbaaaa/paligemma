#!/usr/bin/env python3
# Copyright (c) 2024 Shaojie Li (shaojieli.nlp@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
import torch
from PIL import Image

from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from processing_paligemma import PaligemmaProcessor
from utils import load_hf_model


def get_model_inputs(
    processor: PaligemmaProcessor,
    prompt: str,
    image_filepath: str,
    device: str,
):
    image = Image.open(image_filepath)
    images = [iamge]
    prompts = [prompt]
    model_inputs = processor(images, prompts)
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(device)
    return model_inputs


def _smaple_topp(logits, top_p):
    probs_sort, probs_idx = torch.sort(logits, dim=-1, descending=True)
    cum_probs = torch.cumsum(probs_sort, dim=-1)
    mask = cum_probs - probs_sort > top_p
    probs_sort[mask] = 0
    # redistribute the prob to sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaligemmaProcessor,
    prompt: str,
    image_filepath: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
):
    model_inputs = get_model_inputs(processor, prompt, image_filepath, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            kv_cache=kv_cache,
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # sample the next token
        if do_sample:
            # apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_logits = _smaple_topp(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        assert next_token.size() == (1, 1)
        # remove the batch dimension
        next_token.squeeze(0)
        generated_tokens.append(next_token.item())
        if next_token.item() == stop_token:
            break

        # update the input_ids, attention_mask
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(input_ids, device=input_ids.device)],
            dim=-1,
        )
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    print("Prompt: ", prompt)
    print("Generated text: ", generated_text)


def main(
    model_path: str = None,
    prompt: str = None,
    image_filepath: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print("Loading model")
    model, tokenizer = load_hf_model(model_path, device=device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaligemmaProcessor(
        tokenizer,
        num_image_tokens=num_image_tokens,
        image_size=image_size,
    )

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            prompt,
            image_filepath,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
            device,
        )


if __name__ == "__main__":
    fire.Fire(main)

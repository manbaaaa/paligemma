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

import glob
import json
import os
from typing import Tuple

from safetensors import safe_open
from transformers import AutoTokenizer

from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    safetensor_files = glob.glob(os.path.join(model_path, "safetensors*.json"))

    tensors = {}
    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, frame_work="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    with open(os.path.join(model_path, "config.json")) as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    model = PaliGemmaForConditionalGeneration(config).to(device)

    model.load_state_dict(tensors)
    model.tie_weights()
    return model, tokenizer

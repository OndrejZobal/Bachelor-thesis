#!/usr/bin/env python3

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
)
from torch.utils.data import DataLoader
import os
import torch
import logging
from tqdm import tqdm
import json
from peft import AutoPeftModelForCausalLM

from my_params import parse_args
from my_data import load_data
from dataset_formators import select_mode

from torch.utils.data import DataLoader


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

args = parse_args()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

modify = select_mode(args.mode)

val_dataset = load_data(args.dev_filename, tokenizer, args, modify)
val_dataset = val_dataset[:1000]

logging.info('%s', all(y == 2048 for y in [len(data['input_ids']) for data in val_dataset]))
logging.info('%d', len(val_dataset))

try:
    led = AutoPeftModelForCausalLM.from_pretrained(args.model_name_or_path)
except ValueError:
    led = LEDForConditionalGeneration.from_pretrained(args.model_name_or_path)
    led.config.num_beams = args.beam_size
    led.config.max_length = args.max_source_length + args.max_target_length
    led.config.min_length = 3
    led.config.length_penalty = 1.0
    led.config.early_stopping = args.beam_size > 1
    led.config.no_repeat_ngram_size = 3
    led.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(["__python__"])[0]

led = led.to('cuda')


eval_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
led.eval() 

predictions = []
references = []

report = []

def extract_outputs(model, tokenizer, data):
    hypotheses = []
    references = []
    inputs = []
    with open('eval.log', 'w', encoding='utf-8') as file:
        for batch in tqdm(data, total=len(data)):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            global_attention_mask = batch['global_attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
            hidden_index = batch['hidden_index']

            tokenizer.bos_token = "__python__"
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, # encoded_input,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    max_length=args.max_target_length,
                    use_cache=True,
                    top_k=args.eval_top_k,
                    top_p=args.eval_top_p,
                    temperature=args.eval_temperature,
                    #diversity_penalty=2.1,
                    num_beams=args.beam_size,
                    #num_beam_groups=args.beam_size,
                    no_repeat_ngram_size=args.eval_no_repeat_ngrams,
                    do_sample=args.eval_top_k is not None and args.eval_top_p is not None,
                    num_return_sequences=1,
                    length_penalty=args.length_penalty
                )

            hypothesis = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
            reference = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            source = [tokenizer.decode(inp, skip_special_tokens=False) for inp in input_ids]
            print(f'{len(reference)=} {len(hypothesis)=} {len(source)=}')
            for r, h, s in zip(reference, hypothesis, source):
                report.append({ "label": [r], "prediction": h, "source": s})
            #report.append({ "label": [reference], "prediction": hypothesis, "source": source})

            print("------")
            print(hypothesis, "\n\n", reference)
            hypotheses.extend(hypothesis)
            references.extend(reference)
            inputs.extend(source)


try:
    extract_outputs(led, tokenizer, eval_dataloader)
except BaseException as e:
    print(e)

with open("report.jsonl", 'w') as file:
    for entry in report:
        json.dump(entry, file)
        file.write('\n')

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

# This file is based on a file from the CodeBERT project
# extensively modified by Ondřej Zobal.
# Original file:
# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py

import torch
import json
import logging
from io import open
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 nl,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.nl = nl

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=js['input']
            if code == "": continue
            source = code
            if 'description' in js:
                nl = js['description']
            else:
                nl = ''

            out_code = js['output']
            target = out_code
            if target == "": continue
            examples.append(
                Example(
                        idx = idx,
                        source= source,
                        target = target,
                        nl = nl,
                )
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 source_ids,
                 target_ids,
                 global_attention_mask,
                 input_mask,
                 output_mask,
                 hidden_index,
    ):
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.global_attention_mask = global_attention_mask
        self.input_mask = input_mask
        self.output_mask = output_mask
        self.hidden_index = hidden_index

def convert_examples_to_features(examples, tokenizer, args, convertor, only_source_tokens=False):
    """convert examples to token ids"""
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples)):
        source_tokens_length = args.max_source_length-2
        try:
            pairs = convertor(example.source, example.target, tokenizer, args.max_source_length-3, args.max_target_length-2)
        except ValueError:
            continue

        for source_tokens, target_tokens, global_attention_mask, hidden_index in pairs:
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
            target_ids = torch.tensor(target_ids)

            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = args.max_source_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_ids = torch.tensor(source_ids)

            global_attention_mask += [False] * (len(source_ids) - len(global_attention_mask))
            global_attention_mask = torch.tensor(global_attention_mask)
            input_mask = source_ids.ne(tokenizer.pad_token_id)
            output_mask = target_ids.ne(tokenizer.pad_token_id)

            if example_index < 5:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))
                logger.info("source_tokens: {}".format([t.replace('\u0120','_') + str(m.item()) + (('!' if gat else '') if m else "") for t, gat, m in zip(source_tokens, global_attention_mask, input_mask)]))
                logger.info("source_ids: {}".format(' '.join(map(str, [id.item() for id in source_ids]))))

                if not only_source_tokens:
                    logger.info("target_tokens: {}".format([x.replace('\u0120','_') + str(m.item()) for x,m in zip(target_tokens, output_mask)]))
                    logger.info("target_ids: {}".format(' '.join(map(str, [id.item() for id in target_ids]))))

            features.append(
                InputFeatures(
                    source_ids,
                    target_ids,
                    global_attention_mask,
                    input_mask,
                    output_mask,
                    hidden_index,
                )
            )
    return features

def load_data(file_name, tokenizer, args, convertor):
    train_examples = read_examples(file_name)
    train_features = convert_examples_to_features(train_examples, tokenizer, args, convertor)
    train_dataset = []
    for f in train_features:
        batch = {
            "input_ids": f.source_ids,
            "labels": f.target_ids,
            "attention_mask": f.input_mask,
            "decoder_attention_mask": f.output_mask,
            "global_attention_mask": f.global_attention_mask,
            "hidden_index": f.hidden_index,

        }
        train_dataset.append(batch)
    return train_dataset

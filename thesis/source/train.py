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

# This file was originally taken from the CodeBERT project before being
# extensively modified by Ondřej Zobal.
# Original file:
# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py

import os
import torch
import logging
from transformers import (
    BartForConditionalGeneration,
    LEDForConditionalGeneration,
    BartConfig,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback,
    EarlyStoppingCallback,
    AdamW,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LEDConfig,
)
from transformers.optimization import Adafactor
from peft import AutoPeftModelForCausalLM
from datetime import datetime

from my_params import parse_args
import my_data
from dataset_formators import select_mode


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

args = parse_args()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

modify = select_mode(args.mode)

qconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

matching_layers = []
not_matching_layers = []

try:
    logger.info("Loading model as an autoregressive model.")
    led = AutoPeftModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
    for name, param in led.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    from accelerate import load_checkpoint_and_dispatch
    led = load_checkpoint_and_dispatch(led, "wizard_coder-original-1B", device_map="auto", no_split_module_classes=["GPTBigCodeForCausalLM"])
    led.print_trainable_parameters()
except ValueError:
    logger.info("Loading model as a LED.")
    config = LEDConfig.from_pretrained(args.model_name_or_path)
    config.attention_mode = "tvm"
    led = LEDForConditionalGeneration.from_pretrained(args.model_name_or_path, ignore_mismatched_sizes=True, config=config)

    led.config.num_beams = args.beam_size
    led.config.max_length = args.max_target_length
    led.config.min_length = 3
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3
    config = BartConfig.from_pretrained(args.model_name_or_path)
    bart = BartForConditionalGeneration.from_pretrained('uclanlp/plbart-base')
    bart.config.max_length = 16

    # Print all parameter names and their dimension
    print(list(led.state_dict().keys()))

    if led != bart:
        for layer in led.state_dict().keys():
            if layer.replace("led", "model") in bart.state_dict().keys():
                matching_layers.append(layer)
            else:
                not_matching_layers.append(layer)
    else:
        matching_layers = list(bart.state_dict().keys())
    del bart

logger.info("Loading datasets...")
val_dataset = my_data.load_data(args.dev_filename, tokenizer, args, modify)
train_dataset = my_data.load_data(args.train_filename, tokenizer, args, modify)

training_args = TrainingArguments(
    do_train=args.do_train,
    do_eval=args.do_eval,
    num_train_epochs=args.total_epochs,
    evaluation_strategy="steps",
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir=args.output_dir,
    logging_steps=50,
    eval_steps=args.eval_steps,
    save_steps=args.eval_steps,
    warmup_steps=args.warmup_steps,
    save_total_limit=4,
    learning_rate=args.slow_lr,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    weight_decay=args.weight_decay,
    fp16=True,
)

def create_optimizer(
    model, weight_decay, slow_lr, fast_lr, slow_layers, warmup_steps, num_training_steps
):
    """
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    # This function was taken from huggingface transformer library and modified to allow for multiple learning rates.

    arg_to_scheduler = {
        "linear": get_linear_schedule_with_warmup,
    }

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and n not in slow_layers
            ],
            "weight_decay": weight_decay,
            "lr": fast_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and n not in slow_layers
            ],
            "weight_decay": 0.0,
            "lr": fast_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and n in slow_layers
            ],
            "weight_decay": weight_decay,
            "lr": slow_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and n in slow_layers
            ],
            "weight_decay": 0.0,
            "lr": slow_lr,
        },
    ]
    logger.info(
        "%s",
        [(len(group["params"]), group["lr"]) for group in optimizer_grouped_parameters],
    )
    optimizer_cls = Adafactor if False else AdamW
    optimizer_cls = AdamW
    optimizer_kwargs = {
        # 'betas': (self.args.adam_beta1, self.args.adam_beta2),
        "betas": (0.9, 0.999),
        # 'eps': self.args.adam_epsilon,
        "eps": 1e-8,
    }
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    schedule_func = get_linear_schedule_with_warmup
    scheduler = schedule_func(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler


num_training_steps = (
    len(train_dataset)
    / (args.train_batch_size * args.gradient_accumulation_steps)
    * args.total_epochs
)
logger.info("num_training_steps: %s", num_training_steps)

# Initially freezing slow layer as to not destroy knowlage.
if args.freeze_slow_layers_epochs > 0:
    logger.info("freezing pretrained layers")
    for name, module in led.named_modules():
        if name in matching_layers:
            for param in module.parameters():
                param.requires_grad = False
else:
    logger.info("No layers were froze")


class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.loss = []
        self.eval_loss = []

    def __str__(self):
        out = ""
        for step, logs in self.loss:
            out += (
                "{"
                + f'"step": "{step}", "loss": "{logs["loss"]}", "learning_rate": "{logs["learning_rate"]}", "epoch": "{logs["epoch"]}" "type": "training"'
                + "}\n"
            )
        for step, logs in self.eval_loss:
            out += (
                "{"
                + f'"step": "{step}", "loss": "{logs["eval_loss"]}", "epoch": "{logs.get("epoch", "n/a")}", type": "eval"'
                + "}\n"
            )
        return out

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.loss.append((state.global_step, logs))
        if "eval_loss" in logs:
            self.eval_loss.append((state.global_step, logs))


trainer_log = LoggingCallback()

class UnfreezeCallback(TrainerCallback):
    """A callback that unfreezes model layers after the first epoch."""

    def __init__(self, unfreeze_after_epoch):
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.has_unfrozen = False
        self.trigger_step = None

    def on_epoch_begin(self, _args, state, control, **kwargs):
        self.trigger_step = (
            int(num_training_steps / args.total_epochs * self.unfreeze_after_epoch)
            + state.global_step
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.trigger_step and not self.has_unfrozen:
            for name, module in led.named_modules():
                for param in module.parameters():
                    param.requires_grad = True
            logger.info(
                "Unfroze layers at %.2f%% of the epoch", self.unfreeze_after_epoch * 100
            )
            self.has_unfrozen = True

# instantiate trainer
trainer = Trainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(
        create_optimizer(
            model=led,
            weight_decay=args.weight_decay,
            slow_lr=args.slow_lr,
            fast_lr=args.fast_lr,
            slow_layers=matching_layers,
            warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
        )
    ),
    callbacks=[
        UnfreezeCallback(args.freeze_slow_layers_epochs),
        EarlyStoppingCallback(early_stopping_patience=2),
        trainer_log,
    ],
)

start_timestamp = str(datetime.now())
trainable_params = led.num_parameters()
logger.info("Number of trainable parameters: %d", trainable_params)
try:
    result = trainer.train()
    logger.info("%s", result)
except BaseException as e:
    logger.info("Error %s", e)
    pass

del train_dataset
del val_dataset

led.save_pretrained(args.save_name)
tokenizer.save_pretrained(args.save_name)

# Log training into models save directory
try:
    end_timestamp = str(datetime.now())
    mode = "a" if os.path.realpath(args.save_name) == os.path.realpath(args.model_name_or_path) else "w"
    with open(args.save_name + "/log.jsonl", mode, encoding="utf-8") as file:
        file.write('{"' f'session_start="{start_timestamp}"' f'sesstion_end="{end_timestamp}"' '"}\n')
        file.write(str(trainer_log))
    with open(args.save_name + "/envvars.txt", mode, encoding="utf-8") as file:
        file.write('{"' f'session_start="{start_timestamp}"' f'sesstion_end="{end_timestamp}"' '"}\n')
        file.write(str(args))
except Exception as e:
    logger.error("Error saving logs: %s", e)

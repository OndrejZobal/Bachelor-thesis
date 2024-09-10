#!/bin/sh

mode=comment_generation
dev_file=./plbart-codexglue_valid.jsonl
pretrained_model=./${mode}_trained

python3 gpt-eval.py \
    --model_name_or_path $pretrained_model \
    --dev_filename $dev_file \
    --mode $mode \

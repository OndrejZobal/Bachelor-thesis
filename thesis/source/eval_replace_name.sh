#!/bin/sh

mode=rename
batch_size_eval=1
source_length=2048
target_length=16
output_dir=.
dev_file=./plbart-mask_codexglue_valid.jsonl
model_name_or_path=./${mode}_trained
save_name=./${mode}_trained
no_repeat_ngrams=3
top_k=20
beam_size=10
length_penalty=20

python3 eval.py \
    --do_train --do_eval \
    --dev_filename $dev_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --eval_batch_size $batch_size_eval  \
    --model_name_or_path $model_name_or_path \
    --save_name $save_name \
    --mode $mode \
    --length_penalty $length_penalty\
    --eval_no_repeat_ngrams $no_repeat_ngrams \
    --eval_top_k $top_k \

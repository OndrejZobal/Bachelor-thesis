#!/bin/sh

batch_size=5
batch_size_eval=6
beam_size=5
source_length=2048
target_length=512
output_dir=.
train_file=plbart-codexglue_train.jsonl
dev_file=./plbart-codexglue_valid.jsonl
eval_steps=1500
model_name_or_path=plbart-led
gradient_accumulation_steps=4
warmup_steps=1500
weight_decay=0.002
total_epochs=2
slow_lr=1e-5
fast_lr=1e-5
freeze_slow_layers_epochs=1
mode=comment_generation_special
save_name=${mode}_trained

python3 train.py \
    --do_train --do_eval \
    --train_filename $train_file\
    --dev_filename $dev_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size_eval  \
    --model_name_or_path $model_name_or_path \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --warmup_steps $warmup_steps \
    --weight_decay $weight_decay \
    --total_epochs $total_epochs \
    --slow_lr $slow_lr \
    --fast_lr $fast_lr \
    --freeze_slow_layers_epochs=$freeze_slow_layers_epochs \
    --eval_steps $eval_steps \
    --save_name $save_name \
    --mode $mode \

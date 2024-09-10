#!/bin/sh

mode=wizard
batch_size=4
batch_size_eval=5
beam_size=1
source_length=2048
target_length=2048
output_dir=.
train_file=plbart-bugfixes_train.jsonl
dev_file=plbart-bugfixes_valid.jsonl
eval_steps=200
pretrained_model=wizard_trained
gradient_accumulation_steps=1
warmup_steps=100
weight_decay=0
total_epochs=7
slow_lr=1e-5
fast_lr=3e-5
freeze_slow_layers_epochs=0
save_name=${mode}_trained

python3 train.py \
    --do_train --do_eval \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file\
    --dev_filename $dev_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size_eval  \
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

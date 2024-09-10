#!/bin/sh

mode=rename
dev_file=./plbart-mask_codexglue_valid.jsonl
pretrained_model=./${mode}_trained

python3 gpt-eval.py \
    --model_name_or_path $pretrained_model \
    --dev_filename $dev_file \
    --mode $mode \

# added dual learning rates and decay
# {'loss': 16.4896, 'learning_rate': 3.3333333333333335e-07, 'epoch': 0.02}
# {'loss': 8.8521, 'learning_rate': 6.666666666666667e-07, 'epoch': 0.04}
# {'loss': 1.3223, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.05}
# {'loss': 0.5863, 'learning_rate': 1.3333333333333334e-06, 'epoch': 0.07}
# {'loss': 0.511, 'learning_rate': 1.6666666666666667e-06, 'epoch': 0.09}
# {'loss': 0.4598, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.11}
# {'loss': 0.3828, 'learning_rate': 2.3333333333333336e-06, 'epoch': 0.12}
# {'loss': 0.3672, 'learning_rate': 2.666666666666667e-06, 'epoch': 0.14}
# {'loss': 0.3342, 'learning_rate': 3e-06, 'epoch': 0.16}
# {'loss': 0.3348, 'learning_rate': 3.3333333333333333e-06, 'epoch': 0.18}
# {'eval_loss': 0.3068995475769043, 'eval_runtime': 703.8936, 'eval_samples_per_second': 32.827, 'eval_steps_per_second': 4.104, 'epoch': 0.18}
# {'loss': 0.3068, 'learning_rate': 3.6666666666666666e-06, 'epoch': 0.2}
# {'loss': 0.3023, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.21}
# {'loss': 0.2887, 'learning_rate': 4.333333333333334e-06, 'epoch': 0.23}
# {'loss': 0.2968, 'learning_rate': 4.666666666666667e-06, 'epoch': 0.25}
# {'loss': 0.2755, 'learning_rate': 5e-06, 'epoch': 0.27}
# {'loss': 0.2816, 'learning_rate': 5.333333333333334e-06, 'epoch': 0.28}
# {'loss': 0.2906, 'learning_rate': 5.666666666666667e-06, 'epoch': 0.3}
# {'loss': 0.2921, 'learning_rate': 6e-06, 'epoch': 0.32}
# {'loss': 0.2904, 'learning_rate': 6.333333333333333e-06, 'epoch': 0.34}
# {'loss': 0.2507, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.36}
# {'eval_loss': 0.2577306926250458, 'eval_runtime': 705.4234, 'eval_samples_per_second': 32.756, 'eval_steps_per_second': 4.095, 'epoch': 0.36}
# {'loss': 0.2527, 'learning_rate': 7e-06, 'epoch': 0.37}
# {'loss': 0.2809, 'learning_rate': 7.333333333333333e-06, 'epoch': 0.39}
# {'loss': 0.2727, 'learning_rate': 7.666666666666667e-06, 'epoch': 0.41}

# stock script pl-bart
# {'loss': 12.2148, 'learning_rate': 1.6666666666666667e-06, 'epoch': 0.11}
# {'loss': 0.7283, 'learning_rate': 3.3333333333333333e-06, 'epoch': 0.21}
# {'loss': 0.351, 'learning_rate': 5e-06, 'epoch': 0.32}
# {'loss': 0.295, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.43}
# {'loss': 0.288, 'learning_rate': 8.333333333333334e-06, 'epoch': 0.53}
# {'loss': 0.2524, 'learning_rate': 1e-05, 'epoch': 0.64}
# {'loss': 0.2384, 'learning_rate': 1.1666666666666668e-05, 'epoch': 0.75}
# {'loss': 0.2248, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.85}
# {'loss': 0.2218, 'learning_rate': 1.5e-05, 'epoch': 0.96}
# {'loss': 0.1991, 'learning_rate': 1.6666666666666667e-05, 'epoch': 1.07}

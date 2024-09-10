#!/usr/bin/env python3

import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model: e.g. roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        required=False,
        help="Path to a checkpoint directory",
    )

    # Other parameters
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Low learning rate in the begining."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Penalty for high weights"
    )
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=10,
        help="Total dataset iterations of training",
    )
    parser.add_argument(
        "--slow_lr", type=float, default=5e-5, help="Learning rate for the slow layers"
    )
    parser.add_argument(
        "--fast_lr", type=float, default=5e-5, help="Learning rate for the fast layers"
    )
    parser.add_argument(
        "--freeze_slow_layers_epochs",
        type=float,
        default=0.0,
        help="For how long to freeze slow layers at the begging of training.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Number of steps before evaluation"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="trained_model",
        help="Name of the directory with the final trained model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="no mode selected",
        help="Name of the dataset conversion mode",
    )
    parser.add_argument(
        "--eval_no_repeat_ngrams",
        type=int,
        default=None,
        help="Prevent repeating ngrams during evaluation",
    )
    parser.add_argument(
        "--eval_top_k",
        type=int,
        default=None,
        help="Top K sampeling value",
    )
    parser.add_argument(
        "--eval_top_p",
        type=float,
        default=None,
        help="Top P sampeling value",
    )
    parser.add_argument(
        "--eval_temperature",
        type=float,
        default=1,
        help="Noisyness of the generation process",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Disincentivise longer answers",
    )
    # Print arguments
    args = parser.parse_args()
    logging.info("Loaded arguments: \n%s", args)

    return args

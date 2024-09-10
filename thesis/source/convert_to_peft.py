#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch

NAME = "WizardLM/WizardCoder-1B-V1.0"
PATH = "peft-wizard-1b"

# Quantization
qconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load Large LLM
tokenizer = AutoTokenizer.from_pretrained(NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(NAME, trust_remote_code=True, quantization_config=qconfig, use_cache=False, low_cpu_mem_usage=True)


model.save_pretrained("./model.tmp")
del model
model = AutoModelForCausalLM.from_pretrained("./model.tmp", trust_remote_code=True, quantization_config=qconfig, use_cache=False, low_cpu_mem_usage=True)


# Convert to PEFT
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=1,
    target_modules=["c_proj"],
    init_lora_weights="gaussian",
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

model.print_trainable_parameters()

# Saving
print(f"Saving model to {PATH}")
model.save_pretrained(PATH)
tokenizer.save_pretrained(PATH)
del model
model = AutoPeftModelForCausalLM.from_pretrained(PATH, trust_remote_code=True)
print(type(model))

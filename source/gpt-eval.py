from openai import OpenAI
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import re
import json 

from preprocess_for_plbart import detokenize_code
from my_params import parse_args
from my_data import load_data
from dataset_formators import select_mode


args = parse_args()

VARIABLE_NAME_PROMPT = "I am a system for generating variable names. I will accept a snippet of python code with \"<mask>\" as a replacement for one of the variable names. All mask tokens hide the same variable name. I will respond with three different suggestions for what the name should be. I will consider the context of the code they are found in to generate the most accourate results. I will separate the three variable names with a comma, My output will not contain anything else but the variable names."

DOCSTRING_NAME_PROMPT = "I am an AI for generating docstrings for Python 3 code. You need to preprend your code with \"$ DOCSTR code$\" where \"code\" specifies which docstring convenction you want me to generate. \"NP\": NumPy docstring, \"GO\": Google docstring, \"RE\": for reST docstring and \"NA\": for other/no formatting. I will output a fitting docstring describing the code you gave me. My outputs only contain the docstring text content. I can be used as an API."

COMMENT_PROMPT = "    I am an AI for generating inline code comments. Please add a marker of where you would like me to add a comment. I will not generate a short comment relevant to that part of the code and the particular line, I respond with nothing but the text content of the generated comment and will not contain any surrounding code or even the # symbol. I can be used as an API."

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
client = OpenAI()

modify = select_mode(args.mode)
val_dataset = load_data(args.dev_filename, tokenizer, args, modify)
val_dataset = val_dataset[:1000]
eval_dataloader = DataLoader(val_dataset, batch_size=1)

report = []

for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = batch['global_attention_mask']
    labels = batch['labels']
    hidden_index = batch['hidden_index']

    reference = [[tokenizer.decode(label, skip_special_tokens=True) for label in labels]]
    source = [tokenizer.decode(inp, skip_special_tokens=False) for inp in input_ids]
    source = detokenize_code(source).split("</s>")[0]

    source = re.sub("\\$ \\d+\\$", "# <insert comment here>\n", source)
    print(source)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "assistant",
            "content": COMMENT_PROMPT
        },
        {
            "role": "user",
            "content": f"{source}"
        }
    ],
    temperature=1,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print("source: ", source)

    try:
        # Check if there are choices in the response
        if response.choices:
            # Get the first choice (assuming there's only one completion choice in your request)
            message = response.choices[0].message.content
            if message:
                print("msg:", message)
                # message = [m.strip() for m in message.split(",")]
                # if len(message) != 3:
                #     continue
                report.append({ "label": reference[0], "prediction": message, "source": source})

            print("Message Content:", message)
        else:
            print("No choices available in the response.")
    except AttributeError as e:
        print("Failed to extract message content:", str(e))

with open("report-gpt.jsonl", 'w') as file:
    for entry in report:
        json.dump(entry, file)
        file.write('\n')

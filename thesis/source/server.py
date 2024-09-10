#!/usr/bin/env python3

import asyncio
import json
from aiohttp import web
from queue import Queue
import threading
import toml
import logging
import torch
import ast

from preprocess_for_plbart import tokenize_python, token_masking
from heuristic import lines_for_comment
from peft import AutoPeftModelForCausalLM
from dataset_formators import select_mode
from transformers import (
    LEDForConditionalGeneration,
    AutoTokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEVICE = 'cuda'
CONFIG = None
# The queue where transformed requests will be put
request_queue = Queue()

# The worker thread function to process items from the queue
def process_queue():
    while True:
        # Get an item from the queue
        item = request_queue.get()

        # Process the item
        print(f"Processing: {item}")

        # Mark the item as processed
        request_queue.task_done()


# Setup and start the worker thread
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

MODELS = None
TOKENIZERS = None

def init_models(CONFIG):
    global MODELS
    MODELS = {
        "docstring_generation": LEDForConditionalGeneration.from_pretrained(CONFIG["Comment"]["path_to_docstring_model"]).to(DEVICE),
        "comment_generation": LEDForConditionalGeneration.from_pretrained(CONFIG["Comment"]["path_to_comment_model"]).to(DEVICE),
        "rename": LEDForConditionalGeneration.from_pretrained(CONFIG["Rename"]["path_to_model"]).to(DEVICE),
    }
    if CONFIG["ErrorCorrection"]["enabled"]:
        MODELS["wizard"] = AutoPeftModelForCausalLM.from_pretrained(CONFIG["ErrorCorrection"]["path_to_model"]).to(DEVICE)

def init_tokenizers(CONFIG):
    global TOKENIZERS
    TOKENIZERS = {
        "docstring_generation": AutoTokenizer.from_pretrained(CONFIG["Comment"]["path_to_docstring_model"]),
        "comment_generation": AutoTokenizer.from_pretrained(CONFIG["Comment"]["path_to_comment_model"]),
        "rename": AutoTokenizer.from_pretrained(CONFIG["Rename"]["path_to_model"]),
        # AutoTokenizer.from_pretrained(CONFIG["Rename"]["path_to_model"]),
    }
    if CONFIG["ErrorCorrection"]["enabled"]:
        TOKENIZERS["wizard"] = AutoTokenizer.from_pretrained(CONFIG["ErrorCorrection"]["path_to_model"])


def get_model(name):
    model = MODELS[name]
    device = next(model.parameters()).device
    if device != DEVICE:
        model.to(DEVICE)
    return model

def has_docstring(code_snippet):
    try:
        # Parse the code snippet into an AST
        parsed_ast = ast.parse(code_snippet)

        # Check if the first statement in the function body is a docstring
        first_stmt = parsed_ast.body[0].body[0]
        if isinstance(first_stmt, ast.Expr):
            if isinstance(first_stmt.value, (ast.Str, ast.Constant)):
                logger.info("docstring found")
                return True
    except Exception as e:
        print(f"Error parsing code snippet: {e}")

    return False

def comment_module(snippet, comment_type):
    print("comment module")
    comments = []

    original_snippet = snippet
    snippet = ' '.join(tokenize_python(snippet))
    tokenizer = TOKENIZERS["comment_generation"]

    # Detection
    comments += lines_for_comment(original_snippet)

    if len(comments) == 0:
        return {}

    results = {}
    lines = [tokenizer.tokenize(line+'NEW_LINE') for line in snippet.split('NEW_LINE')]
    print(f'{len(comments)=}')

    if not has_docstring(snippet):
        comments.append(f"DOCSTR {comment_type}")

    print(f'{comment_type=}')
    for comment in comments:
        logger.info('Processing comment %s', comment)
        tokens_before_flag = 0
        # Generate individual comments
        flag = ['$'] + tokenizer.tokenize(f'{comment}') + ['$']
        source = []
        try:
            comment_int = int(comment)
        except ValueError:
            comment_int = 0
            source = flag[:]

        print(comment_int)
        for line in lines[:max(comment_int-1, 0)]:
            tokens_before_flag += len(line)
        for index, line in enumerate(lines):
            if index == comment_int-1:
                source += flag
            source += line
        source = source[:2048 -1] + ['__python__']
        global_attention_mask =\
            [True] * len(lines[0]) +\
            [False] * (tokens_before_flag - len(lines[0])) +\
            [True] * len(flag) +\
            [False] * (len(source) - len(lines[0]) - max(tokens_before_flag - len(lines[0]), 0) - len(flag) - 1) +\
            [True]
        global_attention_mask = torch.tensor(global_attention_mask).to(DEVICE)
        source = tokenizer.convert_tokens_to_ids(source)
        source = torch.tensor([source]).to(DEVICE)
        print(comment, type(comment))
        model = get_model("docstring_generation") if type(comment) is str else get_model("comment_generation")
        response = model.generate(
                input_ids=source, 
                global_attention_mask=global_attention_mask, 
                attention_mask=source.ne(tokenizer.pad_token_id),
                do_sample=True, 
                top_k=CONFIG["Comment"]["top_k"], 
                top_p=CONFIG["Comment"]["top_p"], 
                length_penalty=CONFIG["Comment"]["length_penalty"], 
                temperature=CONFIG["Comment"]["temperature"], 
                num_beams=CONFIG["Comment"]["beam_size"],
        )
        response = tokenizer.decode(response[0], skip_special_tokens=True)
        if type(comment) is str:
            response = response.replace("NEW_LINE", "\n")
        response = response.split('$')[-1]
        results[comment_int] = response

    print(f"final results {results}")
    return results

def rename_module(original_snippet, symbol):
    snippet, _ = token_masking(original_snippet, symbol)
    print(snippet)
    print(symbol)
    print(_)
    snippet = ' '.join(tokenize_python(snippet))
    tokenizer = TOKENIZERS["rename"]
    source, _, global_attention_mask, _ = select_mode("rename")(snippet, symbol, tokenizer, len(snippet), 16)[0]
    for id, mask in zip(source, global_attention_mask):
        print(f'{id}: {mask}, ', end='')
    source = torch.tensor([tokenizer.convert_tokens_to_ids(source)]).to(DEVICE)
    global_attention_mask = torch.tensor(global_attention_mask).to(DEVICE)
    response = get_model("rename").generate(
            input_ids=source, 
            global_attention_mask=global_attention_mask, 
            attention_mask=source.ne(tokenizer.pad_token_id),
            num_return_sequences=3, 
            do_sample=False, 
            num_beams=CONFIG["Rename"]["beam_size"] * 3,
            diversity_penalty=CONFIG["Rename"]["diversity_penalty"],
            num_beam_groups=CONFIG["Rename"]["beam_size"], 
            length_penalty=CONFIG["Rename"]["length_penalty"], 
            temperature=CONFIG["Rename"]["temperature"], 
    )
    response = [tokenizer.decode(out, skip_special_tokens=True) for out in response]
    response = [{"symbol": s, "code": original_snippet} for s in response]
    logger.info(response)
    return response

def error_module(snippet):
    snippet = ' '.join(tokenize_python(snippet))
    tokenizer = TOKENIZERS["error"]
    source, _, global_attention_mask, _ = select_mode("wizard")(snippet, "<sep>", tokenizer, 2048, 2048)[0]
    source = torch.tensor([tokenizer.convert_tokens_to_ids(source)]).to(DEVICE)
    response = get_model("wizard").generate(
            input_ids=source, 
            max_length=2048,
            top_k=CONFIG["ErrorCorrection"]["top_k"], 
            top_p=CONFIG["ErrorCorrection"]["top_p"], 
            length_penalty=CONFIG["ErrorCorrection"]["length_penalty"], 
            temperature=CONFIG["ErrorCorrection"]["temperature"], 
            num_beams=CONFIG["ErrorCorrection"]["beam_size"],
    )
    response = tokenizer.decode(response[0], skip_special_tokens=True)
    response = response.split("### Fixed Code: ", 1)[1]
    logger.info(response)
    return response

def format_response(uuid, response):
    dict_response = [{"id": id, "task": task, "status": status, "result": result} for id, task, status, result in response]
    return json.dumps({"uuid": uuid, 'response': dict_response})+"\n"

def rename_variable_preserving_format(code, old_name, new_name):
    class RenameVariable(ast.NodeTransformer):

        def visit_arg(self, node):
            if node.arg == old_name:
                node.arg = new_name
            return node

        def visit_Name(self, node):
            if node.id == old_name:
                return ast.copy_location(ast.Name(id=new_name, ctx=node.ctx), node)
            return node

    tree = ast.parse(code)
    RenameVariable().visit(tree)

    lines = code.splitlines(keepends=True)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id == new_name) or (isinstance(node, ast.arg) and node.arg == new_name):
            start_line, start_column = node.lineno - 1, node.col_offset
            end_column = start_column + len(old_name)
            line = lines[start_line]
            lines[start_line] = line[:start_column] + new_name + line[end_column:]

    return ''.join(lines)

# The asynchronous handler for incoming requests
async def handle_request(request):
    response = []
    json_data = await request.json()
    uuid = json_data["uuid"]
    for request in json_data["requests"]:
        for task in request["tasks"]:
            if task["task"] == "comment":
                result = comment_module(request["snippet"], task["type"])
                response.append((request["id"], "comment", "ok", result))
            if task["task"] == "error":
                result = error_module(request["snippet"])
                response.append((request["id"], "error", "ok", result))
            if task["task"] == "rename":
                result = rename_module(request["snippet"], task["symbol"])
                for entry in result:
                    entry["code"] = rename_variable_preserving_format(entry["code"], task["symbol"], entry["symbol"])
                response.append((request["id"], "rename", "ok", result))

    json_response = format_response(uuid, response)
    print(json_response)
    return web.Response(text=json_response, content_type='application/json')

# The main function to start the aiohttp server
async def main():
    global CONFIG
    global MODELS
    print('Starting program')
    with open('config.toml', 'r') as f:
        CONFIG = toml.load(f)
        conf_server = CONFIG['Server']

    init_models(CONFIG)
    init_tokenizers(CONFIG)

    app = web.Application()
    app.router.add_post('/', handle_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '::', conf_server["port"])
    await site.start()

    # Run forever
    while True:
        await asyncio.sleep(1)

# Start the event loop and the server
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

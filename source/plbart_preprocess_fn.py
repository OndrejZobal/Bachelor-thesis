from preprocess_for_plbart import *

import re
import json
import subprocess

from tqdm import tqdm

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import ast
import astor
import argparse
import logging
import os
import random
import re
import sys
import io
import tokenize
import argparse

from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    default="",
    type=str,
    help="The input file",
)
parser.add_argument(
    "--mask",
    default="false",
    type=str,
    help="Mask a random identifier: true/false",
)
parser.add_argument(
    "--allow_empty",
    default="true",
    type=str,
    # help="Mask a random identifier",
)
parser.add_argument(
    "--codexglue",
    default="true",
    type=str,
    help="Whether the dataset format is the same as codeXglue: true/false",
)
parser.add_argument(
    "--ln_field",
    default="description",
    type=str,
    help="Key for the description fields",
)
parser.add_argument(
    "--close_traceback",
    default="false",
    type=str,
    help="Filter Tracebacks with farway code.",
)
parser.add_argument(
    "--traceback",
    default="false",
    type=str,
    help="Include tracebacks.",
)
parser.add_argument(
    "--tokenize",
    default="true",
    type=str,
    help="Tokenize code for PLBART: true/false",
)
args = parser.parse_args()
args.mask = args.mask.lower() == "true"
args.codexglue = args.codexglue.lower() == "true"
args.allow_empty = args.allow_empty.lower() == "true"
args.close_traceback = args.close_traceback.lower() == "true"
args.traceback = args.traceback.lower() == "true"
args.tokenzie = args.tokenize.lower() == "true"

MASK = args.mask
ALLOW_EMPTY = args.allow_empty
USING_CODEXGLUE = args.codexglue
TRACEBACK_INFORMATION = args.traceback
CODE_FIELD_TYPE = 'code' if USING_CODEXGLUE else 'input'
OUTPUT_FIELD_TYPE = 'docstring' if USING_CODEXGLUE else 'output'
NL_FIELD_TYPE = args.ln_field
CLOSE_TRACEBACK = args.close_traceback
TOKENIZE = args.tokenize

print(TOKENIZE)
logging.basicConfig(level=logging.INFO)

def prepare(file_name):
    with open(file_name, 'r', encoding='utf-8') as source, open(f'plbart-{file_name}', 'w', encoding='utf-8') as out_file:
        for line in tqdm(source, total=count_file_lines(file_name)):
            ex = json.loads(line.strip())
            code = ''.join(ex[CODE_FIELD_TYPE])
            code_stripped = re.sub("[\n\r\t ]+", " ", code).strip()
            output = ex[OUTPUT_FIELD_TYPE].strip()
            #output = re.sub("[\n\r\t ]+", " ", output).strip()
            if len(code_stripped) == 0 or len(output) == 0 and not ALLOW_EMPTY:
                continue

            if MASK:
                try:
                    code, output = token_masking(code)
                    if output == None:
                        continue
                except SyntaxError:
                    continue
                
            if TOKENIZE:
                _tokens = tokenize_python(code, keep_comments=True if USING_CODEXGLUE else False, remove_docstrings=True)
                if _tokens == None and not ALLOW_EMPTY: continue
                tokenized_code = ' '.join(_tokens)
                tokenized_code = re.sub("[\n\r\t ]+", " ", tokenized_code).strip()
            else:
                tokenized_code = code

            if CLOSE_TRACEBACK:
                # Split the traceback into lines and reverse the order
                lines = ex["traceback"].strip().split('\n')[::-1]
                
                # Regular expression to match lines in a traceback that contain a function call
                pattern = r'File "[^"]+", line \d+, in (\w+)'

                # Walk through each line from the bottom up
                match_traceback = None
                match_fn = None

                for line in lines:
                    match_traceback = re.search(pattern, line)
                    if match_traceback:
                        break

                if not match_traceback:
                    continue

                # Regular expression to match the function name in a Python function definition
                pattern = r'def (\w+)\s*\('
                
                # Search for the pattern in the code snippet
                match_fn = re.search(pattern, ex[CODE_FIELD_TYPE])

                if not match_fn:
                    continue

                if match_traceback.group(1) != match_fn.group(1):
                    continue


            if not USING_CODEXGLUE:
                if TOKENIZE:
                    output = tokenize_python(ex[OUTPUT_FIELD_TYPE], keep_comments=True if USING_CODEXGLUE else False, remove_docstrings=True)
                    if output == None and not ALLOW_EMPTY: continue
                    output = ' '.join(output)
                    output = re.sub("[\n\r\t ]+", " ", output).strip()
                else:
                    output = ex[OUTPUT_FIELD_TYPE]

                nl = ex[NL_FIELD_TYPE]

                if output == tokenized_code and not ALLOW_EMPTY: continue
                output = nl + "<sep>" + output
            #del ex[CODE_FIELD_TYPE]
            #del ex[OUTPUT_FIELD_TYPE]
            ex['input'] = tokenized_code
            ex['output'] = output
            if len(tokenized_code) == 0 and not ALLOW_EMPTY:
                continue
            try:
                # this line can throw error UnicodeEncodeError
                out_file.write(json.dumps(ex) + '\n')
            except:
                print("error writing line")


if __name__ == '__main__':
    random.seed(42)
    prepare(args.file)


#!/bin/sh

mask=true
allow_empty=false
codexglue=true
traceback=false

python3 plbart_preprocess_fn.py \
    --file "$1"  \
    --mask "$mask" \
    --allow_empty "$allow_empty" \
    --codexglue "$codexglue" \
    --traceback "$traceback" \


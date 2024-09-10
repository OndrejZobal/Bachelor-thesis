#!/bin/sh

mask=false
allow_empty=false
codexglue=true
traceback=false

python3 led/plbart_preprocess_fn.py \
    --file "$1"  \
    --mask "$mask" \
    --allow_empty "$allow_empty" \
    --codexglue "$codexglue" \
    --traceback "$traceback" \


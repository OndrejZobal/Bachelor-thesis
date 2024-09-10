#!/bin/sh

mask=false
allow_empty=false
codexglue=false
ln_field="description"
close_traceback=true

python3 plbart_preprocess_fn.py \
    --file "$1"  \
    --mask "$mask" \
    --allow_empty "$allow_empty" \
    --codexglue "$codexglue" \
    --ln_field "$ln_field" \
    --close_traceback "$close_traceback"


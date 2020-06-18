#!/bin/env bash

# Run all.
python main.py \
--inp_file examples/examples.jsonl \
--tree_key tree \
--prefix demo_run_all \
--out_dir examples/out

# Run some.
python main.py \
--inp_file examples/examples.jsonl \
--tree_ids ptb01094,ptb02218 \
--tree_key tree \
--prefix demo_run_some \
--out_dir examples/out

# Can emphasize edges and text.
python main.py \
--inp_file examples/examples.jsonl \
--tree_ids ptb01035 \
--tree_key tree \
--prefix demo_style \
--out_dir examples/out

# Stitch together.
python main.py \
--inp_file examples/examples.jsonl,examples/examples-2.jsonl \
--tree_ids ptb01035 \
--tree_key tree,tree \
--prefix demo_stitch_1,demo_stitch_2 \
--out_dir examples/out

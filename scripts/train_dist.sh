#!/usr/bin/env sh

python -m torch.distributed.launch --nproc_per_node=$1 run_apis/train_dist.py \
    --launcher pytorch \
    --report_freq 400 \
    --data_path $2 \
    --port 23333 \
    --config $3

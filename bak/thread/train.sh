#!/bin/bash
clear
source activate rebel

nohup python run.py --adhoc --cfg conf/c02_selfplay/hunl_sp.yaml \
    env.max_raise_times=6 \
    env.stack_size=800 \
    env.subgame_params.use_cfr=true \
    env.subgame_params.num_iters=1024 \
    env.subgame_params.max_depth=5 \
    selfplay.threads_per_gpu=64 \
    selfplay.cpu_gen_threads=0 \
    data.train_epoch_size=2048 \
    data.train_batch_size=32 \
    data.val_epoch_size=256 \
    replay.prefetch=0 \
    > train.log 2>&1 &
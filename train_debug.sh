#!/bin/bash
clear
source activate rebel
nohup python3 run.py --adhoc --cfg conf/c02_selfplay/hunl_sp_debug.yaml \
    env.max_raise_times=6 \
    env.stack_size=800 \
    env.subgame_params.use_cfr=true \
    env.subgame_params.num_iters=32 \
    env.subgame_params.max_depth=5 \
    selfplay.cpu_gen_processes=64 \
    selfplay.models_per_device=1 \
    data.train_epoch_size=2048 \
    data.train_batch_size=32 \
    data.val_epoch_size=256 \
    replay.prefetch=0 \
    > train_debug.log 2>&1 &

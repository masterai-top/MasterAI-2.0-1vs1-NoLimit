#!/bin/bash
clear
source activate rebel
nohup python3 run.py --adhoc --cfg conf/c02_selfplay/hunl_sp.yaml \
    env.max_raise_times=6 \
    env.stack_size=800 \
    env.subgame_params.use_cfr=true \
    env.subgame_params.num_iters=128 \
    env.subgame_params.max_depth=5 \
    selfplay.cpu_gen_processes=72 \
    selfplay.processes_per_gpu=-1 \
    selfplay.models_per_device=1 \
    data.train_epoch_size=4096 \
    data.train_batch_size=512 \
    data.val_epoch_size=1024 \
    replay.prefetch=0 \
    > train.log 2>&1 &

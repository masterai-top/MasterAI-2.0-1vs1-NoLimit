#!/bin/bash
source activate rebel
#nohup python3 main.py 8888 > log_port_8888.log 2>&1 &
nohup python3 main.py 9999 > log_port_9999.log 2>&1 &

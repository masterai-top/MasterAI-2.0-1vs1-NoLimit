#!/bin/bash
server_port=9999
source activate rebel
nohup python3 main.py ${server_port} >chaos_brain_${server_port}.log 2>&1 &

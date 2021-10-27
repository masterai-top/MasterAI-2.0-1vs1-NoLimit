#!/bin/bash
ps -def | grep "python3 main.py" | grep -v grep | cut -c 9-15 | xargs kill -9
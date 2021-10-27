#!/bin/bash
rm -f *.spec
rm -rf ./dist
rm -rf ./build
source activate chaos_brain
pyinstaller -F main.py

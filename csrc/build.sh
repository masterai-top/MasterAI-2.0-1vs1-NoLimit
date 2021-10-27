#!/bin/bash
rm -rf build

mkdir build
cd build

source activate chaos_brain
cmake ..
make

cp mc*.so ../../robot/gto_mc
cp ../config/lookup_tablev3.bin ../../robot/gto_mc

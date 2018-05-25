#!/bin/bash
#$ -q meta_core.q
#$ -N ml4aad-benchmark-10
#$ -m ea
#$ -v PYTHONPATH=:/home/arabim/mllab/venv/bin
#$ -M arabim@informatik.uni-freiburg.de
timeout 172800 /home/arabim/mllab/venv/bin/python /home/arabim/ML4AAD/ex-02/ex02_4/benchmark.py \
    -p /home/arabim/mllab/venv/bin \
    -w /home/arabim/ML4AAD/ex-02/ex02_4/mllab \
    -f /home/arabim/mllab/cluster/rawAllx1000.json \
    -s 1 2 4 8 16 32 64 128 256 512 1024 \
    -i 1 \
    -I 10

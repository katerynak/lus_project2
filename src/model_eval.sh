#!/usr/bin/env bash

python3 train.py --model bilstm --trainfile ../data/NLSPARQL.train.data --devfile ../data/NLSPARQL.test.data --batch 20 --lr 0.001

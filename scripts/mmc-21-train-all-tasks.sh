#!/usr/bin/env bash

model=$1

for exp in data/processed/mmc_*; do
    python3 covidecg/train.py --model $model $exp
done

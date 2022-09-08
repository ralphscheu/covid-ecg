#!/usr/bin/env bash

python3 covidecg/train.py \
    --exp-config exp_configs/exp-postcovid-ctrl-img.yaml \
    --model-config exp_configs/model-cnnseqpool.yml

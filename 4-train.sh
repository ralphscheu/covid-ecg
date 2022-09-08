#!/usr/bin/env bash

python3 covidecg/train.py \
    --exp-config conf/exp-postcovid-ctrl-img.yaml \
    --model-config conf/model-cnnseqpool.yml

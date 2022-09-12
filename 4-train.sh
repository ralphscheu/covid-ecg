#!/usr/bin/env bash

# python3 covidecg/train.py \
#     --exp-config conf/exp-postcovid-ctrl-img.yaml \
#     --model-config conf/model-cnnseqpool.yml

python3 covidecg/train.py \
    --model-config conf/model-cnnseqlstm.yml \
    data/processed/mmc_recs_10s_postcovid_vs_ctrl

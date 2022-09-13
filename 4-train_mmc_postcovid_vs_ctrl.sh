#!/usr/bin/env bash


python3 covidecg/train.py --model-config conf/model-cnnseqpool.yml data/processed/mmc_recs_postcovid_vs_ctrl

python3 covidecg/train.py --model-config conf/model-cnnseqlstm.yml data/processed/mmc_recs_postcovid_vs_ctrl



python3 covidecg/train.py --model-config conf/model-cnnseqpool.yml data/processed/mmc_10s_recs_postcovid_vs_ctrl

python3 covidecg/train.py --model-config conf/model-cnnseqlstm.yml data/processed/mmc_10s_recs_postcovid_vs_ctrl
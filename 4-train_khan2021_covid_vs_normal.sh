#!/usr/bin/env bash


python3 covidecg/train.py --model-config conf/model-cnnseqpool.yml data/processed/khan2021_covid_vs_normal

python3 covidecg/train.py --model-config conf/model-cnnseqlstm.yml data/processed/khan2021_covid_vs_normal

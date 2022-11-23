#!/usr/bin/env bash


#####################
### GENERATE ECG IMAGES
#####################

rm -rf data/interim/mmc_feats_img/covid/*
python3 covidecg/data/mmc_generate_feats_img.py \
    data/interim/mmc_feats_raw/covid \
    data/interim/mmc_feats_img/covid


rm -rf data/interim/mmc_feats_img/ctrl/*
python3 covidecg/data/mmc_generate_feats_img.py \
    data/interim/mmc_feats_raw/ctrl \
    data/interim/mmc_feats_img/ctrl


rm -rf data/interim/mmc_feats_img/postcovid/*
python3 covidecg/data/mmc_generate_feats_img.py \
    data/interim/mmc_feats_raw/postcovid \
    data/interim/mmc_feats_img/postcovid

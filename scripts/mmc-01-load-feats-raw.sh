#!/usr/bin/env bash


#####################
### LOAD RAW RECORDINGS
#####################


# Extract all recordings for covid patient group
rm -rf data/interim/mmc_feats_raw/covid/*
python3 covidecg/data/mmc_generate_feats_raw.py \
    --prefix covid \
    data/raw/mmc_xml_covid/ data/interim/mmc_feats_raw/covid


# Extract all recordings for control group
rm -rf data/interim/mmc_feats_raw/ctrl/*
python3 covidecg/data/mmc_generate_feats_raw.py \
    --prefix ctrl \
    data/raw/mmc_xml_ctrl data/interim/mmc_feats_raw/ctrl


# Extract all recordings for postcovid patient group
rm -rf data/interim/mmc_feats_raw/postcovid/*
python3 covidecg/data/mmc_generate_feats_raw.py \
    --prefix postcovid \
    data/raw/mmc_xml_postcovid/ data/interim/mmc_feats_raw/postcovid
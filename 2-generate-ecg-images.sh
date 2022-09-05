#!/usr/bin/env bash

rm -rf data/processed/mmc_recs_postcovid
mkdir data/processed/mmc_recs_postcovid
python3 create_images.py --img-height 200 \
    --recordings-file data/interim/mmc_recs_stress_postcovid.csv \
    --recordings-dir data/interim/mmc_recs \
    --output-dir data/processed/mmc_recs_postcovid

rm -rf data/processed/mmc_recs_ctrl
mkdir data/processed/mmc_recs_ctrl
python3 create_images.py --img-height 200 \
    --recordings-file data/interim/mmc_recs_stress_ctrl.csv \
    --recordings-dir data/interim/mmc_recs \
    --output-dir data/processed/mmc_recs_ctrl

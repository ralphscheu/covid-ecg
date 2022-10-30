#!/usr/bin/env bash


###
### Generate train/test sets for each task
###

# Generate train/test sets for mmc_postcovid_mmc_ctrl
rm -rf data/processed/mmc_postcovid_mmc_ctrl
python3 covidecg/data/split_train_test.py --test-ratio 0.2 data/interim/mmc data/processed/mmc_postcovid_mmc_ctrl

# Generate train/test sets for mmc_10s_postcovid_mmc_10s_ctrl
rm -rf data/processed/mmc_10s_postcovid_mmc_10s_ctrl
python3 covidecg/data/split_train_test.py --test-ratio 0.2 data/interim/mmc_10s data/processed/mmc_10s_postcovid_mmc_10s_ctrl

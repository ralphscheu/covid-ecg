#!/usr/bin/env bash

###
### Create task datasets in data/interim via symlinks
###

# mmc covid vs. ctrl
rm -rf data/interim/mmc_tasks_raw/mmc_covid_vs_ctrl/*
mkdir data/interim/mmc_tasks_raw/mmc_covid_vs_ctrl/1_covid
mkdir data/interim/mmc_tasks_raw/mmc_covid_vs_ctrl/0_ctrl
ln -s `pwd`/data/interim/mmc_feats_raw/covid/*.png `pwd`/data/interim/mmc_tasks_raw/mmc_covid_vs_ctrl/1_covid/
ln -s `pwd`/data/interim/mmc_feats_raw/ctrl/*.png `pwd`/data/interim/mmc_tasks_raw/mmc_covid_vs_ctrl/0_ctrl/




###
### Generate train/test sets for each task
###

# mmc covid vs. ctrl
rm -rf data/processed/mmc_tasks_raw/mmc_covid_vs_ctrl/*
python3 covidecg/data/split_train_test.py --test-ratio 0.2 data/interim/mmc_tasks_raw/mmc_covid_vs_ctrl data/processed/mmc_tasks_raw/mmc_covid_vs_ctrl

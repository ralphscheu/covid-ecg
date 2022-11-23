 #!/usr/bin/env bash


#####################
### GENERATE ECG IMAGES
#####################

rm -rf data/interim/mmc_feats_lfcc/covid/*
python3 covidecg/data/mmc_generate_feats_lfcc.py ./data/interim/mmc_feats_raw/covid ./data/interim/mmc_feats_lfcc/covid


rm -rf data/interim/mmc_feats_lfcc/ctrl/*
python3 covidecg/data/mmc_generate_feats_lfcc.py ./data/interim/mmc_feats_raw/ctrl ./data/interim/mmc_feats_lfcc/ctrl


rm -rf data/interim/mmc_feats_lfcc/postcovid/*
python3 covidecg/data/mmc_generate_feats_lfcc.py ./data/interim/mmc_feats_raw/postcovid ./data/interim/mmc_feats_lfcc/postcovid

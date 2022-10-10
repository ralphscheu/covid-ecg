#!/usr/bin/env bash


# #####################
# ### LOAD RAW RECORDINGS
# #####################

# Extract all recordings for postcovid patient group
# rm -rf data/interim/mmc_csv/postcovid
# python3 covidecg/data/extract_recordings.py \
#     --prefix postcovid \
#     --patients-list data/raw/patients_postcovid.csv \
#     data/raw/ecg_export_postcovid data/interim/mmc_csv/postcovid

# # Extract all recordings for control group
# rm -rf data/interim/mmc_csv/ctrl
# python3 covidecg/data/extract_recordings.py \
#     --prefix ctrl \
#     --patients-list data/raw/patients_ctrl.csv \
#     data/raw/ecg_export_ctrl data/interim/mmc_csv/ctrl





# #####################
# ### GENERATE ECG IMAGES
# #####################

# rm -rf data/interim/mmc/postcovid
# python3 covidecg/data/create_images.py --img-height 100 \
#     --recordings-file data/interim/mmc_stress_postcovid.csv \
#     --recordings-dir data/interim/mmc_csv/postcovid \
#     --output-dir data/interim/mmc/postcovid

# rm -rf data/interim/mmc/ctrl
# python3 covidecg/data/create_images.py --img-height 100 \
#     --recordings-file data/interim/mmc_stress_ctrl.csv \
#     --recordings-dir data/interim/mmc_csv/ctrl \
#     --output-dir data/interim/mmc/ctrl

# rm -rf data/interim/mmc_10s
# python covidecg/data/filter_and_symlink.py --min-length 5000 --max-length 5000 data/interim/mmc/ctrl data/interim/mmc_10s/ctrl
# python covidecg/data/filter_and_symlink.py --min-length 5000 --max-length 5000 data/interim/mmc/postcovid data/interim/mmc_10s/postcovid



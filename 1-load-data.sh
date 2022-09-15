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

rm -rf data/interim/mmc/postcovid
python3 covidecg/data/create_images.py --img-height 100 \
    --recordings-file data/interim/mmc_stress_postcovid.csv \
    --recordings-dir data/interim/mmc_csv/postcovid \
    --output-dir data/interim/mmc/postcovid

rm -rf data/interim/mmc/ctrl
python3 covidecg/data/create_images.py --img-height 100 \
    --recordings-file data/interim/mmc_stress_ctrl.csv \
    --recordings-dir data/interim/mmc_csv/ctrl \
    --output-dir data/interim/mmc/ctrl

rm -rf data/interim/mmc_10s
python covidecg/data/filter_and_symlink.py --min-length 5000 --max-length 5000 data/interim/mmc/ctrl data/interim/mmc_10s/ctrl
python covidecg/data/filter_and_symlink.py --min-length 5000 --max-length 5000 data/interim/mmc/postcovid data/interim/mmc_10s/postcovid





# #####################
# ### CONCATENATE RECORDINGS TO SESSIONS
# #####################

# # python3 covidecg/data/concat_recordings_per_session.py \
# #     --recordings-list data/interim/recordings_stress_ecg_postcovid.csv \
# #     data/interim/mmc/postcovid data/interim/mmc_sessions_postcovid

# # python3 covidecg/data/concat_recordings_per_session.py \
# #     --recordings-list data/interim/recordings_stress_ecg_ctrl.csv \
# #     data/interim/mmc/ctrl data/interim/mmc_sessions_ctrl





# ##########################
# ### LOAD KHAN2021 DATA ###
# ##########################

# rm -rf t-dir data/interim/khan2021/normal
# for file in ./data/external/Normal\ Person\ ECG\ Images\ \(859\)/Normal*.jpg; do
#     echo "Processing ${file}"
#     python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
#         --output-dir data/interim/khan2021/normal \
#         --input-layout ecgsheet \
#         "${file}"
# done

# rm -rf data/interim/khan2021/covid
# for file in ./data/external/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/Binder*.jpg; do
#     echo "Processing ${file}"
# 	python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
# 		--output-dir data/interim/khan2021/covid \
# 		--input-layout binder \
#         "${file}"
# done
# for file in ./data/external/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/COVID*.jpg; do
#     echo "Processing ${file}"
# 	python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
# 		--output-dir data/interim/khan2021/covid \
# 		--input-layout ecgsheet \
#         "${file}"
# done





# Generate train/test sets for mmc_postcovid_vs_ctrl
# rm -rf data/processed/mmc_postcovid_vs_ctrl
# splitfolders --output data/processed/mmc_postcovid_vs_ctrl --ratio .8 .2 -- data/interim/mmc

# # Generate train/test sets for mmc_10s_postcovid_vs_ctrl
# rm -rf data/processed/mmc_10s_postcovid_vs_ctrl
# splitfolders --output data/processed/mmc_10s_postcovid_vs_ctrl --ratio .8 .2 -- data/interim/mmc_10s




# # Generate train/test sets for khan2021_covid_vs_normal
# rm -rf data/processed/khan2021_covid_vs_normal
# splitfolders --output data/processed/khan2021_covid_vs_normal --ratio .8 .2 -- data/interim/khan2021


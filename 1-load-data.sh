#!/usr/bin/env bash

#####################
### LOAD MMC DATA ###
#####################

# extract all recordings for postcovid patient group
rm -rf data/interim/mmc_recs_postcovid
mkdir data/interim/mmc_recs_postcovid
python3 covidecg/data/extract_recordings.py \
    --prefix postcovid \
    --patients-list data/raw/patients_postcovid.csv \
    data/raw/ecg_export_postcovid data/interim/mmc_recs_postcovid

# extract all recordings for control group
rm -rf data/interim/mmc_recs_ctrl
mkdir data/interim/mmc_recs_ctrl
python3 covidecg/data/extract_recordings.py \
    --prefix ctrl \
    --patients-list data/raw/patients_ctrl.csv \
    data/raw/ecg_export_ctrl data/interim/mmc_recs_ctrl


# merge mmc recordings directories
rm -rf data/interim/mmc_recs
mkdir data/interim/mmc_recs
ln -s `pwd`/data/interim/mmc_recs_postcovid/* `pwd`/data/interim/mmc_recs/
ln -s `pwd`/data/interim/mmc_recs_ctrl/* `pwd`/data/interim/mmc_recs/


# concatenate recordings of the same session together
python3 covidecg/data/concat_recordings_per_session.py \
    --recordings-list data/interim/recordings_stress_ecg_postcovid.csv \
    data/interim/mmc_recs_postcovid data/interim/mmc_sessions_postcovid

python3 covidecg/data/concat_recordings_per_session.py \
    --recordings-list data/interim/recordings_stress_ecg_ctrl.csv \
    data/interim/mmc_recs_ctrl data/interim/mmc_sessions_ctrl

# merge mmc sessions directories
rm -rf data/interim/mmc_sessions
mkdir data/interim/mmc_sessions
ln -s `pwd`/data/interim/mmc_sessions_postcovid/* `pwd`/data/interim/mmc_sessions/
ln -s `pwd`/data/interim/mmc_sessions_ctrl/* `pwd`/data/interim/mmc_sessions/


##########################
### LOAD KHAN2021 DATA ###
##########################

rm -rf t-dir data/processed/khan2021_normal
mkdir data/processed/khan2021_normal
for file in ./data/external/Normal\ Person\ ECG\ Images\ \(859\)/Normal*.jpg; do
    echo "Processing ${file}"
    python3 load_khan2021_dataset.py --img-height 200 \
        --output-dir data/processed/khan2021_normal \
        --input-layout ecgsheet \
        "${file}"
done

rm -rf t-dir data/processed/khan2021_covid
mkdir data/processed/khan2021_covid
for file in ./data/external/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/Binder*.jpg; do
    echo "Processing ${file}"
	python3 load_khan2021_dataset.py --img-height 200 \
		--output-dir data/processed/khan2021_covid \
		--input-layout binder \
        "${file}"
done

# TODO check whether ecgsheet coordinates from "Normal" dataset also work for COVID samples
for file in ./data/external/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/COVID*.jpg; do
    echo "Processing ${file}"
	python3 load_khan2021_dataset.py --img-height 200 \
		--output-dir data/processed/khan2021_covid \
		--input-layout ecgsheet \
        "${file}"
done

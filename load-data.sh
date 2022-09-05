#!/usr/bin/env bash


khan2021_normal_outdir=./data/processed/khan2021_normal

rm -rf t-dir $khan2021_normal_outdir
mkdir $khan2021_normal_outdir
for file in ./data/external/Normal\ Person\ ECG\ Images\ \(859\)/Normal*.jpg; do
    echo "Processing ${file}"
    python3 load_khan2021_dataset.py --img-height 200 \
        --output-dir $khan2021_normal_outdir \
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

#!/usr/bin/env bash



rm -rf data/interim/khan_feats_img/normal/*
# NORMAL
for file in ./data/external/khan/Normal\ Person\ ECG\ Images\ \(859\)/*.jpg; do
    echo "Processing ${file}"
    python3 covidecg/data/khan_generate_feats_img.py \
        --output-dir data/interim/khan_feats_img/normal \
        --input-layout ecgsheet \
        "${file}"
done




rm -rf data/interim/khan_feats_img/covid/*
# COVID Binder scans
for file in ./data/external/khan/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/Binder*.jpg; do
    echo "Processing ${file}"
	python3 covidecg/data/khan_generate_feats_img.py \
		--output-dir data/interim/khan_feats_img/covid \
		--input-layout binder \
        "${file}"
done
# COVID ECG Printout scans
for file in ./data/external/khan/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/COVID*.jpg; do
    echo "Processing ${file}"
	python3 covidecg/data/khan_generate_feats_img.py \
		--output-dir data/interim/khan_feats_img/covid \
		--input-layout ecgsheet \
        "${file}"
done




rm -rf data/interim/khan_feats_img/abnormal/*
# ABNORMAL HEARTBEAT
for file in ./data/external/khan/ECG\ Images\ of\ Patient\ that\ have\ abnormal\ heart\ beats\ \(548\)/*.jpg; do
    echo "Processing ${file}"
    python3 covidecg/data/khan_generate_feats_img.py \
        --output-dir data/interim/khan_feats_img/abnormal \
        --input-layout ecgsheet \
        "${file}"
done
# ABNORMAL HEARTBEAT ecgsheet2
for file in ./data/external/khan/ECG\ Images\ of\ Patient\ that\ have\ abnormal\ heart\ beats\ \(548\)\ -\ ecgsheet2/*.jpg; do
    echo "Processing ${file}"
    python3 covidecg/data/khan_generate_feats_img.py \
        --output-dir data/interim/khan_feats_img/abnormal \
        --input-layout ecgsheet2 \
        "${file}"
done


#!/usr/bin/env bash



rm -rf data/interim/khan_feats_imgsharpen/normal/*
# NORMAL
for file in ./data/external/khan/Normal\ Person\ ECG\ Images\ \(859\)/*.jpg; do
	python3 covidecg/data/khan_generate_feats_imgsharpen.py \
		--output-dir data/interim/khan_feats_imgsharpen/normal \
		--input-layout ecgsheet \
        "${file}"
done





rm -rf data/interim/khan_feats_imgsharpen/covid/*
# COVID Binder scans
for file in ./data/external/khan/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/Binder*.jpg; do
	python3 covidecg/data/khan_generate_feats_imgsharpen.py \
		--output-dir data/interim/khan_feats_imgsharpen/covid \
		--input-layout binder \
        "${file}"
done
# COVID ECG Printout scans
for file in ./data/external/khan/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/COVID*.jpg; do
	python3 covidecg/data/khan_generate_feats_imgsharpen.py \
		--output-dir data/interim/khan_feats_imgsharpen/covid \
		--input-layout ecgsheet \
        "${file}"
done




rm -rf data/interim/khan_feats_imgsharpen/abnormal/*
# ABNORMAL HEARTBEAT
for file in ./data/external/khan/ECG\ Images\ of\ Patient\ that\ have\ abnormal\ heart\ beats\ \(548\)/*.jpg; do
    python3 covidecg/data/khan_generate_feats_imgsharpen.py \
        --output-dir data/interim/khan_feats_imgsharpen/abnormal \
        --input-layout ecgsheet \
        "${file}"
done
# ABNORMAL HEARTBEAT ecgsheet2
for file in ./data/external/khan/ECG\ Images\ of\ Patient\ that\ have\ abnormal\ heart\ beats\ \(548\)\ -\ ecgsheet2/*.jpg; do
    python3 covidecg/data/khan_generate_feats_imgsharpen.py \
        --output-dir data/interim/khan_feats_imgsharpen/abnormal \
        --input-layout ecgsheet2 \
        "${file}"
done



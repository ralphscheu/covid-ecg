#!/usr/bin/env bash



rm -rf data/interim/khan_feats_imgblur/normal/*
# NORMAL
for file in ./data/interim/khan_feats_img/normal/*.png; do
    python3 covidecg/data/khan_generate_feats_imgblur.py --output-dir data/interim/khan_feats_imgblur/normal "${file}"
done




rm -rf data/interim/khan_feats_imgblur/covid/*
# COVID
for file in ./data/interim/khan_feats_img/covid/*.png; do
	python3 covidecg/data/khan_generate_feats_imgblur.py --output-dir data/interim/khan_feats_imgblur/covid "${file}"
done




rm -rf data/interim/khan_feats_imgblur/abnormal/*
# ABNORMAL HEARTBEAT
for file in ./data/interim/khan_feats_img/abnormal/*.png; do
    python3 covidecg/data/khan_generate_feats_imgblur.py --output-dir data/interim/khan_feats_imgblur/abnormal "${file}"
done

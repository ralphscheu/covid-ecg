#!/usr/bin/env bash



rm -rf ./data/interim/khan_feats_imgsuperres/normal/*
# NORMAL
for file in ./data/interim/khan_feats_imgsuperres_4x/normal/*.png; do
    outFile="./data/interim/khan_feats_imgsuperres/normal/$(basename "${file}")"
    echo "${file} -> ${outFile}"
    python3 covidecg/data/khan_generate_feats_imgsuperres.py --output-dir ./data/interim/khan_feats_imgsuperres/normal --weights_path ./SRGAN_x4-ImageNet-8c4a7569.pth.tar "${file}"
done




rm -rf ./data/interim/khan_feats_imgsuperres/covid/*
# COVID
for file in ./data/interim/khan_feats_imgsuperres_4x/covid/*.png; do
    outFile="./data/interim/khan_feats_imgsuperres/covid/$(basename "${file}")"
    echo "${file} -> ${outFile}"
    python3 covidecg/data/khan_generate_feats_imgsuperres.py --output-dir ./data/interim/khan_feats_imgsuperres/covid --weights_path ./SRGAN_x4-ImageNet-8c4a7569.pth.tar "${file}"
done




rm -rf ./data/interim/khan_feats_imgsuperres/abnormal/*
# ABNORMAL HEARTBEAT
for file in ./data/interim/khan_feats_imgsuperres_4x/abnormal/*.png; do
    outFile="./data/interim/khan_feats_imgsuperres/abnormal/$(basename "${file}")"
    echo "${file} -> ${outFile}"
    python3 covidecg/data/khan_generate_feats_imgsuperres.py --output-dir ./data/interim/khan_feats_imgsuperres/abnormal --weights_path ./SRGAN_x4-ImageNet-8c4a7569.pth.tar "${file}"
done

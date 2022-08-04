covid-ecg
==============================

## Data structure and naming scheme
- **recordings**: A recorded continuos ECG snippet of e.g. 10s length.
- **sessions**: There are one ore more sessions available for each patient. One session can consist of one or more snippets of typically 10s length which were recorded shortly after another.

## Set up environment variables
Some environment variables are being read from a `.env` file.
Create `./.env` in the repository root and provide the following variables according to your setup:
```
LOG_LEVEL=INFO
SAMPLING_RATE=500
PROJECT_ROOT=~/covid-ecg
CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0
```

## Run an experiment
Example command:
```
CUDA_VISIBLE_DEVICES=0 python3 ./train_evaluate.py --exp-config ./exp_configs/exp-postcovid-ctrl-img.yaml --model-config ./exp_configs/model-vgg16.yaml
```
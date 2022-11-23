#!/usr/bin/env bash


bash scripts/load-mmc-data.sh
bash scripts/generate-mmc-tasks.sh


bash scripts/load-khan2021-data.sh
bash scripts/generate-khan2021-tasks.sh

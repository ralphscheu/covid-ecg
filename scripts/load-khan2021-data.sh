#!/usr/bin/env bash




# ##########################
# ### LOAD KHAN2021 DATA ###
# ##########################

rm -rf data/interim/_khan2021_normal
# NORMAL
for file in ./data/external/Normal\ Person\ ECG\ Images\ \(859\)/Normal*.jpg; do
    echo "Processing ${file}"
    python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
        --output-dir data/interim/_khan2021_normal \
        --input-layout ecgsheet \
        "${file}"
done


rm -rf data/interim/_khan2021_covid
# COVID Binder scans
for file in ./data/external/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/Binder*.jpg; do
    echo "Processing ${file}"
	python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
		--output-dir data/interim/_khan2021_covid \
		--input-layout binder \
        "${file}"
done
# COVID ECG Printout scans
for file in ./data/external/ECG\ Images\ of\ COVID-19\ Patients\ \(250\)/COVID*.jpg; do
    echo "Processing ${file}"
	python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
		--output-dir data/interim/_khan2021_covid \
		--input-layout ecgsheet \
        "${file}"
done


rm -rf data/interim/_khan2021_abnormal
# ABNORMAL HEARTBEAT
for file in ./data/external/ECG Images of Patient that have abnormal heart beats (548)/HB*.jpg; do
    echo "Processing ${file}"
    python3 covidecg/data/load_khan2021_dataset.py --img-height 100 \
        --output-dir data/interim/_khan2021_abnormal \
        --input-layout ecgsheet \
        "${file}"
done



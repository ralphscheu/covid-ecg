#!/usr/bin/env bash




# Create undersampled versions of ctrl data for balanced classes tasks













###
### Create task datasets in data/interim via symlinks
###




##### DEV tasks


# # mmc_tasks_raw/mmc_covid20_vs_ctrl20
# expDir=mmc_tasks_raw/mmc_covid20_vs_ctrl20
# rm -rf data/interim/$expDir
# mkdir -p data/interim/$expDir/1_covid
# mkdir -p data/interim/$expDir/0_ctrl
# ln -s `pwd`/data/interim/mmc_feats_raw/covid20/*.csv `pwd`/data/interim/$expDir/1_covid/
# ln -s `pwd`/data/interim/mmc_feats_raw/ctrl20/*.csv `pwd`/data/interim/$expDir/0_ctrl/

# # mmc_tasks_img/mmc_covid20_vs_ctrl20
# expDir=mmc_tasks_img/mmc_covid20_vs_ctrl20
# rm -rf data/interim/$expDir
# mkdir -p data/interim/$expDir/1_covid
# mkdir -p data/interim/$expDir/0_ctrl
# ln -s `pwd`/data/interim/mmc_feats_img/covid20/*.png `pwd`/data/interim/$expDir/1_covid/
# ln -s `pwd`/data/interim/mmc_feats_img/ctrl20/*.png `pwd`/data/interim/$expDir/0_ctrl/

# mmc_tasks_lfcc/mmc_covid20_vs_ctrl20
expDir=mmc_tasks_lfcc/mmc_covid20_vs_ctrl20
rm -rf ./data/interim/$expDir
mkdir -p ./data/interim/$expDir/1_covid
mkdir -p ./data/interim/$expDir/0_ctrl
ln -s `pwd`/data/interim/mmc_feats_lfcc/covid20/*.npy `pwd`/data/interim/$expDir/1_covid/
ln -s `pwd`/data/interim/mmc_feats_lfcc/ctrl20/*.npy `pwd`/data/interim/$expDir/0_ctrl/



###########



# # mmc_tasks_raw/mmc_covid_vs_ctrl610
# expDir=mmc_tasks_raw/mmc_covid_vs_ctrl610
# rm -rf data/interim/$expDir
# mkdir -p data/interim/$expDir/1_covid
# mkdir -p data/interim/$expDir/0_ctrl
# ln -s `pwd`/data/interim/mmc_feats_raw/covid/*.csv `pwd`/data/interim/$expDir/1_covid/
# ln -s `pwd`/data/interim/mmc_feats_raw/ctrl610/*.csv `pwd`/data/interim/$expDir/0_ctrl/


# # mmc_tasks_img/mmc_covid_vs_ctrl610
# rm -rf data/interim/mmc_tasks_img/mmc_covid_vs_ctrl610/*
# mkdir -p data/interim/mmc_tasks_img/mmc_covid_vs_ctrl610/1_covid
# mkdir -p data/interim/mmc_tasks_img/mmc_covid_vs_ctrl610/0_ctrl
# ln -s `pwd`/data/interim/mmc_feats_img/covid/*.png `pwd`/data/interim/mmc_tasks_img/mmc_covid_vs_ctrl610/1_covid/
# ln -s `pwd`/data/interim/mmc_feats_img/ctrl610/*.png `pwd`/data/interim/mmc_tasks_img/mmc_covid_vs_ctrl610/0_ctrl/


# mmc_tasks_lfcc/mmc_covid_vs_ctrl610
rm -rf ./data/interim/mmc_tasks_lfcc/mmc_covid_vs_ctrl610/*
mkdir -p ./data/interim/mmc_tasks_lfcc/mmc_covid_vs_ctrl610/1_covid
mkdir -p ./data/interim/mmc_tasks_lfcc/mmc_covid_vs_ctrl610/0_ctrl
ln -s `pwd`/data/interim/mmc_feats_lfcc/covid/*.npy `pwd`/data/interim/mmc_tasks_lfcc/mmc_covid_vs_ctrl610/1_covid/
ln -s `pwd`/data/interim/mmc_feats_lfcc/ctrl610/*.npy `pwd`/data/interim/mmc_tasks_lfcc/mmc_covid_vs_ctrl610/0_ctrl/



###
### Generate train/test sets for each task
###


# for expDir in mmc_tasks_raw/mmc_covid20_vs_ctrl20 mmc_tasks_raw/mmc_covid_vs_ctrl610; do

#     rm -rf data/processed/$expDir && mkdir data/processed/$expDir
#     python3 covidecg/data/split_train_test.py --test-ratio 0.2  --file-ext csv data/interim/$expDir data/processed/$expDir

# done


# for expDir in mmc_tasks_img/mmc_covid20_vs_ctrl20 mmc_tasks_img/mmc_covid_vs_ctrl610; do

#     rm -rf data/processed/$expDir && mkdir data/processed/$expDir
#     python3 covidecg/data/split_train_test.py --test-ratio 0.2  --file-ext png data/interim/$expDir data/processed/$expDir

# done


for expDir in mmc_tasks_lfcc/mmc_covid20_vs_ctrl20 mmc_tasks_lfcc/mmc_covid_vs_ctrl610; do

    rm -rf ./data/processed/$expDir && mkdir ./data/processed/$expDir
    python3 covidecg/data/split_train_test.py --test-ratio 0.2  --file-ext npy data/interim/$expDir data/processed/$expDir

done


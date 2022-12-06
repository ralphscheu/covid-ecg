#!/usr/bin/env bash


###
### Create task datasets in data/interim via symlinks
###


##### DEV tasks



######################################
########## KHAN_FEATS_IMG ############
######################################

expDir=khan_tasks_img/khan_abnormal_vs_normal548
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_abnormal
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_img/abnormal/*.png `pwd`/data/interim/$expDir/1_abnormal/
ln -s `pwd`/data/interim/khan_feats_img/normal548/*.png `pwd`/data/interim/$expDir/0_normal/

expDir=khan_tasks_img/khan_covid_vs_normal250
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_covid
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_img/covid/*.png `pwd`/data/interim/$expDir/1_covid/
ln -s `pwd`/data/interim/khan_feats_img/normal250/*.png `pwd`/data/interim/$expDir/0_normal/




######################################
######## KHAN_FEATS_IMGBLUR ##########
######################################

expDir=khan_tasks_imgblur/khan_abnormal_vs_normal548
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_abnormal
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_imgblur/abnormal/*.png `pwd`/data/interim/$expDir/1_abnormal/
ln -s `pwd`/data/interim/khan_feats_imgblur/normal548/*.png `pwd`/data/interim/$expDir/0_normal/

expDir=khan_tasks_imgblur/khan_covid_vs_normal250
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_covid
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_imgblur/covid/*.png `pwd`/data/interim/$expDir/1_covid/
ln -s `pwd`/data/interim/khan_feats_imgblur/normal250/*.png `pwd`/data/interim/$expDir/0_normal/




######################################
####### KHAN_FEATS_IMGSHARPEN ########
######################################

expDir=khan_tasks_imgsharpen/khan_abnormal_vs_normal548
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_abnormal
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_imgsharpen/abnormal/*.png `pwd`/data/interim/$expDir/1_abnormal/
ln -s `pwd`/data/interim/khan_feats_imgsharpen/normal548/*.png `pwd`/data/interim/$expDir/0_normal/

expDir=khan_tasks_imgsharpen/khan_covid_vs_normal250
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_covid
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_imgsharpen/covid/*.png `pwd`/data/interim/$expDir/1_covid/
ln -s `pwd`/data/interim/khan_feats_imgsharpen/normal250/*.png `pwd`/data/interim/$expDir/0_normal/



######################################
####### KHAN_FEATS_IMGSUPERRES #######
######################################

expDir=khan_tasks_imgsuperres/khan_abnormal_vs_normal548
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_abnormal
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_imgsuperres/abnormal/*.png `pwd`/data/interim/$expDir/1_abnormal/
ln -s `pwd`/data/interim/khan_feats_imgsuperres/normal548/*.png `pwd`/data/interim/$expDir/0_normal/

expDir=khan_tasks_imgsuperres/khan_covid_vs_normal250
rm -rf data/interim/$expDir
mkdir -p data/interim/$expDir/1_covid
mkdir -p data/interim/$expDir/0_normal
ln -s `pwd`/data/interim/khan_feats_imgsuperres/covid/*.png `pwd`/data/interim/$expDir/1_covid/
ln -s `pwd`/data/interim/khan_feats_imgsuperres/normal250/*.png `pwd`/data/interim/$expDir/0_normal/






###
### Generate train/test sets for each task
###

for expDir in khan_tasks_img/khan_abnormal_vs_normal548 khan_tasks_img/khan_covid_vs_normal250; do
    rm -rf data/processed/$expDir
    splitfolders --output data/processed/$expDir --ratio .8 .2 -- data/interim/$expDir
    mv data/processed/$expDir/val data/processed/$expDir/test
done


for expDir in khan_tasks_imgblur/khan_abnormal_vs_normal548 khan_tasks_imgblur/khan_covid_vs_normal250; do
    rm -rf data/processed/$expDir
    splitfolders --output data/processed/$expDir --ratio .8 .2 -- data/interim/$expDir
    mv data/processed/$expDir/val data/processed/$expDir/test
done


for expDir in khan_tasks_imgsharpen/khan_abnormal_vs_normal548 khan_tasks_imgsharpen/khan_covid_vs_normal250; do
    rm -rf data/processed/$expDir
    splitfolders --output data/processed/$expDir --ratio .8 .2 -- data/interim/$expDir
    mv data/processed/$expDir/val data/processed/$expDir/test
done


for expDir in khan_tasks_imgsuperres/khan_abnormal_vs_normal548 khan_tasks_imgsuperres/khan_covid_vs_normal250; do
    rm -rf data/processed/$expDir
    splitfolders --output data/processed/$expDir --ratio .8 .2 -- data/interim/$expDir
    mv data/processed/$expDir/val data/processed/$expDir/test
done



# khan2021 abnormal vs. normal
rm -rf data/interim/khan_tasks_img/khan_abnormal_vs_normal/*
mkdir data/interim/khan_tasks_img/khan_abnormal_vs_normal/1_abnormal
mkdir data/interim/khan_tasks_img/khan_abnormal_vs_normal/0_normal
ln -s `pwd`/data/interim/_khan2021_abnormal/*.png `pwd`/data/interim/khan_tasks_img/khan_abnormal_vs_normal/1_abnormal/
ln -s `pwd`/data/interim/_khan2021_normal/*.png `pwd`/data/interim/khan_tasks_img/khan_abnormal_vs_normal/0_normal/


# khan2021 covid vs. normal
rm -rf data/interim/khan_tasks_img/khan_covid_vs_normal/*
mkdir data/interim/khan_tasks_img/khan_covid_vs_normal/1_covid
mkdir data/interim/khan_tasks_img/khan_covid_vs_normal/0_normal
ln -s `pwd`/data/interim/khan_feats_img/covid/*.png `pwd`/data/interim/khan_tasks_img/khan_covid_vs_normal/1_covid/
ln -s `pwd`/data/interim/khan_feats_img/normal/*.png `pwd`/data/interim/khan_tasks_img/khan_covid_vs_normal/0_normal/






###
### Generate train/test sets for each task
###

rm -rf data/processed/khan_tasks_img/khan_abnormal_vs_normal
splitfolders --output data/processed/khan_tasks_img/khan_abnormal_vs_normal --ratio .8 .2 -- data/interim/khan_tasks_img/khan_abnormal_vs_normal
mv data/processed/khan_tasks_img/khan_abnormal_vs_normal/val data/processed/khan_tasks_img/khan_abnormal_vs_normal/test



rm -rf data/processed/khan_tasks_img/khan_covid_vs_normal
splitfolders --output data/processed/khan_tasks_img/khan_covid_vs_normal --ratio .8 .2 -- data/interim/khan_tasks_img/khan_covid_vs_normal
mv data/processed/khan_tasks_img/khan_covid_vs_normal/val data/processed/khan_tasks_img/khan_covid_vs_normal/test



###
### Create task datasets in data/interim via symlinks
###

# khan2021 abnormal vs. normal
rm -rf data/interim/khan2021_abnormal_khan2021_normal
mkdir -p data/interim/khan2021_abnormal_khan2021_normal/1_abnormal
mkdir -p data/interim/khan2021_abnormal_khan2021_normal/0_normal
ln -s `pwd`/data/interim/_khan2021_abnormal/*.png `pwd`/data/interim/khan2021_abnormal_khan2021_normal/1_abnormal/
ln -s `pwd`/data/interim/_khan2021_normal/*.png `pwd`/data/interim/khan2021_abnormal_khan2021_normal/0_normal/


# khan2021 covid vs. normal
rm -rf data/interim/khan2021_covid_khan2021_normal
mkdir -p data/interim/khan2021_covid_khan2021_normal/1_covid
mkdir -p data/interim/khan2021_covid_khan2021_normal/0_normal
ln -s `pwd`/data/interim/_khan2021_covid/*.png `pwd`/data/interim/khan2021_covid_khan2021_normal/1_covid/
ln -s `pwd`/data/interim/_khan2021_normal/*.png `pwd`/data/interim/khan2021_covid_khan2021_normal/0_normal/



# khan2021 abnormal vs. normal (undersampled to 548 samples)
rm -rf data/interim/khan2021_abnormal_khan2021_normal548
mkdir -p data/interim/khan2021_abnormal_khan2021_normal548/1_abnormal
mkdir -p data/interim/khan2021_abnormal_khan2021_normal548/0_normal
ln -s `pwd`/data/interim/_khan2021_abnormal/*.png `pwd`/data/interim/khan2021_abnormal_khan2021_normal548/1_abnormal/
ln -s `pwd`/data/interim/_khan2021_normal548/*.png `pwd`/data/interim/khan2021_abnormal_khan2021_normal548/0_normal/


# khan2021 covid vs. normal (undersampled to 250 samples chosen from the 397 from above)
rm -rf data/interim/khan2021_covid_khan2021_normal250
mkdir -p data/interim/khan2021_covid_khan2021_normal250/1_covid
mkdir -p data/interim/khan2021_covid_khan2021_normal250/0_normal
ln -s `pwd`/data/interim/_khan2021_covid/*.png `pwd`/data/interim/khan2021_covid_khan2021_normal250/1_covid/
ln -s `pwd`/data/interim/_khan2021_normal250/*.png `pwd`/data/interim/khan2021_covid_khan2021_normal250/0_normal/







###
### Generate train/test sets for each task
###


# Generate train/test sets for khan2021_abnormal_khan2021_normal
rm -rf data/processed/khan2021_abnormal_khan2021_normal
splitfolders --output data/processed/khan2021_abnormal_khan2021_normal --ratio .8 .2 -- data/interim/khan2021_abnormal_khan2021_normal
mv data/processed/khan2021_abnormal_khan2021_normal/val data/processed/khan2021_abnormal_khan2021_normal/test


# Generate train/test sets for khan2021_covid_khan2021_normal
rm -rf data/processed/khan2021_covid_khan2021_normal
splitfolders --output data/processed/khan2021_covid_khan2021_normal --ratio .8 .2 -- data/interim/khan2021_covid_khan2021_normal
mv data/processed/khan2021_covid_khan2021_normal/val data/processed/khan2021_covid_khan2021_normal/test



# Generate train/test sets for khan2021_abnormal_khan2021_normal548
rm -rf data/processed/khan2021_abnormal_khan2021_normal548
splitfolders --output data/processed/khan2021_abnormal_khan2021_normal548 --ratio .8 .2 -- data/interim/khan2021_abnormal_khan2021_normal548
mv data/processed/khan2021_abnormal_khan2021_normal548/val data/processed/khan2021_abnormal_khan2021_normal548/test


# Generate train/test sets for khan2021_covid_khan2021_normal250
rm -rf data/processed/khan2021_covid_khan2021_normal250
splitfolders --output data/processed/khan2021_covid_khan2021_normal250 --ratio .8 .2 -- data/interim/khan2021_covid_khan2021_normal250
mv data/processed/khan2021_covid_khan2021_normal250/val data/processed/khan2021_covid_khan2021_normal250/test

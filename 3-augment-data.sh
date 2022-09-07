#!/usr/bin/env bash

### NOISE ###
indir=data/processed/mmc_recs_ctrl
outdir=data/processed/mmc_recs_ctrl_noise
rm -rf t-dir $outdir
mkdir $outdir
for file in $indir/*.png; do
    python3 augment_ecgleadsgrid.py \
        --aug-method noise --noise-mode speckle \
        $file $outdir/`basename $file`
    echo $file \> $outdir/`basename $file`
done

indir=data/processed/mmc_recs_postcovid
outdir=data/processed/mmc_recs_postcovid_noise
rm -rf t-dir $outdir
mkdir $outdir
for file in $indir/*.png; do
    python3 augment_ecgleadsgrid.py \
        --aug-method noise --noise-mode speckle \
        $file $outdir/`basename $file`
    echo $file \> $outdir/`basename $file`
done


### COMBINE CLEAN AND NOISY VARIANTS
outdir=data/processed/mmc_recs_ctrl_aug
rm -rf t-dir $outdir
mkdir $outdir
for file in data/processed/mmc_recs_ctrl/*.png; do
    fileBaseName=`basename $file`
    ln -s `pwd`/$file $outdir/${fileBaseName%.*}.png
done
for file in data/processed/mmc_recs_ctrl_noise/*.png; do
    fileBaseName=`basename $file`
    ln -s `pwd`/$file $outdir/${fileBaseName%.*}_noise.png
done


outdir=data/processed/mmc_recs_postcovid_aug
rm -rf t-dir $outdir
mkdir $outdir
for file in data/processed/mmc_recs_postcovid/*.png; do
    fileBaseName=`basename $file`
    ln -s `pwd`/$file $outdir/${fileBaseName%.*}.png
done
for file in data/processed/mmc_recs_postcovid_noise/*.png; do
    fileBaseName=`basename $file`
    ln -s `pwd`/$file $outdir/${fileBaseName%.*}_noise.png
done


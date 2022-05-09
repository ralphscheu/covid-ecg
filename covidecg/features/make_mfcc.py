# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):

    # create output dir if not exists
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    files = list(Path(input_dir).glob('*RUN*.txt'))
    for file in tqdm(files, desc="Processing files"):

        signal = np.loadtxt(file, skiprows=12)

        if signal.shape[1] > 5000:
            continue
        elif signal.shape[1] < 5000: 
            print(f"Short signal! ({signal.shape[1]})")

        mfcc_feat = []
        for lead_i, lead_signal in enumerate(signal):
            lead_mfcc = mfcc(lead_signal, int(os.environ['SAMPLING_RATE']))
            mfcc_feat.append(lead_mfcc)
        
        mfcc_feat = np.stack(mfcc_feat)
        np.save(os.path.join(output_dir, f"{file.stem}_mfcc.npy"), mfcc_feat)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

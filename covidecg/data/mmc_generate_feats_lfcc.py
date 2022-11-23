import os
from covidecg.data.dataset import *
from tqdm import tqdm
import click
from dotenv import find_dotenv, load_dotenv
from PIL import Image
import logging
import pathlib
import spafe.features.lfcc


@click.command()
@click.argument('input_dir', required=True, type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.argument('output_dir', required=True, type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option('--sampling-rate', default=500, type=int)
def main(input_dir, output_dir, sampling_rate):
    
    files = list(pathlib.Path(input_dir).glob('*.csv'))
    for file in tqdm(files, desc="Processing files"):
        
        rec_signal = data_utils.load_signal(file)
        rec_signal = data_utils.clean_signal(rec_signal)
        lfcc = np.stack([spafe.features.lfcc.lfcc(lead_signal, fs=sampling_rate) for lead_signal in rec_signal], axis=0)
        
        out_file = output_dir / f"{file.stem}.npy"
        np.save(out_file, lfcc)
        # print(f"{file} -> {out_file}")

if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()
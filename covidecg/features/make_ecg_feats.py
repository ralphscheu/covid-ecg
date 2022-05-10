import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np
import neurokit2 as nk
from covidecg.features.utils import compute_rr_intervals


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def main(input_dir, output_file):

    # create output dir if not exists
    # try:
    #     os.mkdir(output_dir)
    # except FileExistsError:
    #     pass

    feats = pd.DataFrame(columns=['run_id', 'rpeak_location', 'rpeak_energy', 'rr_intervals', 'rr_interval_mean', 'rr_interval_std'])

    files = list(Path(input_dir).glob('*RUN*.txt'))
    for file in tqdm(files, desc="Processing files"):

        run_id = file.stem

        # load raw signal
        raw_signal = np.loadtxt(file, skiprows=12)
        raw_signal = raw_signal[1, :]  # Lead II

        # extract wave and rate information
        try:
            signals, info = nk.ecg_process(raw_signal, sampling_rate=int(os.environ['SAMPLING_RATE']))
        except IndexError:
            continue
        rpeak_location = info['ECG_R_Peaks']
        rr_intervals = compute_rr_intervals(rpeak_location)
        
        feats = pd.concat([
            feats,
            pd.DataFrame({
                'run_id': run_id,
                'rpeak_location': " ".join(map(str, rpeak_location)),
                'rpeak_energy': None,
                'rr_intervals': " ".join(map(str, rr_intervals)),
                'rr_interval_mean': rr_intervals.mean(),
                'rr_interval_std': rr_intervals.std()
            }, index=[run_id])
        ], ignore_index=True)

    feats.to_csv(output_file, sep=';')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

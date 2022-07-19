# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import covidecg.data.utils as data_utils
warnings.filterwarnings('ignore')


@click.command()
@click.argument('recordings_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--recordings-list', type=click.Path(exists=True))
def main(recordings_dir, output_dir, recordings_list):

    logger = logging.getLogger(__name__)

    # create output dir if not exists
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    
    print(recordings_list)
    recordings_list = pd.read_csv(recordings_list, sep=';')

    recordings_list = recordings_list.loc[recordings_list.ecg_length == 5000]

    for session_id in tqdm(recordings_list.session.unique(), desc="Processing sessions"):
        recordings_in_session = recordings_list.loc[recordings_list.session == session_id]

        fullsession_signal = pd.DataFrame()

        for recording_id in recordings_in_session.recording:
            # print(session_id, ":", recording_id)
            # read recording signal
            rec_signal = pd.read_csv(os.path.join(recordings_dir, recording_id+'.csv'), index_col=0)
            # print("recording", recording_id, ":", rec_signal.shape)
            fullsession_signal = pd.concat([fullsession_signal, rec_signal])
        
        # print("fullsession_signal:", fullsession_signal.shape)
        fullsession_signal.to_csv(os.path.join(output_dir, session_id+'.csv'))


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import xml.etree.cElementTree as ET
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

namespaces = {'': 'urn:hl7-org:v3'}
datetime_format = '%Y%m%d%H%M%S.%f'


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_dir', type=click.Path(exists=False, file_okay=False, path_type=Path))
@click.option('--prefix', required=True, type=str)
@click.option('--patients-list', required=True, type=click.Path())
@click.option('--min-length', type=int, default=5000)
@click.option('--max-length', type=int, default=5000)
@click.option('--sampling-rate', type=float, default=500)
def main(input_dir, output_dir, prefix, patients_list, min_length, max_length, sampling_rate):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Processing XML files in {input_dir}')

    os.makedirs(output_dir)
    
    logging.basicConfig(level=logging.INFO, format=os.getenv('LOG_FORMAT'), handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / 'extract_recordings.log')])

    patients_list = pd.read_csv(patients_list, sep=';')

    recording_list = pd.DataFrame()

    files = list(Path(input_dir).glob('*.xml'))
    # loop through all downloaded session files
    for file in tqdm(files, desc="Processing files"):
        
        file_name_split = file.name.split('-')
        pat_id = prefix + file_name_split[0]
        ecg_type = file_name_split[-2]

        tree = ET.parse(file)
        root = tree.getroot()

        session_start = root.find('effectiveTime/low', namespaces).get('value')
        session_id = pat_id + "_" + session_start

        # find all recordings in session
        for recording_counter, recording in enumerate(root.findall('component/series', namespaces)):

            # effectiveTimeLow = ecg_run.find('effectiveTime/low', namespaces).get('value')
            sequenceSet = recording.find('component/sequenceSet', namespaces)
            leads = sequenceSet.findall('component/sequence', namespaces)[1:13]  # first instance contains metadata

            rec_df = pd.DataFrame()
            # loop through leads in recording
            for lead in leads:
                lead_name = lead.find('code', namespaces).attrib['code']
                lead_digits = lead.find('value/digits', namespaces).text
                lead_digits = lead_digits.split(' ')  # convert to list for saving into DataFrame column
                rec_df[lead_name] = lead_digits
                rec_df[lead_name] = rec_df[lead_name].astype(float)
                
            max_amplitude = rec_df['MDC_ECG_LEAD_II'].max() - rec_df['MDC_ECG_LEAD_II'].min()
            if max_amplitude > 5000 or max_amplitude < 10:
                logging.warn(f"Max amplitude of {max_amplitude} in Lead II -> Sample dismissed!")
                continue

            # number of samples in recording
            # -> ecg_length / sampling_rate = recording length in seconds
            ecg_length = len(rec_df.index)
            if ecg_length > max_length or ecg_length < min_length:
                continue

            # generate DataFrame Index as milliseconds from start of run (integer)
            ms_between_samples = int(1.0 / sampling_rate * 1000)
            rec_df.index = pd.RangeIndex(0, ms_between_samples * len(rec_df.index), ms_between_samples)

            # save ECG data
            recording_id = f"{session_id}_rec{recording_counter}"
            rec_df.to_csv(f"{output_dir}/{recording_id}.csv", index=True)
            
            # get patient information
            patients_list['birth_date'] = pd.to_datetime(patients_list['birth_date'], format='%Y-%m-%d')
            patients_list['diagnose_date'] = pd.to_datetime(patients_list['diagnose_date'], format='%Y-%m-%d')
            patients_list['age'] = (patients_list['diagnose_date'] - patients_list['birth_date']).astype('<m8[Y]')
            pat_info = patients_list.loc[ patients_list['nr'].astype(str) == file_name_split[0]].iloc[0]

            # exclude if recording was done more than 7 days before diagnosis (covid and postcovid patients)
            from datetime import datetime
            recording_date = datetime.strptime(session_start, datetime_format)
            # logger.info(f"diagnose_date: {pat_info['diagnose_date']}")
            # logger.info(f"recording_date: {recording_date}")
            # logger.info(f"Delta: {(recording_date - pat_info['diagnose_date']).days} days")
            if prefix in ['covid', 'postcovid'] and (recording_date - pat_info['diagnose_date']).days < -7:
                continue


            recording_list = pd.concat(
                [
                    recording_list,
                    pd.DataFrame({
                        'recording': recording_id,
                        'recording_date': recording_date,
                        'session': session_id,
                        'pat_id': pat_id,
                        'pat_group': prefix,
                        'pat_gender': pat_info['gender'],
                        'pat_age': pat_info['age'],
                        'pat_diagnosis': pat_info['diagnose'],
                        'pat_diagnosis_date': pat_info['diagnose_date'],
                        'ecg_type': ecg_type,
                        'ecg_length': ecg_length
                    }, index=[recording_id])
                ]
            )

    # save stress ECG recordings metadata
    recordings_stress_ecg = recording_list.loc[recording_list.ecg_type == 'Belastungs']
    recordings_stress_ecg.to_csv(f'data/interim/mmc_recs_stress_{prefix}.csv', sep=';', index=False)
    
    # TODO save labels in text file (alternative: extract labels from filenames later on)
    
    logger.info(f'Done. Saved csv files to {output_dir}')


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    main()

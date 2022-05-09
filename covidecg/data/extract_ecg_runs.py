# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import xml.etree.cElementTree as ET
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import pandas as pd

namespaces = {'': 'urn:hl7-org:v3'}
datetime_format = '%Y%m%d%H%M%S.%f'


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.argument('prefix', type=str)
def main(input_dir, output_dir, prefix):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Processing XML files in {input_dir}')

    # create output dir if not exists
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    ecg_runs = pd.DataFrame(
        columns=[
            'pat_id', 'pat_group', 'ecg_type', 'ecg_length'
        ]
    )

    files = list(Path(input_dir).glob('*.xml'))
    for file in tqdm(files, desc="Processing files"):

        file_name_split = file.name.split('-')
        pat_id = prefix + file_name_split[0]
        ecg_type = file_name_split[-2]

        tree = ET.parse(file)
        root = tree.getroot()

        for run_i, ecg_run in enumerate(
            root.findall('component/series', namespaces)):

            effectiveTimeLow = ecg_run.find('effectiveTime/low', namespaces).get('value')
            # effectiveTimeHigh = ecg_run.find('effectiveTime/high', namespaces).get('value')
            sequenceSet = ecg_run.find('component/sequenceSet', namespaces)
            leads = sequenceSet.findall('component/sequence', namespaces)[1:13]

            lead_names = []
            digits = []

            for lead in leads:
                lead_names.append(lead.find('code', namespaces).attrib['code'])
                lead_digits = lead.find('value/digits', namespaces).text
                digits.append(lead_digits)

            run_id = f"{pat_id}_RUN{effectiveTimeLow}"

            with open(f"{output_dir}/{run_id}.txt", 'w') as f:
                # save lead names order for reference
                for ln in lead_names:
                    f.write(f'{ln}\n')
                # save ECG measurements
                for d in digits:
                    f.write(f'{d}\n')

            ecg_runs = pd.concat(
                [
                    ecg_runs,
                    pd.DataFrame({
                        'pat_id': pat_id,
                        'pat_group': prefix,
                        'ecg_type': ecg_type,
                        'ecg_length': len(digits[0].split(' '))
                    }, index=[run_id])
                ]
            )

    ecg_runs.to_csv(f'data/interim/ecg_runs_{prefix}.csv')


    logger.info(f'Done. Saved txt files to {output_dir}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

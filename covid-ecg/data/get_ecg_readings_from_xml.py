# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import xml.etree.cElementTree as ET
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

namespaces = {'': 'urn:hl7-org:v3'}
datetime_format = '%Y%m%d%H%M%S.%f'


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Processing XML files in {input_dir}')

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    files = list(Path(input_dir).glob('*.xml'))
    for file in tqdm(files, desc="Processing files"):
        ecg = {}
        
        file_name_split = file.name.split('-')
        ecg['id'] = file_name_split[0]
        ecg['type'] = file_name_split[-2]
        
        tree = ET.parse(file)
        root = tree.getroot()
        
        # ECG data
        ecg_runs = root.findall('component/series', namespaces)
        for ecg_run in ecg_runs:
            # TODO these effective times seem to be more accurate regarding the 10s recording, maybe not use the times higher up in the xml hierarchy
            effectiveTimeLow = ecg_run.find('effectiveTime/low', namespaces).get('value')
            effectiveTimeHigh = ecg_run.find('effectiveTime/high', namespaces).get('value')
            sequenceSet = ecg_run.find('component/sequenceSet', namespaces)
            leads = sequenceSet.findall('component/sequence', namespaces)[1:13]
            
            lead_names = []
            digits = []

            for lead in leads:
                lead_names.append( lead.find('code', namespaces).attrib['code'] )
                digits.append( lead.find('value/digits', namespaces).text )
            
            # write out to file
            with open(f'{output_dir}/{file.name}_{effectiveTimeLow}_{effectiveTimeHigh}.txt', 'w') as f:
                for ln in lead_names:
                    f.write(f'{ln}\n')
                for d in digits:
                    f.write(f'{d}\n')

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

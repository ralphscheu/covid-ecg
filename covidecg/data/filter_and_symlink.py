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
import cv2


@click.command()
@click.argument('in_dir', required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('out_dir', required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option('--min-length', required=True, type=int)
@click.option('--max-length', required=True, type=int)
def main(in_dir, out_dir, min_length, max_length):
    os.makedirs(out_dir)
    min_length = min_length // 5
    max_length = max_length // 5
    for f in Path(in_dir).glob('*.png'):
        input_img = cv2.imread(str(f))  # load ecggrid image
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY )  # convert to grayscale
        # print(f"{f.name} - input_img width: {input_img.shape[1]}")
        
        ecggrid_col_width = input_img.shape[1] // 4
        if min_length >= ecggrid_col_width >= max_length:
            dst = out_dir / f.name
            os.symlink(f.resolve(), dst.resolve())
            print(f"Created symlink {f} -> {dst}")


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', default='INFO'), 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()

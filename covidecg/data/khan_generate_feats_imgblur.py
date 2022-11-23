import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import click
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import covidecg.data.utils as data_utils
import os
from pathlib import Path
from PIL import Image




@click.command()
@click.argument('in_file', required=True, type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option('--output-dir', required=True, type=click.Path(path_type=Path, file_okay=False))
def main(in_file, output_dir):
    
    im = cv2.imread(str(in_file))  # load ECG image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    
    im = cv2.GaussianBlur(im,(5,5),0)
    
    out_file = output_dir / in_file.name
    Image.fromarray(im).convert('L').save(out_file)
    print(f"{in_file} -> {out_file}")

    

if __name__ == '__main__':
    main()
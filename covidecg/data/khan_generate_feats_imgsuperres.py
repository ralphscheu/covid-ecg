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
from covidecg.models.srgan_pytorch.inference import main as srgan_inference


@click.command()
@click.argument('in_file', required=True, type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option('--output-dir', required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option('--weights-path', required=True, type=click.Path(exists=True, path_type=Path, dir_okay=False))
def main(in_file, output_dir, weights_path):
    
    im = cv2.imread(str(in_file))  # load ECG image
    
    out_file = os.path.join(output_dir, in_file.stem + '.png')
    
    im = srgan_inference(in_file, out_file, weights_path)
    
    # reduce dimensions by factor 4 to restore original image size before Super Resolution
    im = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4), cv2.INTER_AREA)


    # arrange ECG leads in grid and save as png
    Image.fromarray(im).convert('L').save(out_file)
    print(f"Processed {in_file}. Saved output in {out_file}")


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()

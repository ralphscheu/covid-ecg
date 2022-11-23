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
from covidecg.data.khan_generate_feats_img import crop_to_grid, remove_grid_background, slice_binder_scan, slice_ecgsheet_scan, slice_ecgsheet2_scan



def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=3.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



@click.command()
@click.argument('in_file', required=True, type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option('--output-dir', required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option('--input-layout', required=True, type=click.Choice(['ecgsheet', 'ecgsheet2', 'binder']))
@click.option('--img-height', default=200, type=int)
def main(in_file, output_dir, input_layout, img_height):

    im = cv2.imread(str(in_file))  # load ECG image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # convert to RGB color space
    
    if input_layout == 'binder':
        binder_layout = 'portrait' if im.shape[0] > im.shape[1] else 'landscape'
    else:
        binder_layout = None
    

    RMGRID_CUTOFF_ECGSHEET = 40
    RMGRID_CUTOFF_BINDER = 170

    if input_layout == 'ecgsheet':
        im = crop_to_grid(im)  # crop out irrelevant parts
        im = remove_grid_background(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), RMGRID_CUTOFF_ECGSHEET)  # remove grid in background
        im = unsharp_mask(im)
        ecg_grid_array = slice_ecgsheet_scan(im, img_height)
        
    elif input_layout == 'ecgsheet2':
        im = remove_grid_background(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), RMGRID_CUTOFF_BINDER)  # remove grid in background
        im = unsharp_mask(im)
        ecg_grid_array = slice_ecgsheet2_scan(im, img_height)
        
    elif input_layout == 'binder':
        im = crop_to_grid(im)  # crop out irrelevant parts
        im = remove_grid_background(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), RMGRID_CUTOFF_BINDER)  # remove grid in background
        im = unsharp_mask(im)
        ecg_grid_array = slice_binder_scan(im, img_height, binder_layout)

    # arrange ECG leads in grid and save as png
    ecg_leads_grid_savepath = os.path.join(output_dir, in_file.stem + '.png')
    ecg_leads_grid = data_utils.generate_ecg_leads_grid(ecg_grid_array)
    Image.fromarray(ecg_leads_grid).convert('L').save(ecg_leads_grid_savepath)
    print(f"Processed {in_file}. Saved output in {ecg_leads_grid_savepath}")


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()

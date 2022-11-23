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


def crop_to_grid(im):
    # Gen lower mask (0-5) and upper mask (175-180) of RED
    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(im_hsv, (  0, 50, 30), (10,   255, 255))
    mask2 = cv2.inRange(im_hsv, (170, 50, 30), (180, 255, 255))
    # Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2 )
    nonzero_coords = cv2.findNonZero(mask)
    
    if nonzero_coords is None:
        # HB (400) - HB (483)
        x, y, w, h = 68, 283, 2109, 1235
    else:
        x, y, w, h = cv2.boundingRect(nonzero_coords) # Find minimum spanning bounding box
    return im[y:y+h, x:x+w]


def remove_grid_background(im, cutoff_value):
    im[im >= cutoff_value] = 255
    return im



def slice_ecgsheet_scan(img, img_height):

    lead_width, lead_height = 430, 240
    row0_baseline, row1_baseline, row2_baseline = 181, 480, 783
    col0_left, col1_left, col2_left, col3_left = 134, 635, 1120, 1615

    lead_I =    img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_II =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_III =  img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_aVR =  img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_aVL =  img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_aVF =  img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V1 =   img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col2_left:col2_left+lead_width]
    lead_V2 =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col2_left:col2_left+lead_width]
    lead_V3 =   img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col2_left:col2_left+lead_width]
    lead_V4 =   img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col3_left:col3_left+lead_width]
    lead_V5 =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col3_left:col3_left+lead_width]
    lead_V6 =   img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col3_left:col3_left+lead_width]
    
    arr = []
    for lead in [lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF, lead_V1, lead_V2, lead_V3, lead_V4, lead_V5, lead_V6]:
        # lead.shape[1] // 2 -> reduce width by 50% so 100px equals 1 second as in other data sources
        lead = cv2.resize(lead, (lead.shape[1] // 2, img_height), interpolation=cv2.INTER_AREA)
        arr.append(lead)
    arr = np.stack(arr, axis=0)
    return arr



def slice_ecgsheet2_scan(img, img_height):

    lead_width, lead_height = 900, 200
    row0_baseline, row1_baseline, row2_baseline, row3_baseline, row4_baseline, row5_baseline = 400, 569, 741, 914, 1086, 1258
    col0_left, col1_left = 210, 1200

    lead_I =    img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_II =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_III =  img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_aVR =  img[row3_baseline - lead_height // 2:row3_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_aVL =  img[row4_baseline - lead_height // 2:row4_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_aVF =  img[row5_baseline - lead_height // 2:row5_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_V1 =   img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V2 =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V3 =   img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V4 =   img[row3_baseline - lead_height // 2:row3_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V5 =   img[row4_baseline - lead_height // 2:row4_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V6 =   img[row5_baseline - lead_height // 2:row5_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    
    arr = []
    for lead in [lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF, lead_V1, lead_V2, lead_V3, lead_V4, lead_V5, lead_V6]:
        # lead.shape[1] // 2 -> reduce width by 50% so 100px equals 1 second as in other data sources
        lead = cv2.resize(lead, (lead.shape[1] // 2, img_height), interpolation=cv2.INTER_AREA)
        arr.append(lead)
    arr = np.stack(arr, axis=0)
    return arr


def slice_binder_scan(img, img_height, binder_layout):
    if binder_layout == 'portrait':
        lead_width, lead_height = 145, 84
        row0_baseline, row1_baseline, row2_baseline = 42, 152, 264
        col0_left, col1_left, col2_left, col3_left = 48, 221, 396, 566
        img = cv2.resize(img, (723, 432), cv2.INTER_CUBIC)
    elif binder_layout == 'landscape':
        lead_width, lead_height = 245, 143
        row0_baseline, row1_baseline, row2_baseline = 73, 262, 452
        col0_left, col1_left, col2_left, col3_left = 81, 375, 670, 965
        img = cv2.resize(img, (1232, 735), cv2.INTER_CUBIC)
    else:
        raise Exception(f"Invalid value {binder_layout} for binder_layout")
    
    lead_I =    img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_II =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_III =  img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col0_left:col0_left+lead_width]
    lead_aVR =  img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_aVL =  img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_aVF =  img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col1_left:col1_left+lead_width]
    lead_V1 =   img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col2_left:col2_left+lead_width]
    lead_V2 =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col2_left:col2_left+lead_width]
    lead_V3 =   img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col2_left:col2_left+lead_width]
    lead_V4 =   img[row0_baseline - lead_height // 2:row0_baseline + lead_height // 2, col3_left:col3_left+lead_width]
    lead_V5 =   img[row1_baseline - lead_height // 2:row1_baseline + lead_height // 2, col3_left:col3_left+lead_width]
    lead_V6 =   img[row2_baseline - lead_height // 2:row2_baseline + lead_height // 2, col3_left:col3_left+lead_width]
    print(f"col3_left+lead_width: {col3_left+lead_width}")
    
    arr = []
    for lead in [lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF, lead_V1, lead_V2, lead_V3, lead_V4, lead_V5, lead_V6]:
        print(f"lead.shape: {lead.shape}")
        # lead_width * (100/70) -> expand width by (100/70) so 100px equals 1 second as in other data sources
        lead = cv2.resize(lead, (int(lead_width * (100.0/70.0)), img_height), interpolation=cv2.INTER_AREA)
        arr.append(lead)
    arr = np.stack(arr, axis=0)
    return arr


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
        ecg_grid_array = slice_ecgsheet_scan(im, img_height)
        
    elif input_layout == 'ecgsheet2':
        im = remove_grid_background(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), RMGRID_CUTOFF_BINDER)  # remove grid in background
        ecg_grid_array = slice_ecgsheet2_scan(im, img_height)
        
    elif input_layout == 'binder':
        im = crop_to_grid(im)  # crop out irrelevant parts
        im = remove_grid_background(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), RMGRID_CUTOFF_BINDER)  # remove grid in background
        ecg_grid_array = slice_binder_scan(im, img_height, binder_layout)

    # arrange ECG leads in grid and save as png
    ecg_leads_grid_savepath = os.path.join(output_dir, in_file.stem + '.png')
    ecg_leads_grid = data_utils.generate_ecg_leads_grid(ecg_grid_array)
    Image.fromarray(ecg_leads_grid).convert('L').save(ecg_leads_grid_savepath)
    print(f"Processed {in_file}. Saved output in {ecg_leads_grid_savepath}")


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()

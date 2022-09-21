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

# Define constants for extracting leads from scanned ECG sheet
ECGSHEET_LEAD_WIDTH = 430
ECGSHEET_COL0_RIGHT = 642
ECGSHEET_COL1_RIGHT = 1134
ECGSHEET_COL2_RIGHT = 1625
ECGSHEET_COL3_RIGHT = 2114
ECGSHEET_COL0_LEFT = ECGSHEET_COL0_RIGHT - ECGSHEET_LEAD_WIDTH
ECGSHEET_COL1_LEFT = ECGSHEET_COL1_RIGHT - ECGSHEET_LEAD_WIDTH
ECGSHEET_COL2_LEFT = ECGSHEET_COL2_RIGHT - ECGSHEET_LEAD_WIDTH
ECGSHEET_COL3_LEFT = ECGSHEET_COL3_RIGHT - ECGSHEET_LEAD_WIDTH
ECGSHEET_LEAD_HEIGHT = 240
ECGSHEET_ROW0_BASELINE = 461
ECGSHEET_ROW1_BASELINE = 762
ECGSHEET_ROW2_BASELINE = 1065
ECGSHEET_ROW0_TOP = ECGSHEET_ROW0_BASELINE - (ECGSHEET_LEAD_HEIGHT // 2)
ECGSHEET_ROW0_BOTTOM = ECGSHEET_ROW0_BASELINE + (ECGSHEET_LEAD_HEIGHT // 2)
ECGSHEET_ROW1_TOP = ECGSHEET_ROW1_BASELINE - (ECGSHEET_LEAD_HEIGHT // 2)
ECGSHEET_ROW1_BOTTOM = ECGSHEET_ROW1_BASELINE + (ECGSHEET_LEAD_HEIGHT // 2)
ECGSHEET_ROW2_TOP = ECGSHEET_ROW2_BASELINE - (ECGSHEET_LEAD_HEIGHT // 2)
ECGSHEET_ROW2_BOTTOM = ECGSHEET_ROW2_BASELINE + (ECGSHEET_LEAD_HEIGHT // 2)

def filter_fn(x, cutoff=50):
    # set pixel to white if brighter than cutoff value
    return 255 if x > cutoff else x
filter_fn = np.vectorize(filter_fn)

def remove_background_grid(img):
    """ Remove background millimeter grid in Khan2021 scans """
    img = cv2.fastNlMeansDenoising(np.uint8(img), dst=None, h=20, templateWindowSize=4, searchWindowSize=20)
    img = filter_fn(img).astype('float32')
    return img

def do_enhance(im, decision_value=220):
    im_contrast = im.copy()
    im_contrast[im_contrast < decision_value] = 0
    im_contrast[im_contrast >= decision_value] = 255
    return im_contrast

def ecgsheet_to_ecgimgdata(im, img_height=100):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY )  # convert to grayscale
    lead_I = im[ECGSHEET_ROW0_TOP:ECGSHEET_ROW0_BOTTOM, ECGSHEET_COL0_LEFT:ECGSHEET_COL0_RIGHT]
    lead_II = im[ECGSHEET_ROW1_TOP:ECGSHEET_ROW1_BOTTOM, ECGSHEET_COL0_LEFT:ECGSHEET_COL0_RIGHT]
    lead_III = im[ECGSHEET_ROW2_TOP:ECGSHEET_ROW2_BOTTOM, ECGSHEET_COL0_LEFT:ECGSHEET_COL0_RIGHT]
    lead_aVR = im[ECGSHEET_ROW0_TOP:ECGSHEET_ROW0_BOTTOM, ECGSHEET_COL1_LEFT:ECGSHEET_COL1_RIGHT]
    lead_aVL = im[ECGSHEET_ROW1_TOP:ECGSHEET_ROW1_BOTTOM, ECGSHEET_COL1_LEFT:ECGSHEET_COL1_RIGHT]
    lead_aVF = im[ECGSHEET_ROW2_TOP:ECGSHEET_ROW2_BOTTOM, ECGSHEET_COL1_LEFT:ECGSHEET_COL1_RIGHT]
    lead_V1 = im[ECGSHEET_ROW0_TOP:ECGSHEET_ROW0_BOTTOM, ECGSHEET_COL2_LEFT:ECGSHEET_COL2_RIGHT]
    lead_V2 = im[ECGSHEET_ROW1_TOP:ECGSHEET_ROW1_BOTTOM, ECGSHEET_COL2_LEFT:ECGSHEET_COL2_RIGHT]
    lead_V3 = im[ECGSHEET_ROW2_TOP:ECGSHEET_ROW2_BOTTOM, ECGSHEET_COL2_LEFT:ECGSHEET_COL2_RIGHT]
    lead_V4 = im[ECGSHEET_ROW0_TOP:ECGSHEET_ROW0_BOTTOM, ECGSHEET_COL3_LEFT:ECGSHEET_COL3_RIGHT]
    lead_V5 = im[ECGSHEET_ROW1_TOP:ECGSHEET_ROW1_BOTTOM, ECGSHEET_COL3_LEFT:ECGSHEET_COL3_RIGHT]
    lead_V6 = im[ECGSHEET_ROW2_TOP:ECGSHEET_ROW2_BOTTOM, ECGSHEET_COL3_LEFT:ECGSHEET_COL3_RIGHT]
    
    arr = []
    for l in [lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF, lead_V1, lead_V2, lead_V3, lead_V4, lead_V5, lead_V6]:
        l_filtered = remove_background_grid(l)
        # l_filtered.shape[1] // 2 -> reduce width by 50% so 100px equals 1 second as in other data sources
        l_filtered = cv2.resize(l_filtered, (l_filtered.shape[1] // 2, img_height), interpolation=cv2.INTER_AREA)
        arr.append(l_filtered)
    arr = np.stack(arr, axis=0)
    return arr


def binder_to_ecgimgdata(img, img_height, enhance, decision_value):
    img_color = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )  # convert to grayscale
    # Crop on grayscale version to remove whitespace
    nonzero_coords = cv2.findNonZero(~img) # Find all non-zero points
    x, y, w, h = cv2.boundingRect(nonzero_coords) # Find minimum spanning bounding box
    
    img_color = img_color[y:y+h, x:x+w]
    img_color[img_color[:, :, 0] > 170] = 255
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY )  # convert to grayscale
    img = cv2.resize(img, (730, 430), interpolation=cv2.INTER_AREA)  # resize to common dimensions

    lead_width, lead_height = 145, 84
    row0_baseline, row1_baseline, row2_baseline = 60, 168, 277
    col0_left, col1_left, col2_left, col3_left = 34, 223, 391, 566
    
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
        # lead.shape[1] * (100/70) -> expand width by (100/70) so 100px equals 1 second as in other data sources
        lead = cv2.resize(lead, (int(lead.shape[1] * (100.0/70.0)), img_height), interpolation=cv2.INTER_AREA)
        if enhance:
            lead = do_enhance(lead, decision_value)
        arr.append(lead)
    arr = np.stack(arr, axis=0)
    return arr


@click.command()
@click.argument('in_file', required=True, type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option('--output-dir', required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option('--input-layout', required=True, type=click.Choice(['ecgsheet', 'binder']))
@click.option('--img-height', default=100, type=int)
@click.option('--enhance', is_flag=True, default=False)
@click.option('--enhance-decision-value', default=220, type=int)
def main(in_file, output_dir, input_layout, img_height, enhance, enhance_decision_value):
    
    os.makedirs(output_dir, exist_ok=True)

    input_img = cv2.imread(str(in_file))  # load ECG image

    if input_layout == 'ecgsheet':
        ecg_img_data = ecgsheet_to_ecgimgdata(input_img, img_height)  # 3D numpy array containing extracted images for 12 ECG leads
    elif input_layout == 'binder':
        ecg_img_data = binder_to_ecgimgdata(input_img, img_height, enhance, enhance_decision_value)

    ecg_leads_grid_savepath = os.path.join(output_dir, in_file.stem + '.png')
    ecg_leads_grid = data_utils.generate_ecg_leads_grid(ecg_img_data)
    Image.fromarray(ecg_leads_grid).convert('L').save(ecg_leads_grid_savepath)


if __name__ == '__main__':
    load_dotenv(find_dotenv())  # load environment variables set in .env file
    main()

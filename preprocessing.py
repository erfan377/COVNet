import os
from argparse import ArgumentParser
from tqdm import tqdm
from shutil import copyfile
import shlex
from subprocess import Popen, DEVNULL
from pathlib import Path

import pydicom
import mdai

import pandas as pd
import numpy as np

import cv2
from scipy import ndimage
from skimage.exposure import equalize_hist
from skimage.transform import resize
from PIL import Image
from collections import defaultdict
import shutil 

import torch
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from constants import *
from utils import make_dirs

MIP_STEP = 5
def build_dicom_df(data_dir):
    """ Extracts dicoms data and saves to df

    Keyword arguments:
    data_dir -- directory to look for dicoms
    """
    df = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.dcm'):

                ds = pydicom.dcmread(os.path.join(root, f))
                if ds.SOPClassUID != CT_CODE:  # remove non-cts
                    continue
                df.append({
                    'path': os.path.join(root, f),
                    'series_instance_uid': ds.SeriesInstanceUID,
                    'rows': ds.Rows,
                    'columns': ds.Columns,
                    'bits_stored': ds.BitsStored,
                    'window_width': ds.WindowWidth,
                    'window_center': ds.WindowCenter,
                    'rescale_intercept': ds.get('RescaleIntercept', 0),
                    'rescale_slope': ds.get('RescaleSlope', 1),
                    'slice_thickness': ds.get('SliceThickness', None),
                    'image_position_patient_x': ds.ImagePositionPatient[0],
                    'image_position_patient_y': ds.ImagePositionPatient[1],
                    'image_position_patient_z': ds.ImagePositionPatient[2],
                    'image_orientation_patient_row_x': ds.ImageOrientationPatient[0],
                    'image_orientation_patient_row_y': ds.ImageOrientationPatient[1],
                    'image_orientation_patient_row_z': ds.ImageOrientationPatient[2],
                    'image_orientation_patient_col_x': ds.ImageOrientationPatient[3],
                    'image_orientation_patient_col_y': ds.ImageOrientationPatient[4],
                    'image_orientation_patient_col_z': ds.ImageOrientationPatient[5],
                    'slice_location': ds.get('SliceLocation', None),
                    'pixel_spacing_y': ds.PixelSpacing[0],
                    'pixel_spacing_x': ds.PixelSpacing[1],
                    'sop_class_uid': ds.SOPClassUID,
                    'sop_instance_uid': ds.SOPInstanceUID,
                    'mask_non_empty': 0
                })

    df = pd.DataFrame(df)

    # remove series with less than x slices
    ct = df.groupby('series_instance_uid', axis=0, group_keys=False,
                    as_index=False).filter(lambda x: len(x) > MIN_SLICE).reset_index()

    sort_df = []
    for key, std in ct.groupby('series_instance_uid', as_index=False):

        # only keep compatible images
        std = std[
            (std['rows'] == std['rows'].mode().item()) &
            (std['columns'] == std['columns'].mode().item()) &
            (std['image_orientation_patient_row_x'] == std['image_orientation_patient_row_x'].mode().item()) &
            (std['image_orientation_patient_row_y'] == std['image_orientation_patient_row_y'].mode().item()) &
            (std['image_orientation_patient_row_z'] == std['image_orientation_patient_row_z'].mode().item()) &
            (std['image_orientation_patient_col_x'] == std['image_orientation_patient_col_x'].mode().item()) &
            (std['image_orientation_patient_col_y'] == std['image_orientation_patient_col_y'].mode().item()) &
            (std['image_orientation_patient_col_z'] == std['image_orientation_patient_col_z'].mode().item())]

        std = std.sort_values(by='image_position_patient_z').reset_index()
        sort_df.append(std)
    df = pd.concat(sort_df, ignore_index=True)
    df = df.drop(columns=['index', 'level_0'])
    
    return df


def process_jsons(df, data_dir):
    """ Extracts labels from jsons

    Keyword arguments:
    df -- the main dataframe to be updated
    data_dir -- directory to look for jsons
    """
    label_dict = defaultdict(list)
    df['label_count'] = 0
    df['label_path'] = ''
    df['label_details'] = np.empty((len(df), 0)).tolist()

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                # Load labels from json
                labels = mdai.common_utils.json_to_dataframe(
                    os.path.join(root, f))

                for index, row in labels['annotations'].iterrows():
                    sop_id = row['SOPInstanceUID']
                    label_name = row['labelName']
                    # Filter unnecessary labels and update df
                    if (label_name in ALLOWED_LABELS and 
                        sop_id in df['sop_instance_uid'].values and
                        row['data'] != None):
                        label_dict[sop_id].append([label_name, row['data']['vertices']])

    for instance_id, data in label_dict.items():
        idx = df[df['sop_instance_uid'] == instance_id].index.item()
        df.at[idx, 'label_count'] = len(data)
        df.at[idx, 'label_details'] = data
    
    return df, label_dict


def get_minimal_transform(img, threshold=125, mode='largest'):
    """Transformation to rotate and crop DICOM images to the smallest bounding box.

    Keyword arguments:
    img -- input image (h x w)
    threshold -- threshold for the binary mask
    mode -- largest or all contours
    """

    assert(mode in ['largest', 'all'])

    # convert to black/white
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # convert to uint8 to work with the contour fn
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if mode == 'largest':
        areas = [cv2.contourArea(contour) for contour in contours]
        order = np.flip(np.argsort(areas))
        contour = contours[order[0]]

    elif mode == 'all':
        stacked_contours = np.vstack(contours)
        contour = cv2.convexHull(stacked_contours)

    # find the smallest rotated rect for the contour
    rect = cv2.minAreaRect(contour)
    center, (w, h), angle = rect

    # do not rotate the image
    width = np.ceil(w).astype(np.int32)
    height = np.ceil(h).astype(np.int32)
    if angle <= -45:
        angle += 90
    elif angle >= 45:
        angle -= 90
        height = np.ceil(w).astype(int)
        width = np.ceil(h).astype(int)

    # rotate
    theta = angle * np.pi / 180  # convert to rad
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))

    # translate
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x],
                        [v_x[1], v_y[1], s_y]])

    return mapping, width, height


def minimize(img, mapping, width, height):
    """Rotate and crop DICOM images to the smallest rotated bounding box.

    Keyword arguments:
    img -- input image
    mapping -- transformation to apply
    width -- width of the output image
    height -- height of the output image
    """
    return cv2.warpAffine(img, mapping, (width, height),
                          flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def load_img(path):
    """Generic image loader.

      Keyword arguments:
      path -- path to an JPG or DCM image
    """
    print('path', path)
    ext = path[-3:].lower()
    assert(ext in ['dcm', 'jpg', 'dic'])

    if path[-3:].lower() in ['dcm', 'dic']:
        ds = pydicom.dcmread(path, force=True)
        img = ds.pixel_array
        # squeeze between 0 and 1 and invert if needed
        if img.max() > 0 or img.min() < 0: # avoid divide by 0
            img = (img - img.min()) / (img.max() - img.min())
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            img = 1 - img

    elif path[-3:].lower() == 'jpg':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = (img - img.min()) / (img.max() - img.min())

    return img


def blend(img, mask, dilation=5):
    """Apply a dilated mask to the image.

      Keyword arguments:
      img -- a torch tensor (h x w)
      mask -- a binary mask (h x w)
      dilation -- number of pixels to dilate the mask with
    """
    
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    mask = TF.resize(mask, size=img.shape,
                     interpolation=TF.InterpolationMode.NEAREST)

    mask = mask.squeeze().numpy()
    mask = ndimage.binary_dilation(mask, iterations=dilation)

    # mask
    img = np.where(mask > 0, img, mask)

    return img


def preprocess(path, output_size=(320, 320)):
    """Preprocess pipeline for different sources of samples.
        Keyword arguments:
        path -- path to an JPG or DCM image
        output_size -- dimensions of the preprocessed image
    """
    img = load_img(path)

    # remove useless "padding"
    mapping, width, height = get_minimal_transform(
        img, threshold=0.5, mode='largest')
    img = minimize(img, mapping, width, height)

    # histogram equalization
    img = equalize_hist(img)

    # generate the lung mask and blend the img with the mask
    mask = generate(img)
    img = blend(img, mask)

    # minimize the result
    mapping, width, height = get_minimal_transform(
        img * 255, threshold=0, mode='all')
    img = minimize(img, mapping, width, height)

    # to tensor and resize
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    img = TF.resize(img, size=output_size, interpolation=Image.BILINEAR)

    return img


def generate_mask(path, output):
    """ 
    Runs lung segmentation model on folder or file
    """
    cmd = f'lungmask \'{path}\' {output}'
    split_cmd = shlex.split(cmd)
    p = Popen(split_cmd)
    p.wait()
    return load_img(output)


def process_label(data, size):
    """ Processes the labels for each dicom

        Keyword arguments:
        vertices -- coordinates of labels
        size -- image size
    """
    label = np.zeros(size)
    # Fill polygon
    for _, vertices in data:
        cv2.fillPoly(label, np.int32([vertices]), color=255)
    label = np.expand_dims(label, axis=2)

    # HWC to CHW
    label = label.transpose((2, 0, 1))
    if label.max() > 1:
        label = label / 255

    return torch.from_numpy(label).type(torch.FloatTensor)


def preprocess_mask(img, mask, output_size=(512, 512)):
    """ Pipeline for processesing the image

      Keyword arguments:
      img -- loaded dicom image
      mask -- lung segmentation mask
      output_size -- In case we need to resize dicom files
    """
    img = blend(img, mask)

    # to tensor and resize
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    img = TF.resize(img, size=output_size,
                    interpolation=TF.InterpolationMode.BILINEAR)
    return img


def save_update_torch(df, file, save_dir, column, instance_id):
    """
    Saves torch files and updates dataframe with new path
    """
    idx = df[df['sop_instance_uid'] == instance_id].index.item()
    name = f'{idx}-{instance_id}.pt'
    df.at[idx, column] = os.path.join(save_dir, name)
    torch.save(file, os.path.join(save_dir, name))
    
def covnet_process(imgs, labels, rescaleSlope, rescaleIntercept):
    data_mip = []
    label_mip = []
    for i in range(0, len(imgs), MIP_STEP):
        if i + MIP_STEP < len(imgs):
            imgs_sub = torch.stack(imgs[i:i+MIP_STEP])
            labels_sub = torch.stack(labels[i:i+MIP_STEP])
        else:
            imgs_sub = torch.stack(imgs[i:])
            labels_sub = torch.stack(labels[i:])
        img = torch.max(imgs_sub, dim=0).values
        label = torch.max(labels_sub, dim=0).values
        img = TF.resize(img, size=(224,224))
        label = TF.resize(label, size=(224,224))
        img = torch.clip(img * rescaleSlope + rescaleIntercept, (-600 - 750), (-600 + 750))
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        img = torch.nan_to_num(img, nan=0.0)
        data_mip.append(img)
        label_mip.append(label)
        
    return torch.stack(data_mip), torch.stack(label_mip)
        


def midrc_parse(df, labels, output_dir, volume=True):
    """ Processes midrc dicoms and labels
    Keyword arguments:
      df -- processed dataframe of dicoms
      labels -- Dict containing the label coordinates
      output_dir -- directory to save torch files
    """
    dicom_dir = os.path.join(output_dir, 'dicoms')
    label_dir = os.path.join(output_dir, 'labels')
    temp_dir = os.path.join(output_dir, 'temp')
    make_dirs(dicom_dir, label_dir, temp_dir)
    unique_set = set()

    for main_indx, row in tqdm(df.iterrows(), total=len(df)):
        dir_path = os.path.dirname(row['path'])
        series_uid = row['series_instance_uid']

        # apply lung segmentation as a volume
        if volume and series_uid not in unique_set:
            
            # generate the lung mask and blend the img with the mask
            output_mask_path = os.path.join(temp_dir, series_uid + '.dcm')
            masks = generate_mask(dir_path, output_mask_path)
            df_instance = df[df['series_instance_uid'] == series_uid].reset_index()
            unique_set.add(series_uid)
            
            data_list = []
            label_list = []
            first_instance_id = df_instance.loc[0, 'sop_instance_uid']
            rescaleSlope = df_instance.loc[0, 'rescale_slope']
            rescaleIntercept = df_instance.loc[0, 'rescale_intercept']
            # Process the SOP Instances from the lung volume
            for idx in range(len(df_instance)):
                instance_path = df_instance.loc[idx, 'path']
                instance_id = df_instance.loc[idx, 'sop_instance_uid']
                img = load_img(instance_path)
                
                img = preprocess_mask(img, masks[idx])
                if img.max() > 0:  # Determines which files have empty lung
                    df.at[df['sop_instance_uid'] ==
                          instance_id, 'mask_non_empty'] = 1
                data_list.append(img)

                # Check if there's a label for instance
                label = torch.zeros_like(img)
                if instance_id in labels:
                    label = process_label(labels[instance_id], img.shape[1:])
                label_list.append(label)

            img, label = covnet_process(data_list, label_list, rescaleSlope, rescaleIntercept)
            save_update_torch(df, img, dicom_dir, 'path', first_instance_id)
            save_update_torch(df, label, label_dir, 'label_path', first_instance_id)

    shutil.rmtree(temp_dir)
    return df

def cli_main():
    # seed
    pl.seed_everything(365)

    # parse cmd-line arguments
    parser = ArgumentParser()
    parser.add_argument('--ricord', action='store_true', default=False)
    parser.add_argument('--midrc', action='store_true', default=False)
    parser.add_argument('--volume', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Path to input files')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    assert(args.ricord or args.midrc)

    # parse ricord
    if args.ricord:
        df = pd.read_csv(os.path.join(args.data_dir, 'ricord.csv'), sep=';')

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            path = os.path.join(args.data_dir, row['path'])
            img = preprocess(path, args.data_dir)

            # write to disk
            output_path = os.path.join(
                args.output_dir, row['sop_instance_uid'] + '.pt')
            row['path'] = output_path
            torch.save(img, output_path)

    # parse midrc
    if args.midrc:

        df = build_dicom_df(args.data_dir)
        df, labels = process_jsons(df, args.data_dir)
        df = midrc_parse(df, labels, args.output_dir, args.volume)
        df = df[df['label_path'] != '']
        # save CSV
        df.to_csv(os.path.join(args.output_dir, 
                                          'metadata.csv'), index=True)


if __name__ == '__main__':
    cli_main()

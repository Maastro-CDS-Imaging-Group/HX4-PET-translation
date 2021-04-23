import warnings
import glob
import math
from os.path import join
from datetime import time, datetime

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator
from skimage.draw import polygon
import SimpleITK as sitk
import pydicom as pdcm
from pydicom.tag import Tag

from viz_utils import NdimageVisualizer



def read_pet_dicoms_to_sitk(dicom_series_dir):
    # Read slices
    dicom_filepaths = sorted(glob.glob(f"{dicom_series_dir}/*"))
    dicom_filepaths = dicom_filepaths[1:] # Ignore the 1st .dcm file (not a slice)
    pet_slices = [pdcm.read_file(dcm) for dcm in dicom_filepaths]
    pet_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Convert to numpy array of SUV values
    patient_weight = float(pet_slices[0].PatientWeight)
    pet_np = get_physical_values_pt(pet_slices, patient_weight, dtype=np.float32)
    pet_np = pet_np.transpose(2,0,1) # HWD to DHW

    # Convert to sitk
    slice_spacing = pet_slices[1].ImagePositionPatient[2] - pet_slices[0].ImagePositionPatient[2]
    pixel_spacing = np.asarray([pet_slices[0].PixelSpacing[0], pet_slices[0].PixelSpacing[1], slice_spacing])
    image_position_patient = [float(k) for k in pet_slices[0].ImagePositionPatient]
    pet_sitk = np2sitk(pet_np, pixel_spacing, image_position_patient)
    return pet_sitk


def read_ct_dicoms_to_sitk(dicom_series_dir, rtstruct_filepath=None):
    # Read slices
    dicom_filepaths = sorted(glob.glob(f"{dicom_series_dir}/*"))
    ct_slices = [pdcm.read_file(dcm) for dcm in dicom_filepaths]
    ct_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Convert to numpy array
    ct_np = get_physical_values_ct(ct_slices, dtype=np.float32)  # HWD

    slice_spacing = ct_slices[1].ImagePositionPatient[2] - ct_slices[0].ImagePositionPatient[2]
    pixel_spacing = np.asarray([ct_slices[0].PixelSpacing[0], ct_slices[0].PixelSpacing[1], slice_spacing])
    image_position_patient = [float(k) for k in ct_slices[0].ImagePositionPatient]

    # RTstruct given ?
    if rtstruct_filepath is None:
        # Convert CT to sitk and return
        ct_np = ct_np.transpose(2,0,1) # HWD to DHW
        ct_sitk = np2sitk(ct_np, pixel_spacing, image_position_patient)
        return ct_sitk

    else:
        # Convert RTstruct to masks (binary numpy arrays)
        axial_positions = np.asarray([k.ImagePositionPatient[2] for k in ct_slices])
        masks = get_masks(rtstruct_filepath,
                          labels=['GTVp1', 'BODY'],
                          image_position_patient=image_position_patient,
                          axial_positions=axial_positions,
                          pixel_spacing=pixel_spacing,
                          shape=ct_np.shape,
                          dtype=np.int8)

        # Convert masks to sitk
        for k in masks.keys():
            masks[k] = masks[k].transpose(2,0,1)  # HWD to DHW
            masks[k] = np2sitk(masks[k], pixel_spacing, image_position_patient)

        # Convert CT to sitk
        ct_np = ct_np.transpose(2,0,1)  # HWD to DHW
        ct_sitk = np2sitk(ct_np, pixel_spacing, image_position_patient)
        return ct_sitk, masks
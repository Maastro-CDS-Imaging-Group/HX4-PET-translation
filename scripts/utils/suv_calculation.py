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



def get_physical_values_pt(slices, patient_weight, dtype=np.float32):
    s = slices[0]
    units = s.Units
    if units == 'BQML':
        acquisition_datetime = datetime.strptime(
            s[Tag(0x00080022)].value + s[Tag(0x00080032)].value.split('.')[0],
            "%Y%m%d%H%M%S")
        serie_datetime = datetime.strptime(
            s[Tag(0x00080021)].value + s[Tag(0x00080031)].value.split('.')[0],
            "%Y%m%d%H%M%S")

        try:
            if (serie_datetime <= acquisition_datetime) and (
                    serie_datetime > datetime(1950, 1, 1)):
                scan_datetime = serie_datetime
            else:
                scan_datetime_value = s[Tag(0x0009100d)].value
                if isinstance(scan_datetime_value, bytes):
                    scan_datetime_str = scan_datetime_value.decode(
                        "utf-8").split('.')[0]
                elif isinstance(scan_datetime_value, str):
                    scan_datetime_str = scan_datetime_value.split('.')[0]
                else:
                    raise ValueError(
                        "The value of scandatetime is not handled")
                scan_datetime = datetime.strptime(scan_datetime_str,
                                                  "%Y%m%d%H%M%S")

            start_time_str = s.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            start_time = time(int(start_time_str[0:2]),
                              int(start_time_str[2:4]),
                              int(start_time_str[4:6]))
            start_datetime = datetime.combine(scan_datetime.date(), start_time)
            decay_time = (scan_datetime - start_datetime).total_seconds()
        except KeyError:
            warnings.warn("Estimation of time decay for SUV"
                          " computation from average parameters")
            decay_time = 1.75 * 3600  # From Martin's code
        return get_suv_from_bqml(slices,
                                 decay_time,
                                 patient_weight,
                                 dtype=dtype)

    elif units == 'CNTS':
        return get_suv_philips(slices, dtype=dtype)
    else:
        raise ValueError('The {} units is not handled'.format(units))


def get_suv_philips(slices, dtype=np.float32):
    image = list()
    suv_scale_factor_tag = Tag(0x70531000)
    for s in slices:
        im = (float(s.RescaleSlope) * s.pixel_array +
              float(s.RescaleIntercept)) * float(s[suv_scale_factor_tag].value)
        image.append(im)
    return np.stack(image, axis=-1).astype(dtype)


def get_suv_from_bqml(slices, decay_time, patient_weight, dtype=np.float32):
    # Get SUV from raw PET
    image = list()
    for s in slices:
        pet = float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
        half_life = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        total_dose = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        decay = 2**(-decay_time / half_life)
        actual_activity = total_dose * decay

        im = pet * patient_weight * 1000 / actual_activity
        image.append(im)
    return np.stack(image, axis=-1).astype(dtype)

"""
Auto generate body masks for unregistered ldCT and HX4-PET in the training set
using thresholding and morphological ops on ldCT.

Can do this on-the-fly during training, but would tremendously slow down the training.
Therefore, this is actually an implementation detail for speed-up.


Run this script after the data preparation script.


Data dir structure:

	HX4-PET-Translation (root)
		|- Original (unchanged)
		|
		|- Processed
    		|- val (unchanged)
            |
            |-train
                |- PB048 - ..., ldct_body.nrrd
                |- ...

"""

import os
import glob
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk

from utils.auto_mask_generation import get_body_mask
from utils.io import write_sitk_to_nrrd


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
HU_THRESHOLD = -300


def main():

    data_dir = f"{DATA_ROOT_DIR}/Processed/train"
    patient_ids = sorted(os.listdir(data_dir))

    for p_id in tqdm(patient_ids):
        # print(f"{p_id}: ")

        ldct_sitk = sitk.ReadImage(f"{data_dir}/{p_id}/ldct.nrrd")

        # Generate body mask
        ldct_np = sitk.GetArrayFromImage(ldct_sitk)
        body_mask_np = get_body_mask(ldct_np, HU_THRESHOLD)
        body_mask_sitk = sitk.GetImageFromArray(body_mask_np)
        body_mask_sitk.CopyInformation(ldct_sitk)

        # Write to NRRD
        output_path = f"{data_dir}/{p_id}/ldct_body.nrrd"
        write_sitk_to_nrrd(body_mask_sitk, output_path)



if __name__ == '__main__':
	main()
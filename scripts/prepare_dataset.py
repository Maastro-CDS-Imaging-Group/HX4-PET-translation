"""
Data dir structure:

	HX4-PET-Translation (root)
		|- Original (input)
		|	|- N010 - FDG (PET, CT), HX4 (PET, CT), reg_HX4_to_FDG (PET, CT), aorta_CT_HX4_def.mat
		|	|- ...
		|
		|- Processed (output)
    		|- val
            |	|- N010 - fdg_pet.nrrd, pct.nrrd, hx4_pet.nrrd, ldct.nrrd, hx4_pet_reg.nrrd, ldct_reg.nrrd,
    		|	|         pct_body.nrrd, pct_gtv.nrrd
    		|	|- ...
            |
            |-train
                |- PB048 - ...
                |- ...

"""

import os
import glob
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk

from utils.io import read_pet_dicoms_to_sitk, read_ct_dicoms_to_sitk, write_sitk_to_nrrd
from utils.cropping_resampling import crop_and_resample_pet_ct, crop_pet_ct_pairs_to_common_roi
from utils.sitk_utils import apply_body_mask


DATA_ROOT_DIR = "/workspace/data/Chinmay/Datasets/HX4-PET-Translation"
RTSTRUCT_ROI_NAMES_FILE = "../generated_metadata/selected_rtstruct_roi_names.csv"
RESAMPLE_SPACING = (1.0, 1.0, 3.0)

POSSIBLE_BODY_ROI_NAMES = ['BODY', 'bodycontour']


def main():
    source_dir = f"{DATA_ROOT_DIR}/Original"
    target_dir = f"{DATA_ROOT_DIR}/Processed"

    os.makedirs(f"{target_dir}/train", exist_ok=True)
    os.makedirs(f"{target_dir}/val", exist_ok=True)

    patient_ids = sorted(os.listdir(source_dir))
    rtstruct_roi_info = pd.read_csv(RTSTRUCT_ROI_NAMES_FILE, index_col=0)
    rtstruct_roi_info = rtstruct_roi_info.to_dict(orient='index')

    for p_id in tqdm(patient_ids):
        print(f"{p_id}: ")

        # -------------------
        # Read images to sitk
        print("\tReading images ... ")

        # FDG-PET
        dicom_series_dir = f"{source_dir}/{p_id}/FDG/PT"
        fdg_pet_sitk = read_pet_dicoms_to_sitk(dicom_series_dir, p_id)

        # pCT and masks
        dicom_series_dir = f"{source_dir}/{p_id}/FDG/CT"
        rtstruct_filepath = glob.glob(f"{source_dir}/{p_id}/FDG/RTSTRUCT/*")[0]
        pct_sitk, masks = read_ct_dicoms_to_sitk(dicom_series_dir, p_id, rtstruct_filepath, rtstruct_roi_info)

        # HX4-PET
        dicom_series_dir = f"{source_dir}/{p_id}/HX4/PT"
        hx4_pet_sitk = read_pet_dicoms_to_sitk(dicom_series_dir, p_id)

        # ldCT
        dicom_series_dir = f"{source_dir}/{p_id}/HX4/CT"
        ldct_sitk = read_ct_dicoms_to_sitk(dicom_series_dir, p_id)

        # HX4-PET-reg
        mhd_filepath = f"{source_dir}/{p_id}/reg_HX4_to_FDG/image_transformed/result.mhd"
        hx4_pet_reg_sitk = sitk.ReadImage(mhd_filepath, sitk.sitkFloat32)

        # ldCT-reg
        mhd_filepath = f"{source_dir}/{p_id}/reg_HX4_to_FDG/image_registered/result.1.mhd"
        ldct_reg_sitk = sitk.ReadImage(mhd_filepath, sitk.sitkFloat32)


        # -----------------------------------------------------------------
        # For each PET-CT pair, crop and resample to common ROI and spacing
        print("\tCropping and resampling ... ")

        # FDG-PET, pCT and masks
        fdg_pet_sitk, pct_sitk, masks = crop_and_resample_pet_ct(fdg_pet_sitk, pct_sitk, masks=masks, resample_spacing=RESAMPLE_SPACING)

        # HX4-PET and ldCT
        hx4_pet_sitk, ldct_sitk = crop_and_resample_pet_ct(hx4_pet_sitk, ldct_sitk, resample_spacing=RESAMPLE_SPACING)

        # HX4-PET-reg and ldCT-reg
        hx4_pet_reg_sitk, ldct_reg_sitk = crop_and_resample_pet_ct(hx4_pet_reg_sitk, ldct_reg_sitk, resample_spacing=RESAMPLE_SPACING)


        # ---------------------------------------------------------------------
        # Crop to common ROI, across FDG-PET/pCT and HX4-PET-reg/ldCT-reg pairs
        print("\tCropping further ... ")

        fdg_pet_sitk, pct_sitk, hx4_pet_reg_sitk, ldct_reg_sitk, masks = crop_pet_ct_pairs_to_common_roi(
            fdg_pet_sitk, pct_sitk, hx4_pet_reg_sitk, ldct_reg_sitk, masks)


        # ------------------------------
        # Write processed images to NRRD
        print("\tWriting images to NRRD ... ")

        if p_id.startswith('N'):  # Nitro patients in val set
            dataset_split = 'val'
        elif p_id.startswith('PB'):  # PET Boost patients in trian set
            dataset_split = 'train'

        os.makedirs(f"{target_dir}/{dataset_split}/{p_id}", exist_ok=True)

        # FDG-PET and pCT
        write_sitk_to_nrrd(fdg_pet_sitk, f"{target_dir}/{dataset_split}/{p_id}/fdg_pet.nrrd")
        write_sitk_to_nrrd(pct_sitk, f"{target_dir}/{dataset_split}/{p_id}/pct.nrrd")

        # HX4-PET and ldCT
        write_sitk_to_nrrd(hx4_pet_sitk, f"{target_dir}/{p_id}/hx4_pet.nrrd")
        write_sitk_to_nrrd(ldct_sitk, f"{target_dir}/{p_id}/ldct.nrrd")

        # HX4-PET-reg and ldCT-reg
        write_sitk_to_nrrd(hx4_pet_reg_sitk, f"{target_dir}/{dataset_split}/{p_id}/hx4_pet_reg.nrrd")
        write_sitk_to_nrrd(ldct_reg_sitk, f"{target_dir}/{dataset_split}/{p_id}/ldct_reg.nrrd")

        # Masks
        gtv_roi_name = rtstruct_roi_info[p_id]['gtv-roi-name']
        gtv_mask = masks[gtv_roi_name]
        write_sitk_to_nrrd(gtv_mask, f"{target_dir}/{dataset_split}/{p_id}/pct_gtv.nrrd")

        body_roi_name = rtstruct_roi_info[p_id]['body-roi-name']  # Patient N046 doesn't have a body mask
        if body_roi_name in POSSIBLE_BODY_ROI_NAMES:
            body_mask = masks[body_roi_name]
            write_sitk_to_nrrd(body_mask, f"{target_dir}/{dataset_split}/{p_id}/pct_body.nrrd")

        print("\tComplete")

if __name__ == '__main__':
    main()
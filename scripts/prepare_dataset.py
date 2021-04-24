"""
Data dir structure:

	HX4-PET-Translation (root)
		|- Original (input)
		|	|- N010 - FDG (PET, CT), HX4 (PET, CT), reg_HX4_to_FDG (PET, CT), aorta_CT_HX4_def.mat
		|	|- ...
		|
		|- Processed (output)
			|- N010 - fdg_pet.nrrd, pct.nrrd, hx4_pet.nrrd, ldct.nrrd, hx4_pet_reg.nrrd, ldct_reg.nrrd, 
			|         pct_body.nrrd, pct_gtv.nrrd, aorta_ct_hx4_def.nrrd 
			|- ...

TODO: Write this script

"""

import os
import glob
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk

from utils.io import read_pet_dicoms_to_sitk, read_ct_dicoms_to_sitk, write_sitk_to_nrrd
from utils.cropping_resampling import crop_and_resample_pet_ct, crop_pet_ct_pairs_to_common_roi


DATA_ROOT_DIR = "/workspace/data/Chinmay/Datasets/HX4-PET-Translation"
RTSTRUCT_ROI_NAMES_FILE = "../generated_matadata/selected_rtstruct_roi_names.csv"

POSSIBLE_BODY_ROI_NAMES = ['BODY', 'bodycontour']


def main():
    source_dir = f"{DATA_ROOT_DIR}/Original"
    target_dir = f"{DATA_ROOT_DIR}/Processed"
    
    patient_ids = sorted(os.listdit(source_dir))
    rtstruct_roi_info = pd.read_csv(RTSTRUCT_ROI_NAMES_FILE).to_dict()

    for p_id in tqdm(patient_ids):

        # -------------------
        # Read images to sitk

        # FDG-PET
        dicom_series_dir = f"{source_dir}/{p_id}/FDG/PT"
        fdg_pet_sitk = read_pet_dicoms_to_sitk(dicom_series_dir, p_id)

        # pCT and masks
        dicom_series_dir = f"{source_dir}/{p_id}/FDG/CT"
        rtstruct_filepath = glob.glob(f"{source_dir}/{p_id}/FDG/RTSTRUCT/*")[0]
        pct_sitk, masks = read_ct_dicoms_to_sitk(dicom_series_dir, p_id, rtstruct_filepath, rtstruct_roi_info)

        # HX4-PET
        dicom_series_dir = f"{source_dir}/{p_id}/HX4/PT"
        hx4_pet_sitk = read_pet_dicoms_to_sitk(dicom_series_dir)

        # ldCT
        dicom_series_dir = f"{source_dir}/{p_id}/HX4/CT"
        ldct_sitk = read_ct_dicoms_to_sitk(dicom_series_dir)

        # HX4-PET-reg
        mhd_filepath = f"{source_dir}/{p_id}/reg_HX4_to_FDG/image_transformed/result.mhd"
        hx4_pet_reg_sitk = sitk.ReadImage(mhd_filepath, sitk.sitkFloat32)

        # ldCT-reg
        mhd_filepath = f"{source_dir}/{p_id}/reg_HX4_to_FDG/image_registered/result.1.mhd"
        ldct_reg_sitk = sitk.ReadImage(mhd_filepath, sitk.sitkFloat32)


        # -----------------------------------------------------------------
        # For each PET-CT pair, crop and resample to common ROI and spacing

        # FDG-PET, pCT and masks
        fdg_pet_sitk, pct_sitk, masks = crop_and_resample_pet_ct(fdg_pet_sitk, pct_sitk, masks)

        # HX4-PET and ldCT
        hx4_pet_sitk, ldct_sitk = crop_and_resample_pet_ct(hx4_pet_sitk, ldct_sitk)

        # HX4-PET-reg and ldCT-reg
        hx4_pet_reg_sitk, ldct_reg_sitk = crop_and_resample_pet_ct(hx4_pet_reg_sitk, ldct_reg_sitk)


        # ---------------------------------------------------------------------
        # Crop to common ROI, across FDG-PET/pCT and HX4-PET-reg/ldCT-reg pairs
        fdg_pet_sitk, pct_sitk, hx4_pet_reg_sitk, ldct_reg_sitk, masks = crop_pet_ct_pairs_to_common_roi(
            fdg_pet_sitk, pct_sitk, hx4_pet_reg_sitk, ldct_reg_sitk, masks
            )


        # ------------------------------------
        # Apply body mask to pCT, if available
        
        body_roi_name = rtstruct_roi_info[p_id]['body-roi-name']
        
        if body_roi_name in POSSIBLE_BODY_ROI_NAMES:  # Patient N046 doesn't have a body mask
            pct_np = sitk.GetArrayFromImage(pct_sitk)
            body_mask_np = sitk.GetArrayFromImage(masks[body_roi_name])

            # Apply mask and set out-of-mask region HU as -1000 (air) 
            pct_body_np = pct_np * body_mask_np
            pct_body_np = pct_body_np + (body_mask_np == 0).astype(np.uint8) * (-1000)

            pct_body_sitk = sitk.GetImageFromArray(pct_body_np)
            pct_body_sitk.CopyInformation(pct_sitk)
            pct_sitk = pct_body_sitk


        # ------------------------------
        # Write processed images to NRRD

        # FDG-PET and pCT
        write_sitk_to_nrrd(fdg_pet_sitk, f"{target_dir}/{p_id}/fdg_pet.nrrd")
        write_sitk_to_nrrd(pct_sitk, f"{target_dir}/{p_id}/pct.nrrd")

        # HX4-PET and ldCT
        write_sitk_to_nrrd(hx4_pet_sitk, f"{target_dir}/{p_id}/hx4_pet.nrrd")
        write_sitk_to_nrrd(ldct_sitk, f"{target_dir}/{p_id}/ldct.nrrd")

        # HX4-PET-reg and ldCT-reg
        write_sitk_to_nrrd(hx4_pet_reg_sitk, f"{target_dir}/{p_id}/hx4_pet_reg.nrrd")
        write_sitk_to_nrrd(ldct_reg_sitk, f"{target_dir}/{p_id}/ldct_reg.nrrd")

        # Masks
        gtv_roi_name = rtstruct_roi_info[p_id]['gtv-roi-name']
        gtv_mask = masks[gtv_roi_name]
        write_sitk_to_nrrd(gtv_mask, f"{target_dir}/{p_id}/pct_gtv.nrrd")

        body_roi_name = rtstruct_roi_info[p_id]['body-roi-name']  # Patient N046 doesn't have a body mask
        if body_roi_name in POSSIBLE_BODY_ROI_NAMES:
            body_mask = masks[body_roi_name]
            write_sitk_to_nrrd(body_mask, f"{target_dir}/{p_id}/pct_body.nrrd")


if __name__ == '__main__':
    main()
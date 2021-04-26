"""
Info per patient:
    1. Spacing of all NRRDs
    2. Origin " "
    3. Array size " "
    4. Intensity range " "

Stats across all patients: 
    1. FDG-PET: Min and max SUV
    2. HX4-PET: Min and max SUV
    3. HX4-PET-reg: Min and max SUV
    4. FDG-PET, pCT, HX4-PET-reg, ldCT-reg: Common array size  
"""

import os
import glob
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk

from utils.io import read_nrrd_to_sitk


DATA_ROOT_DIR = "/workspace/data/Chinmay/Datasets/HX4-PET-Translation"
OUTPUT_FILE = "../generated_metadata/processed_dataset_info.txt"


def main():
    data_dir = f"{DATA_ROOT_DIR}/Processed"
    patient_ids = sorted(os.listdir(data_dir))

    global_suv_mins = {'fdg_pet': 999, 'hx4_pet': 999, 'hx4_pet_reg': 999}
    global_suv_maxs = {'fdg_pet': -999, 'hx4_pet': -999, 'hx4_pet_reg': -999}
    global_paired_image_sizes = {}

    ofile = open(OUTPUT_FILE, 'w')

    for p_id in tqdm(patient_ids):
        ofile.write(f"{p_id}: \n")

        sitk_images = {}

        # FDG-PET and pCT
        sitk_images['fdg_pet'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/fdg_pet.nrrd")
        sitk_images['pct'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/pct.nrrd")

        # HX4-PET and ldCT
        sitk_images['hx4_pet'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/hx4_pet.nrrd")
        sitk_images['ldct'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/ldct.nrrd")

        # HX4-PET-reg and ldCT-reg
        sitk_images['hx4_pet_reg'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/hx4_pet_reg.nrrd")
        sitk_images['ldct_reg'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/ldct_reg.nrrd")

        # Masks
        sitk_images['gtv_mask'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/pct_gtv.nrrd")
        
        try:
            sitk_images['body_mask'] = read_nrrd_to_sitk(f"{data_dir}/{p_id}/pct_body.nrrd")
        except RuntimeError:
            print(f"Body mask not available for {p_id}")  # Specifically, N046

        for image_name in sitk_images.keys():

            # Patient-wise image info
            ofile.write(f"\t{image_name}: \n")
            sitk_image = sitk_images[image_name]
            np_image = sitk.GetArrayFromImage(sitk_image).transpose(2,1,0)  # WHD
            
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            array_size = np_image.shape 
            
            ofile.write(f"\t\tOrigin: {origin}\n")
            ofile.write(f"\t\tSpacing: {spacing}\n")
            ofile.write(f"\t\tArray size: {array_size}\n")

            if 'mask' not in image_name:
                intensity_range = (np_image.min(), np_image.max())
                ofile.write(f"\t\tIntensity range: {intensity_range}\n")

            # Global stats
            if any([image_name == k for k in ('fdg_pet', 'pct', 'hx4_pet_reg', 'ldct_reg')]):
                if str(array_size) not in global_paired_image_sizes.keys():
                        global_paired_image_sizes[str(array_size)] = 1
                else: 
                        global_paired_image_sizes[str(array_size)] += 1

            if 'pet' in image_name:
                if np_image.min() < global_suv_mins[image_name]:
                    global_suv_mins[image_name] = np_image.min()
                if np_image.max() > global_suv_maxs[image_name]:
                    global_suv_maxs[image_name] = np_image.max()

    print("Global stats:")
    print(f"\tGloabl paired-image size counts: {global_paired_image_sizes}")
    print(f"\tGlobal SUV mins: {global_suv_mins}")
    print(f"\tGlobal SUV maxs: {global_suv_maxs}")

    ofile.close()


if __name__ == '__main__':
    main()
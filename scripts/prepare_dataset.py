"""
Data dir structure:

	HX4-PET-Translation (root)
		|- Original (input)
		|	|- N010 - FDG (PET, CT), HX4 (PET, CT), reg_HX4_to_FDG (PET, CT), aorta_CT_HX4_def.mat
		|	|- ...
		|
		|- Processed (output)
			|- N010 - fdg_pet.nrrd, pct.nrrd, hx4_pet.nrrd, ldct.nrrd, hx4_pet_reg.nrrd, ldct_reg.nrrd, 
			|         pct_body.nrrd, pct_gtvp1.nrrd, aorta_ct_hx4_def.nrrd 
			|- ...

TODO: Write this script

"""



DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation/Original"
SAMPLE_PATIENT = "N010"


# Read images to sitk

# FDG-PET
dicom_series_dir = f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/FDG/PT"
fdg_pet_sitk = read_pet_dicoms_to_sitk(dicom_series_dir)

# pCT
dicom_series_dir = f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/FDG/CT"
rtstruct_filepath = glob.glob(f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/FDG/RTSTRUCT/*")[0]
pct_sitk, masks = read_ct_dicoms_to_sitk(dicom_series_dir, rtstruct_filepath)

# HX4-PET
dicom_series_dir = f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/HX4/PT"
hx4_pet_sitk = read_pet_dicoms_to_sitk(dicom_series_dir)

# ldCT
dicom_series_dir = f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/HX4/CT"
ldct_sitk = read_ct_dicoms_to_sitk(dicom_series_dir)

# HX4-PET-reg
mhd_filepath = f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/reg_HX4_to_FDG/image_transformed/result.mhd"
hx4_pet_reg_sitk = sitk.ReadImage(mhd_filepath, sitk.sitkFloat32)

# ldCT-reg
mhd_filepath = f"{DATA_ROOT_DIR}/{SAMPLE_PATIENT}/reg_HX4_to_FDG/image_registered/result.1.mhd"
ldct_reg_sitk = sitk.ReadImage(mhd_filepath, sitk.sitkFloat32)



# Crop and resample to common ROIs and spacing

# FDG-PET, pCT and masks
fdg_pet_sitk, pct_sitk = crop_and_resample_pet_ct(fdg_pet_sitk, pct_sitk)

# HX4-PET and ldCT
hx4_pet_sitk, ldct_sitk = crop_and_resample_pet_ct(hx4_pet_sitk, ldct_sitk)

# HX4-PET-reg and ldCT-reg
hx4_pet_reg_sitk, ldct_reg_sitk = crop_and_resample_pet_ct(hx4_pet_reg_sitk, ldct_reg_sitk)


# Crop to common ROI, across FDG-PET/pCT and HX4-PET-reg/ldCT-reg pairs
fdg_pet_sitk, pct_sitk, hx4_pet_reg_sitk, ldct_reg_sitk = crop_pet_ct_pairs_to_common_roi(
	fdg_pet_sitk, pct_sitk, hx4_pet_reg_sitk, ldct_reg_sitk
	)
import os

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
OUTPUT_DIR = "../generated_metadata"


def get_stats(dataset_split='train'):

    print("Split:", dataset_split)

    dataset_split_dir = f"{DATA_ROOT_DIR}/Processed/{dataset_split}"
    patient_ids = sorted(os.listdir(dataset_split_dir))


    fig, axs = plt.subplots(2, 6, figsize=(36, 12))

    fdg_pet_range = IntensityRange()
    pct_range = IntensityRange()
    hx4_pet_range = IntensityRange()
    ldct_range = IntensityRange()
    hx4_pet_reg_range = IntensityRange()
    ldct_reg_range = IntensityRange()

    print("Processing ...")
    for p_id in tqdm(patient_ids):

        # Read images
        patient_dir = f"{dataset_split_dir}/{p_id}"
        fdg_pet, pct, hx4_pet, ldct, hx4_pet_reg, ldct_reg = read_all_images_of_patient(patient_dir)

        # Plot Histogram
        plot_histogram(fdg_pet, axs[0][0], bin_min=-1, bin_max=20, bin_size=0.25, units='SUV', title="FDG-PET")
        plot_histogram(pct, axs[0][1], bin_min=-1050, bin_max=2050, bin_size=20, units='HU', title="pCT")
        plot_histogram(hx4_pet, axs[0][2], bin_min=-1, bin_max=5, bin_size=0.25, units='SUV', title="HX4-PET")
        plot_histogram(ldct, axs[0][3], bin_min=-1050, bin_max=2050, bin_size=20, units='HU', title="ldCT")
        plot_histogram(hx4_pet_reg, axs[0][4], bin_min=-1, bin_max=5, bin_size=0.25, units='SUV', title="HX4-PET-reg")
        plot_histogram(ldct_reg, axs[0][5], bin_min=-1050, bin_max=2050, bin_size=20, units='HU', title="ldCT-reg")

        # Record range
        fdg_pet_range.record_range(fdg_pet)
        pct_range.record_range(pct)
        hx4_pet_range.record_range(hx4_pet)
        ldct_range.record_range(ldct)
        hx4_pet_reg_range.record_range(hx4_pet_reg)
        ldct_reg_range.record_range(ldct_reg)


    # Plot patient vs. range
    fdg_pet_range.plot_ranges(axs[1][0], patient_ids, units='SUV', title="FDG-PET")
    pct_range.plot_ranges(axs[1][1], patient_ids, units='HU', title="pCT")
    hx4_pet_range.plot_ranges(axs[1][2], patient_ids, units='SUV', title="HX4-PET")
    ldct_range.plot_ranges(axs[1][3], patient_ids, units='HU', title="ldCT")
    hx4_pet_reg_range.plot_ranges(axs[1][4], patient_ids, units='SUV', title="HX4-PET-reg")
    ldct_reg_range.plot_ranges(axs[1][5], patient_ids, units='HU', title="ldCT-reg")

    fig.suptitle(f"Intensity statistics - {dataset_split}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/intensity_stats_{dataset_split}.png")
    print("Figure saved")

    # Print global ranges
    print("Global range --")
    print("FDG-PET:", fdg_pet_range.get_global_range())
    print("pCT:", pct_range.get_global_range())
    print("HX4-PET:", hx4_pet_range.get_global_range())
    print("ldCT:", ldct_range.get_global_range())
    print("HX4-PET-reg:", hx4_pet_reg_range.get_global_range())
    print("ldCT-reg:", ldct_reg_range.get_global_range())


class IntensityRange:
    def __init__(self):
        self.mins, self.maxs = [], []

    def record_range(self, image_np):
        self.mins.append(image_np.min())
        self.maxs.append(image_np.max())

    def get_global_range(self):
        return min(self.mins), max(self.maxs)

    def plot_ranges(self, ax, patient_ids, units, title):
        ax.plot(patient_ids, self.mins, 'b', label="Minimum")
        ax.plot(patient_ids, self.maxs, 'r', label="Maximum")
        ax.set_ylabel(units)
        ax.set_xticklabels(patient_ids, rotation=90)
        ax.legend()
        ax.set_title(title)


def plot_histogram(image_np, ax, bin_min, bin_max, bin_size, units, title):
    ax.hist(image_np.flatten(), bins=np.arange(bin_min, bin_max, bin_size), histtype='step')
    ax.set_xlabel(units)
    ax.set_ylabel("voxels")
    ax.set_title(title)


def read_all_images_of_patient(patient_dir):
    fdg_pet = sitk2np(sitk.ReadImage(f"{patient_dir}/fdg_pet.nrrd"))
    pct = sitk2np(sitk.ReadImage(f"{patient_dir}/pct.nrrd"))
    hx4_pet = sitk2np(sitk.ReadImage(f"{patient_dir}/hx4_pet.nrrd"))
    ldct = sitk2np(sitk.ReadImage(f"{patient_dir}/ldct.nrrd"))
    hx4_pet_reg = sitk2np(sitk.ReadImage(f"{patient_dir}/hx4_pet_reg.nrrd"))
    ldct_reg = sitk2np(sitk.ReadImage(f"{patient_dir}/ldct_reg.nrrd"))
    return fdg_pet, pct, hx4_pet, ldct, hx4_pet_reg, ldct_reg


def sitk2np(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)


if __name__ == '__main__':
    get_stats(dataset_split='train')
    get_stats(dataset_split='val')
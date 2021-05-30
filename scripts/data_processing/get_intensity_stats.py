import os

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
OUTPUT_DIR = "../generated_metadata"


def get_stats(dataset_splits=['train']):

    print("Splits:", dataset_splits)

    patient_ids = []
    patient_dirs = []
    for dataset_split in dataset_splits:
        dataset_split_dir = f"{DATA_ROOT_DIR}/Processed/{dataset_split}"
        patient_ids_in_split = sorted(os.listdir(dataset_split_dir))
        patient_ids.extend(patient_ids_in_split)
        patient_dirs.extend([f"{dataset_split_dir}/{p_id}" for p_id in patient_ids_in_split])

    print("#Patients:", len(patient_dirs))


    fig, axs = plt.subplots(2, 6, figsize=(36, 12))

    fdg_pet_range_mean = IntensityRangeAndMean()
    pct_range_mean = IntensityRangeAndMean()
    hx4_pet_range_mean = IntensityRangeAndMean()
    ldct_range_mean = IntensityRangeAndMean()
    hx4_pet_reg_range_mean = IntensityRangeAndMean()
    ldct_reg_range_mean = IntensityRangeAndMean()

    print("Processing ...")
    for i, patient_dir in enumerate(tqdm(patient_dirs)):

        patient_id = patient_ids[i]

        # Read images
        fdg_pet, pct, hx4_pet, ldct, hx4_pet_reg, ldct_reg = read_all_images_of_patient(patient_dir)

        # Plot Histogram
        plot_histogram(fdg_pet, axs[0][0], bin_min=-1, bin_max=20, bin_size=0.25, units='SUV', title="FDG-PET")
        plot_histogram(pct, axs[0][1], bin_min=-1050, bin_max=2050, bin_size=20, units='HU', title="pCT")
        plot_histogram(hx4_pet, axs[0][2], bin_min=-1, bin_max=5, bin_size=0.25, units='SUV', title="HX4-PET")
        plot_histogram(ldct, axs[0][3], bin_min=-1050, bin_max=2050, bin_size=20, units='HU', title="ldCT")
        plot_histogram(hx4_pet_reg, axs[0][4], bin_min=-1, bin_max=5, bin_size=0.25, units='SUV', title="HX4-PET-reg")
        plot_histogram(ldct_reg, axs[0][5], bin_min=-1050, bin_max=2050, bin_size=20, units='HU', title="ldCT-reg")

        # Record range and mean
        fdg_pet_range_mean.record_range_and_mean(fdg_pet, patient_id)
        pct_range_mean.record_range_and_mean(pct, patient_id)
        hx4_pet_range_mean.record_range_and_mean(hx4_pet, patient_id)
        ldct_range_mean.record_range_and_mean(ldct, patient_id)
        hx4_pet_reg_range_mean.record_range_and_mean(hx4_pet_reg, patient_id)
        ldct_reg_range_mean.record_range_and_mean(ldct_reg, patient_id)


    # Plot patient vs. range
    fdg_pet_range_mean.plot_ranges_and_means(axs[1][0], units='SUV', title="FDG-PET")
    pct_range_mean.plot_ranges_and_means(axs[1][1],  units='HU', title="pCT")
    hx4_pet_range_mean.plot_ranges_and_means(axs[1][2],  units='SUV', title="HX4-PET")
    ldct_range_mean.plot_ranges_and_means(axs[1][3],  units='HU', title="ldCT")
    hx4_pet_reg_range_mean.plot_ranges_and_means(axs[1][4], units='SUV', title="HX4-PET-reg")
    ldct_reg_range_mean.plot_ranges_and_means(axs[1][5], units='HU', title="ldCT-reg")

    fig.suptitle(f"Intensity statistics - {''.join(dataset_splits)}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/intensity_stats_{''.join(dataset_splits)}.png")
    print("Figure saved")

    # Print global ranges
    print("Global range --")
    print("FDG-PET:", fdg_pet_range_mean.get_global_range())
    print("pCT:", pct_range_mean.get_global_range())
    print("HX4-PET:", hx4_pet_range_mean.get_global_range())
    print("ldCT:", ldct_range_mean.get_global_range())
    print("HX4-PET-reg:", hx4_pet_reg_range_mean.get_global_range())
    print("ldCT-reg:", ldct_reg_range_mean.get_global_range())


class IntensityRangeAndMean:
    def __init__(self):
        self.mins, self.maxs = {}, {}
        self.means = {}

    def record_range_and_mean(self, image_np, patient_id):
        self.mins[patient_id] = image_np.min()
        self.maxs[patient_id] = image_np.max()
        self.means[patient_id] = image_np.mean()

    def get_global_range(self):
        return min(self.mins.values()), max(self.maxs.values())

    def plot_ranges_and_means(self, ax, units, title):
        patient_ids = self.mins.keys()
        ax.plot(patient_ids, self.mins.values(), 'b', label="Minimum")
        ax.plot(patient_ids, self.maxs.values(), 'r', label="Maximum")
        ax.plot(patient_ids, self.means.values(), 'g', label="Mean")
        ax.set_ylabel(units)
        ax.set_xticklabels(patient_ids, rotation=90)
        ax.legend()
        ax.set_title(title)
        ax.grid(True)


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
    # get_stats(dataset_splits=['train'])
    # get_stats(dataset_splits=['val'])
    get_stats(dataset_splits=['train', 'val'])
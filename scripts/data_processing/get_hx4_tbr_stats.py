import os
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
OUTPUT_DIR = "../generated_metadata"


def main():
    data_dir = f"{DATA_ROOT_DIR}/Processed"
    dataset_splits = ['train', 'val']

    suv_aorta_mean_file = f"{DATA_ROOT_DIR}/SUVmean_aorta_HX4.csv"
    suv_aorta_mean_values = pd.read_csv(suv_aorta_mean_file, index_col=0)
    suv_aorta_mean_values = suv_aorta_mean_values.to_dict()['HX4 aorta SUVmean baseline']

    patient_ids = []
    patient_dirs = []
    for dataset_split in dataset_splits:
        patient_ids_split = sorted(os.listdir(f"{data_dir}/{dataset_split}"))
        patient_ids.extend(patient_ids_split)
        patient_dirs.extend([f"{data_dir}/{dataset_split}/{p_id}" for p_id in patient_ids_split])



    hx4_pet_range_mean = IntensityRangeAndMean()
    hx4_pet_reg_range_mean = IntensityRangeAndMean()

    fig, axs = plt.subplots(2,2, figsize=(12, 12))

    for i, patient_dir in enumerate(tqdm(patient_dirs)):
        patient_id = patient_ids[i]

        hx4_pet = sitk.ReadImage(f"{patient_dir}/hx4_pet.nrrd")
        hx4_pet_reg = sitk.ReadImage(f"{patient_dir}/hx4_pet_reg.nrrd")

        hx4_pet = sitk.GetArrayFromImage(hx4_pet)
        hx4_pet_reg = sitk.GetArrayFromImage(hx4_pet_reg)

        # Normalize with SUVmean_aorta value
        suv_mean_aorta = suv_aorta_mean_values[patient_id]
        hx4_pet = hx4_pet / suv_mean_aorta
        hx4_pet_reg = hx4_pet_reg / suv_mean_aorta

        # Plot histogram
        plot_histogram(hx4_pet, axs[0][0], bin_min=-1, bin_max=10, bin_size=0.25, units='TBR', title="HX4-PET")
        plot_histogram(hx4_pet_reg, axs[0][1], bin_min=-1, bin_max=10, bin_size=0.25, units='TBR', title="HX4-PET-reg")

        # Calulate and store range and mean
        hx4_pet_range_mean.record_range_and_mean(hx4_pet, patient_id)
        hx4_pet_reg_range_mean.record_range_and_mean(hx4_pet_reg, patient_id)

    # Calulate and store range and mean
    hx4_pet_range_mean.plot_ranges_and_means(axs[1][0],  units='TBR', title="HX4-PET")
    hx4_pet_reg_range_mean.plot_ranges_and_means(axs[1][1],  units='TBR', title="HX4-PET-reg")

    # Save figure
    fig.suptitle(f"Stats of HX4 images normalized with SUVmean_aorta (or \"TBR\") - {''.join(dataset_splits)}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/hx4_tbr_stats_{''.join(dataset_splits)}.png")
    print("Figure saved")

    # Print global ranges
    print("Global range --")
    print("HX4-PET:", hx4_pet_range_mean.get_global_range())
    print("HX4-PET-reg:", hx4_pet_reg_range_mean.get_global_range())



def plot_histogram(image_np, ax, bin_min, bin_max, bin_size, units, title):
    ax.hist(image_np.flatten(), bins=np.arange(bin_min, bin_max, bin_size), histtype='step')
    ax.set_xlabel(units)
    ax.set_ylabel("voxels")
    ax.set_title(title)



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



if __name__ == '__main__':
    main()
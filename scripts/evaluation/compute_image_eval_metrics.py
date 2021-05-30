"""
Input:
    - Ground truth (HX4-PET-reg)
    - A model's final predictions (i.e. from last checkpoint).
    - Other stuff for pre-processing - body masks and SUVmean_aorta file

Output:
    - CSV file containing patient-wise and aggregate values of all 6 metrics for the given model.
"""

import os
import argparse
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import sys
sys.path.append("../")
from utils.io import read_nrrd_to_sitk
from utils.sitk_utils import sitk2np
from utils.metrics import ssim, mse, psnr, mae, nmi, histogram_chi2
from utils.auto_mask_generation import get_body_mask


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
RESULTS_ROOT_DIR = "/home/chinmay/Desktop/Projects-Work/HX4-PET-Translation/Results/HX4-PET-Translation"
HX4_TBR_RANGE = (0.0, 3.0)
METRIC_DICT = {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim, "nmi": nmi, "histogram_chi2": histogram_chi2}

DEFAULT_MODEL_NAME = "hx4_pet_pix2pix_lambda10"
CHECKPOINT_NUMBER = 60000


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name",
                        type=str,
                        default=DEFAULT_MODEL_NAME,
                        help="Model name"
                        )

    parser.add_argument("--checkpoint_number",
                        type=int,
                        default=CHECKPOINT_NUMBER,
                        help="Refers to the number of iterations that produced the checkpoint"
                        )

    args = parser.parse_args()
    return args


def main(args):
    print("Model name:", args.model_name)
    print("Checkpoint number:", args.checkpoint_number)

    preds_dir = f"{RESULTS_ROOT_DIR}/{args.model_name}/val/saved/{args.checkpoint_number}"
    output_file = f"{RESULTS_ROOT_DIR}/validation_metrics-{args.model_name}-{args.checkpoint_number}.csv"

    suv_aorta_mean_file =  f"{DATA_ROOT_DIR}/SUVmean_aorta_HX4.csv"
    suv_aorta_mean_values = pd.read_csv(suv_aorta_mean_file, index_col=0)
    suv_aorta_mean_values = suv_aorta_mean_values.to_dict()['HX4 aorta SUVmean baseline']


    patient_ids = sorted(os.listdir(f"{DATA_ROOT_DIR}/Processed/val"))

    # Dict of lists
    metrics_record = {metric_name: [] for metric_name in METRIC_DICT.keys()}

    print("Computing image quality metrics ...")

    for p_id in tqdm(patient_ids):

        # Fetch images
        hx4_pet_reg = sitk2np(read_nrrd_to_sitk(f"{DATA_ROOT_DIR}/Processed/val/{p_id}/hx4_pet_reg.nrrd"))
        pred = sitk2np(read_nrrd_to_sitk(f"{preds_dir}/{p_id}.nrrd"))
        if p_id == 'N046':
            pct = sitk2np(read_nrrd_to_sitk(f"{DATA_ROOT_DIR}/Processed/val/{p_id}/pct.nrrd"))
            body_mask = get_body_mask(pct, hu_threshold=-300)
        else:
            body_mask = sitk2np(read_nrrd_to_sitk(f"{DATA_ROOT_DIR}/Processed/val/{p_id}/pct_body.nrrd"))

        # ------------------------------------------------------------
        # Preprocess the same way as done in the data loading pipeline

        # Apply body mask to ground truth
        hx4_pet_reg = body_mask * hx4_pet_reg

        # Normalize with SUVmean_aorta values
        suv_aorta_mean_value = suv_aorta_mean_values[p_id]
        hx4_pet_reg = hx4_pet_reg / suv_aorta_mean_value
        pred = pred / suv_aorta_mean_value

        # Clip to the allowed TBR range
        hx4_pet_reg = np.clip(hx4_pet_reg, HX4_TBR_RANGE[0], HX4_TBR_RANGE[1])
        pred = np.clip(pred, HX4_TBR_RANGE[0], HX4_TBR_RANGE[1])

        # Compute and record metrics
        patient_metrics = {}
        for metric_name in METRIC_DICT.keys():
            metric_function = METRIC_DICT[metric_name]
            metric_value = metric_function(hx4_pet_reg, pred)
            metrics_record[metric_name].append(metric_value)

    # Calculate mean and stddev of each metric
    for metric_name in metrics_record.keys():
        mean = np.mean(metrics_record[metric_name])
        std_dev = np.std(metrics_record[metric_name])
        metrics_record[metric_name].extend([mean, std_dev])

    # Write output to CSV file
    metrics_record = pd.DataFrame.from_dict(metrics_record, orient='columns')
    metrics_record.index = [*patient_ids, 'Mean', 'Std-dev']
    metrics_record.to_csv(output_file)

    # Display summary
    print("Summary:")
    for metric_name in METRIC_DICT.keys():
        mean = metrics_record.loc['Mean', metric_name]
        std_dev = metrics_record.loc['Std-dev', metric_name]
        print(f"\t{metric_name}: {mean:.3f} +/- {std_dev:.3f}")



if __name__ == '__main__':
    args = get_args()
    main(args)
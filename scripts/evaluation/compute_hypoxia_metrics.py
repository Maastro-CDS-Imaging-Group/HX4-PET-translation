"""
Input:
    - Ground truth (HX4-PET-reg)
    - A model's final predictions (i.e. from last checkpoint).
    - GTV mask
    - Other stuff for pre-processing - body masks and SUVmean_aorta file

Output:
    - CSV file containing patient-wise and aggregate values of all 3 hypoxia related metrics for the given model.
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
from utils.metrics import ssim, mse
from utils.auto_mask_generation import get_body_mask


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
RESULTS_ROOT_DIR = "/home/chinmay/Desktop/Projects-Work/HX4-PET-Translation/Results/HX4-PET-Translation"
HX4_TBR_RANGE = (0.0, 3.0)
METRIC_NAMES = ('MSE-GTV', 'SSIM-GTV', 'tumour-classification-gt', 'tumour-classification-pred', 'hypoxic-region-seg-dice')

# Hypoxia measurement settings
TBR_THRESHOLD = 1.4
HYPOXIC_VOLUME_THRESHOLD = 1000 / (1.0 * 1.0 * 3.0)  # 1000 mm3 / voxel size.

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
    output_file = f"{RESULTS_ROOT_DIR}/hypoxia_metrics-{args.model_name}-{args.checkpoint_number}.csv"

    suv_aorta_mean_file =  f"{DATA_ROOT_DIR}/SUVmean_aorta_HX4.csv"
    suv_aorta_mean_values = pd.read_csv(suv_aorta_mean_file, index_col=0)
    suv_aorta_mean_values = suv_aorta_mean_values.to_dict()['HX4 aorta SUVmean baseline']


    patient_ids = sorted(os.listdir(f"{DATA_ROOT_DIR}/Processed/val"))

    # Dict of lists
    metrics_record = {metric_name: [] for metric_name in METRIC_NAMES}

    print("Computing hypoxia related metrics ...")

    for p_id in tqdm(patient_ids):

        # Fetch images
        ground_truth = sitk2np(read_nrrd_to_sitk(f"{DATA_ROOT_DIR}/Processed/val/{p_id}/hx4_pet_reg.nrrd"))
        pred = sitk2np(read_nrrd_to_sitk(f"{preds_dir}/{p_id}.nrrd"))
        gtv_mask = sitk2np(read_nrrd_to_sitk(f"{DATA_ROOT_DIR}/Processed/val/{p_id}/pct_gtv.nrrd"))


        # ----------------------------------------------------------------
        # Crop the images to a GTV-size bbox to make computation efficient
        ground_truth = crop_image_to_tumour_size(ground_truth, gtv_mask)
        pred = crop_image_to_tumour_size(pred, gtv_mask)
        gtv_mask = crop_image_to_tumour_size(gtv_mask, gtv_mask)


        # ------------------------------------------------------------
        # Preprocess the same way as done in the data loading pipeline
        # Note: Body masking not needed because the tumour is already within the body region.

        # Normalize with SUVmean_aorta values
        suv_aorta_mean_value = suv_aorta_mean_values[p_id]
        ground_truth = ground_truth / suv_aorta_mean_value
        pred = pred / suv_aorta_mean_value

        # Clip to the allowed TBR range
        ground_truth = np.clip(ground_truth, HX4_TBR_RANGE[0], HX4_TBR_RANGE[1])
        pred = np.clip(pred, HX4_TBR_RANGE[0], HX4_TBR_RANGE[1])


        # --------------------------
        # Compute and record metrics
        patient_metrics = {}

        # Masked MSE and SSIM
        pred_masked = create_masked_array(pred, gtv_mask)
        ground_truth_reg_masked = create_masked_array(ground_truth, gtv_mask)
        mse_gtv_value = mse(ground_truth_reg_masked, pred_masked)
        ssim_gtv_value = ssim(ground_truth_reg_masked, pred_masked)
        metrics_record['MSE-GTV'].append(mse_gtv_value)
        metrics_record['SSIM-GTV'].append(ssim_gtv_value)

        # Tumour classification
        pred_hypoxia_seg = pred * gtv_mask
        pred_hypoxia_seg = pred_hypoxia_seg > TBR_THRESHOLD
        ground_truth_hypoxia_seg = ground_truth * gtv_mask
        ground_truth_hypoxia_seg = ground_truth_hypoxia_seg > TBR_THRESHOLD

        pred_hv = np.sum(pred_hypoxia_seg) # Hypoxic volume.
        pred_gtv_hypoxic = np.uint8(pred_hv > HYPOXIC_VOLUME_THRESHOLD)
        metrics_record['tumour-classification-pred'].append(pred_gtv_hypoxic)
        gt_hv = np.sum(ground_truth_hypoxia_seg)
        gt_gtv_hypoxic = np.uint8(gt_hv > HYPOXIC_VOLUME_THRESHOLD)
        metrics_record['tumour-classification-gt'].append(gt_gtv_hypoxic)
        # print(p_id, gt_hv)

        # Hypoxic region segmentation Dice
        dice_value = dice(ground_truth_hypoxia_seg, pred_hypoxia_seg)
        metrics_record['hypoxic-region-seg-dice'].append(dice_value)


    # Calculate mean and stddev of each metric
    for metric_name in metrics_record.keys():
        if metric_name in ['MSE-GTV', 'SSIM-GTV', 'hypoxic-region-seg-dice']:
            mean = np.mean(metrics_record[metric_name])
            std_dev = np.std(metrics_record[metric_name])
            metrics_record[metric_name].extend([mean, std_dev])
        else:
            classification_accuracy = np.sum(np.array(metrics_record['tumour-classification-gt'])[:len(patient_ids)] == np.array(metrics_record['tumour-classification-pred'])[:len(patient_ids)]) / len(patient_ids)
            metrics_record[metric_name].extend([classification_accuracy, np.nan])

    # Write output to CSV file
    metrics_record = pd.DataFrame.from_dict(metrics_record, orient='columns')
    metrics_record.index = [*patient_ids, 'Mean', 'Std-dev']
    metrics_record.to_csv(output_file)

    # Display summary
    print("Summary:")
    for metric_name in METRIC_NAMES:
        if metric_name in ['MSE-GTV', 'SSIM-GTV', 'hypoxic-region-seg-dice']:
            mean = metrics_record.loc['Mean', metric_name]
            std_dev = metrics_record.loc['Std-dev', metric_name]
            print(f"\t{metric_name}: {mean:.3f} +/- {std_dev:.3f}")
        else:
            classification_accuracy = metrics_record.loc['Mean', metric_name]
            print(f"\tAccuracy: {classification_accuracy:.3f}")


def crop_image_to_tumour_size(image, gtv_mask):
    gtv_voxel_coords = np.argwhere(gtv_mask)
    z1, y1, x1 = gtv_voxel_coords[:, 0].min(), gtv_voxel_coords[:, 1].min(), gtv_voxel_coords[:, 2].min()
    z2, y2, x2 = gtv_voxel_coords[:, 0].max(), gtv_voxel_coords[:, 1].max(), gtv_voxel_coords[:, 2].max()
    return image[z1:z2, y1:y2, x1:x2]


def create_masked_array(image, mask):
    """
    Create a masked array after applying the respective mask.
    This mask array will filter values across different operations such as mean
    """
    mask = mask.astype(np.bool)
    # Masked array needs negated masks as it decides
    # what element to ignore based on True values
    negated_mask = ~mask
    return np.ma.masked_array(image * mask, mask=negated_mask)


def dice(gt_seg, pred_seg):
    dice_score = 2 * np.sum(gt_seg * pred_seg) / (gt_seg.sum() + pred_seg.sum())
    return float(dice_score)




if __name__ == '__main__':
    args = get_args()
    main(args)
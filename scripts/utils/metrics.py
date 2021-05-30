"""
Metric implementations taken from midaGAN
"""

import numpy as np
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def mae(gt, pred):
    """Compute Mean Absolute Error (MAE)"""
    mae_value = np.mean(np.abs(gt - pred))
    return float(mae_value)


def mse(gt, pred):
    """Compute Mean Squared Error (MSE)"""
    mse_value = np.mean((gt - pred)**2)
    return float(mse_value)


def nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    nmse_value = np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2
    return float(nmse_value)


def psnr(gt, pred, maxval=None):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    maxval = gt.max() if maxval is None else maxval
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=maxval)
    return float(psnr_value)


def ssim(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    size = (gt.shape[0] * gt.shape[1]) if gt.ndim == 4 else gt.shape[0]

    ssim_sum = 0
    for channel in range(gt.shape[0]):
        # Format is CxHxW or DxHxW
        if gt.ndim == 3:
            target = gt[channel]
            prediction = pred[channel]
            ssim_sum += structural_similarity(target, prediction, data_range=maxval)

        # Format is CxDxHxW
        elif gt.ndim == 4:
            for slice_num in range(gt.shape[1]):
                target = gt[channel, slice_num]
                prediction = pred[channel, slice_num]
                ssim_sum += structural_similarity(target, prediction, data_range=maxval)
        else:
            raise NotImplementedError(f"SSIM for {gt.ndim} images not implemented")

    return ssim_sum / size


def nmi(gt, pred):
    """Normalized Mutual Information.
    Implementation taken from scikit-image 0.19.0.dev0 source --
        https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/simple_metrics.py#L193-L261
    Not using scikit-image because NMI is supported only in >=0.19.
    """
    bins = 100  # 100 bins by default
    hist, bin_edges = np.histogramdd(
            [np.reshape(gt, -1), np.reshape(pred, -1)],
            bins=bins,
            density=True,
            )
    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))
    nmi_value = (H0 + H1) / H01
    return float(nmi_value)


def histogram_chi2(gt, pred):
    """Chi-squared distance computed between histograms of the GT and the prediction.
    More about comparing two histograms --
        https://stackoverflow.com/questions/6499491/comparing-two-histograms
    """
    bins = 100  # 100 bins by default

    # Compute histograms
    gt_histogram, gt_bin_edges = np.histogram(gt, bins=bins)
    pred_histogram, pred_bin_edges = np.histogram(pred, bins=bins)

    # Normalize the histograms to convert them into discrete distributions
    gt_histogram = gt_histogram / gt_histogram.sum()
    pred_histogram = pred_histogram / pred_histogram.sum()

    # Compute chi-squared distance
    bin_to_bin_distances = (pred_histogram - gt_histogram)**2 / (pred_histogram + gt_histogram)
    # Remove NaN values caused by 0/0 division. Equivalent to manually setting them as 0.
    bin_to_bin_distances = bin_to_bin_distances[np.logical_not(np.isnan(bin_to_bin_distances))]
    chi2_distance_value = np.sum(bin_to_bin_distances)
    return float(chi2_distance_value)




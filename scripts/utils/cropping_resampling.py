import numpy as np
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator
import SimpleITK as sitk

from utils.sitk_utils import np2sitk 



# -------------------------------------------------------
# For cropping and resampling a single PET-CT pair
# -------------------------------------------------------

def crop_and_resample_pet_ct(pet_sitk, ct_sitk, masks=None, resample_spacing=(1.0, 1.0, 3.0)):

    bbox = get_volume_intersection_bbox(pet_sitk, ct_sitk)

    pet_crop_res_sitk = resample_and_crop(pet_sitk, bbox, resample_spacing, order=3, is_mask=False)
    ct_crop_res_sitk = resample_and_crop(ct_sitk, bbox, resample_spacing, order=3, is_mask=False)

    # Masks given ?
    if masks is not None:
        masks_crop_res = {}
        for k in masks.keys():
            masks_crop_res[k] = resample_and_crop(masks[k], bbox, resample_spacing, is_mask=True)

        return pet_crop_res_sitk, ct_crop_res_sitk, masks_crop_res
    else:
        return pet_crop_res_sitk, ct_crop_res_sitk


def resample_and_crop(sitk_image, bounding_box, resampling=(1.0, 1.0, 3.0), order=3, is_mask=False):

    pixel_spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    np_volume = sitk.GetArrayFromImage(sitk_image)
    np_volume = np_volume.transpose(2,1,0)  # DHW to WHD
    resampling = np.asarray(resampling)

    if is_mask:
        np_volume = resample_np_binary_volume(np_volume, origin, pixel_spacing, resampling, bounding_box)
    else:
        np_volume = resample_np_volume(np_volume, origin, pixel_spacing, resampling, bounding_box, order=order)

    origin = np.asarray([bounding_box[0], bounding_box[1], bounding_box[2]])
    sitk_image = sitk.GetImageFromArray(np_volume.transpose(2,1,0))
    sitk_image.SetSpacing(resampling)
    sitk_image.SetOrigin(origin)
    return sitk_image

def resample_np_volume(np_volume,
                       origin,
                       current_pixel_spacing,
                       resampling_px_spacing,
                       bounding_box,
                       order=3):

    zooming_matrix = np.identity(3)
    zooming_matrix[0, 0] = resampling_px_spacing[0] / current_pixel_spacing[0]
    zooming_matrix[1, 1] = resampling_px_spacing[1] / current_pixel_spacing[1]
    zooming_matrix[2, 2] = resampling_px_spacing[2] / current_pixel_spacing[2]

    offset = ((bounding_box[0] - origin[0]) / current_pixel_spacing[0],
              (bounding_box[1] - origin[1]) / current_pixel_spacing[1],
              (bounding_box[2] - origin[2]) / current_pixel_spacing[2])

    output_shape = np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing

    np_volume = affine_transform(np_volume,
                                 zooming_matrix,
                                 offset=offset,
                                 mode='mirror',
                                 order=order,
                                 output_shape=output_shape.astype(int))
    return np_volume


def grid_from_spacing(start, spacing, n):
    return np.asarray([start + k * spacing for k in range(n)])


def resample_np_binary_volume(np_volume, origin, current_pixel_spacing, resampling_px_spacing, bounding_box):

    x_old = grid_from_spacing(origin[0], current_pixel_spacing[0], np_volume.shape[0])
    y_old = grid_from_spacing(origin[1], current_pixel_spacing[1], np_volume.shape[1])
    z_old = grid_from_spacing(origin[2], current_pixel_spacing[2], np_volume.shape[2])

    output_shape = (np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing).astype(int)

    x_new = grid_from_spacing(bounding_box[0], resampling_px_spacing[0], output_shape[0])
    y_new = grid_from_spacing(bounding_box[1], resampling_px_spacing[1], output_shape[1])
    z_new = grid_from_spacing(bounding_box[2], resampling_px_spacing[2], output_shape[2])
    interpolator = RegularGridInterpolator((x_old, y_old, z_old),
                                           np_volume,
                                           method='nearest',
                                           bounds_error=False,
                                           fill_value=0)
    x, y, z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))

    return interpolator(pts).reshape(output_shape)



# -------------------------------------------------------
# Cropping, between two PET-CT pairs
# -------------------------------------------------------

def crop_pet_ct_pairs_to_common_roi(pet_sitk_1, ct_sitk_1,
                                    pet_sitk_2, ct_sitk_2,
                                    masks=None
                                    ):

    bbox = get_volume_intersection_bbox(pet_sitk_1, pet_sitk_2)

    pet_sitk_1_crop = crop_sitk(pet_sitk_1, bbox)
    ct_sitk_1_crop = crop_sitk(ct_sitk_1, bbox)
    pet_sitk_2_crop = crop_sitk(pet_sitk_2, bbox)
    ct_sitk_2_crop = crop_sitk(ct_sitk_2, bbox)

    if masks is None:
        return pet_sitk_1_crop, ct_sitk_1_crop, pet_sitk_2_crop, ct_sitk_2_crop
    else:
        masks_crop = {}
        for k in masks.keys():
            masks_crop[k] = crop_sitk(masks[k], bbox)
        return pet_sitk_1_crop, ct_sitk_1_crop, pet_sitk_2_crop, ct_sitk_2_crop, masks_crop


def crop_sitk(sitk_image, bbox):
    # Get WHD indices from physcial bbox points
    x1, y1, z1 = sitk_image.TransformPhysicalPointToIndex((bbox[0], bbox[1], bbox[2]))
    x2, y2, z2 = sitk_image.TransformPhysicalPointToIndex((bbox[3], bbox[4], bbox[5]))

    # To numpy, crop
    np_image = sitk.GetArrayFromImage(sitk_image)
    np_image = np_image.transpose(2,1,0)  # DHW to WHD
    np_image_crop = np_image[x1:x2, y1:y2, z1:z2]

    # Back to sitk, update metadata
    np_image_crop = np_image_crop.transpose(2,1,0)  # WHD to DHW
    spacing = sitk_image.GetSpacing()
    origin = (bbox[0], bbox[1], bbox[2])
    sitk_image_crop = np2sitk(np_image_crop, spacing, origin)
    return sitk_image_crop


def get_volume_intersection_bbox(sitk_image_1, sitk_image_2):
    origin_1, origin_2 = sitk_image_1.GetOrigin(), sitk_image_2.GetOrigin()
    spacing_1, spacing_2 = sitk_image_1.GetSpacing(), sitk_image_2.GetSpacing()
    width_1, width_2 = sitk_image_1.GetWidth(), sitk_image_2.GetWidth()
    height_1, height_2 = sitk_image_1.GetHeight(), sitk_image_2.GetHeight()
    depth_1, depth_2 = sitk_image_1.GetDepth(), sitk_image_2.GetDepth()

    x_max_phy_pos_1, x_max_phy_pos_2 = origin_1[0] + spacing_1[0]*width_1, origin_2[0] + spacing_2[0]*width_2
    y_max_phy_pos_1, y_max_phy_pos_2 = origin_1[1] + spacing_1[1]*height_1, origin_2[1] + spacing_2[1]*height_2
    z_max_phy_pos_1, z_max_phy_pos_2 = origin_1[2] + spacing_1[2]*depth_1, origin_2[2] + spacing_2[2]*depth_2

    x_min_crop_phy_pos, x_max_crop_phy_pos = max(origin_1[0], origin_2[0]), min(x_max_phy_pos_1, x_max_phy_pos_2)
    y_min_crop_phy_pos, y_max_crop_phy_pos = max(origin_1[1], origin_2[1]), min(y_max_phy_pos_1, y_max_phy_pos_2)
    z_min_crop_phy_pos, z_max_crop_phy_pos = max(origin_1[2], origin_2[2]), min(z_max_phy_pos_1, z_max_phy_pos_2)

    bbox = np.array([x_min_crop_phy_pos, y_min_crop_phy_pos, z_min_crop_phy_pos,
                     x_max_crop_phy_pos, y_max_crop_phy_pos, z_max_crop_phy_pos])
    return bbox
import numpy as np
import SimpleITK as sitk



def np2sitk(np_image, pixel_spacing, origin):
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.SetSpacing(pixel_spacing)
    sitk_image.SetOrigin(origin)
    return sitk_image


def apply_body_mask(ct_sitk, body_mask_sitk):
    # Convert to numpy (DHW)
    ct_np = sitk.GetArrayFromImage(ct_sitk) 
    body_mask_np = sitk.GetArrayFromImage(body_mask_sitk)

    # Apply mask and set out-of-mask region HU as -1000 (air) 
    ct_bodyonly_np = ct_np * body_mask_np
    ct_bodyonly_np = ct_bodyonly_np + (body_mask_np == 0).astype(np.uint8) * (-1000)

    # Convert to sitk (WHD)
    ct_bodyonly_sitk = sitk.GetImageFromArray(ct_bodyonly_np)
    ct_bodyonly_sitk.CopyInformation(ct_sitk)
    
    return ct_bodyonly_sitk
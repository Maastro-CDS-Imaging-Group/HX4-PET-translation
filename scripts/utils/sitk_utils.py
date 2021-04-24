import SimpleITK as sitk



def np2sitk(np_image, pixel_spacing, origin):
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.SetSpacing(pixel_spacing)
    sitk_image.SetOrigin(origin)
    return sitk_image
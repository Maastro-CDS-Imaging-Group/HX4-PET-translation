import numpy as np
from skimage.draw import polygon
import pydicom as pdcm



def get_masks(rtstruct_file,
              labels,
              image_position_patient=None,
              axial_positions=None,
              pixel_spacing=None,
              shape=None,
              dtype=np.int8):
    contours = read_structure(rtstruct_file, labels=labels)
    return get_mask_from_contour(contours, image_position_patient, axial_positions, pixel_spacing,
                                 shape, dtype=dtype)


def read_structure(rtstruct_file, labels):
    structure = pdcm.read_file(rtstruct_file)
    contours = []
    for i, roi_seq in enumerate(structure.StructureSetROISequence):
        contour = {}
        for label in labels:
            if roi_seq.ROIName == label:
                contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
                contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
                contour['name'] = roi_seq.ROIName
                assert contour['number'] == roi_seq.ROINumber
                contour['contours'] = [s.ContourData
                    for s in structure.ROIContourSequence[i].ContourSequence]
                contours.append(contour)
    return contours


def get_mask_from_contour(contours, image_position_patient, axial_positions, pixel_spacing, shape, dtype=np.uint8):
    z = np.asarray(axial_positions)
    pos_r = image_position_patient[1]
    spacing_r = pixel_spacing[1]
    pos_c = image_position_patient[0]
    spacing_c = pixel_spacing[0]

    output = {}
    for con in contours:
        mask = np.zeros(shape, dtype=dtype)
        for current in con['contours']:
            nodes = np.array(current).reshape((-1, 3))           
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            
            try:  # Try-except patch by Chinmay because caused error in one patient (N016)
                z_index = np.where((nodes[0, 2] - 0.001 < z)
                                & (z < nodes[0, 2] + 0.001))[0][0]
            except IndexError:
                continue

            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            if len(rr) > 0 and len(cc) > 0:
                if np.max(rr) > 512 or np.max(cc) > 512:
                    raise Exception("The RTSTRUCT file is compromised")
            mask[rr, cc, z_index] = 1
        output[con['name']] = mask
    return output


def get_physical_values_ct(slices, dtype=np.float32):
    image = list()
    for s in slices:
        image.append(float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept))
    return np.stack(image, axis=-1).astype(dtype)
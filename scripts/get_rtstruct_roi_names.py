import os
import glob
from tqdm.auto import tqdm
import pandas as pd
import pydicom


DATA_ROOT_DIR = "/home/chinmay/Datasets/HX4-PET-Translation"
OUTPUT_FILEPATH_1 = "../generated_metadata/all_rtstruct_roi_names.txt"
OUTPUT_FILEPATH_2 = "../generated_metadata/selected_rtstruct_roi_names.csv"


def get_all_roi_names():
    patient_ids = sorted(os.listdir(f"{DATA_ROOT_DIR}/Original"))
    roi_names = {}
    for p_id in tqdm(patient_ids):
        rtstruct_filepath = glob.glob(f"{DATA_ROOT_DIR}/Original/{p_id}/FDG/RTSTRUCT/*.dcm")[0]
        structure = pydicom.read_file(rtstruct_filepath)
        patient_roi_names = []
        for roi_seq in structure.StructureSetROISequence:
            patient_roi_names.append(roi_seq.ROIName)        
        roi_names[p_id] = sorted(patient_roi_names)
    
    # Write output to a file
    with open(OUTPUT_FILEPATH_1, 'w') as of:
        output = ""
        for p_id in roi_names.keys():
            output += f"{p_id} "
            output += '  '.join(roi_names[p_id])
            output += "\n"
        of.write(output)   

    return roi_names 
    

def select_relevant_roi_names(roi_names):
    possible_gtv_roi_names = ['GTVp1', 'GTV1', 'GTV-1', 'GTV-prim', 'GTV-prim1']
    possible_body_roi_names = ['BODY', 'bodycontour']

    relevant_roi_names = {}
    for p_id in roi_names:
        relevant_roi_names[p_id] = {}
        patient_roi_names = roi_names[p_id]
        for gtv_roi_name in possible_gtv_roi_names:
            if gtv_roi_name in patient_roi_names:
                relevant_roi_names[p_id]['gtv-roi-name'] = gtv_roi_name
        for body_roi_name in possible_body_roi_names:
            if body_roi_name in patient_roi_names:            
                relevant_roi_names[p_id]['body-roi-name'] = body_roi_name

    # Corrections
    relevant_roi_names['N031']['gtv-roi-name'] = 'GTVp2'  # This one doesn't have the 'GTVp1', but only 'GTVp2'
    relevant_roi_names['N046']['body-roi-name'] = ''         # This one doesn't have a body mask
    
    # Display
    for k in relevant_roi_names.keys():
        print(k, relevant_roi_names[k])
        
    # Write output to a file
    relevant_roi_names = pd.DataFrame.from_dict(relevant_roi_names, orient='index')
    relevant_roi_names.to_csv(OUTPUT_FILEPATH_2)


if __name__ == '__main__':
    roi_names = get_all_roi_names()
    select_relevant_roi_names(roi_names)
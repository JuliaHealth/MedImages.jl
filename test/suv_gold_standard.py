import pydicom
import os
import numpy as np
import SimpleITK as sitk
from datetime import datetime

def parse_time(time_str):
    if '.' in time_str:
        return datetime.strptime(time_str, "%H%M%S.%f")
    return datetime.strptime(time_str, "%H%M%S")

def calculate_suv_python(dicom_dir):
    # 1. Use SimpleITK to load the series (handles spatial metadata)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    img_sitk = reader.Execute()
    
    # 2. Use pydicom to get SUV-related metadata from one file
    ds = pydicom.dcmread(dicom_names[0])
    weight_kg = float(ds.PatientWeight)
    
    radio_info = ds.RadiopharmaceuticalInformationSequence[0]
    total_dose_bq = float(radio_info.RadionuclideTotalDose)
    half_life_s = float(radio_info.RadionuclideHalfLife)
    inj_time_str = radio_info.RadiopharmaceuticalStartTime
    
    scan_time_str = getattr(ds, 'AcquisitionTime', getattr(ds, 'SeriesTime', None))
    
    t_inj = parse_time(inj_time_str)
    t_scan = parse_time(scan_time_str)
    
    delta_s = (t_scan - t_inj).total_seconds()
    if delta_s < 0:
        delta_s += 24 * 3600
        
    decay_factor = 2.0**(-delta_s / half_life_s)
    actual_dose = total_dose_bq * decay_factor
    
    suv_factor = (weight_kg * 1000.0) / actual_dose
    print(f"Python SUV Factor: {suv_factor}")
    print(f"Injection Time: {inj_time_str}, Scan Time: {scan_time_str}, Delta: {delta_s}s")
    
    # 3. Apply SUV factor to the SITK image
    # Note: sitk.GetArrayFromImage gives (z, y, x)
    # voxel_data = sitk.GetArrayFromImage(img_sitk)
    # suv_voxel_data = voxel_data * suv_factor
    # res_img = sitk.GetImageFromArray(suv_voxel_data)
    # res_img.CopyInformation(img_sitk)
    
    # Simpler way in SITK:
    res_img = sitk.Cast(img_sitk, sitk.sitkFloat32) * suv_factor
    
    # 4. Save
    sitk.WriteImage(res_img, "pythonsuv.nii.gz")
    return suv_factor

if __name__ == "__main__":
    dicom_dir = "/home/jm/project_ssd/MedImages.jl/test/visual_output/local/dicoms/5-PET_WB_120min_Uncorrected/resources/DICOM/files"
    calculate_suv_python(dicom_dir)

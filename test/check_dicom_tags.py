import pydicom
import os

def check_tags(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    if not files:
        print("No files found.")
        return
    
    ds = pydicom.dcmread(files[0])
    print(f"File: {files[0]}")
    
    tags = ["PatientWeight", "AcquisitionTime", "SeriesTime"]
    for t in tags:
        val = getattr(ds, t, "MISSING")
        print(f"{t}: {val}")
        
    if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
        seq = ds.RadiopharmaceuticalInformationSequence
        if len(seq) > 0:
            item = seq[0]
            print("RadiopharmaceuticalInformationSequence[0]:")
            subtags = ["RadionuclideTotalDose", "RadionuclideHalfLife", "RadiopharmaceuticalStartTime"]
            for st in subtags:
                print(f"  {st}: {getattr(item, st, 'MISSING')}")
        else:
            print("RadiopharmaceuticalInformationSequence is empty.")
    else:
        print("RadiopharmaceuticalInformationSequence is MISSING.")

if __name__ == "__main__":
    dicom_dir = "/home/jm/project_ssd/MedImages.jl/test/visual_output/local/dicoms/5-PET_WB_120min_Uncorrected/resources/DICOM/files"
    check_tags(dicom_dir)

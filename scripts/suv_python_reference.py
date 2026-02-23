import pydicom
from pathlib import Path
from datetime import datetime
import sys

def get_suv_factor(dicom_folder):
    """
    Args:
        dicom_folder (str): Path to the directory containing DICOM files.

    Returns:
        float: The SUV conversion factor (to be multiplied by pixel values).
        None: If an error occurs or required DICOM tags are missing.
    """
    try:
        dicom_path = Path(dicom_folder)

        # 1. Search for DICOM files in the provided folder
        # Look for files with the .dcm extension
        dcm_files = list(dicom_path.glob("*.dcm"))

        # If no .dcm files found, try reading all files (DICOM files often lack extensions)
        if not dcm_files:
            dcm_files = [f for f in dicom_path.iterdir() if f.is_file()]

        if not dcm_files:
            print(f"   [SUV ERROR] Folder is empty or contains no DICOM files: {dicom_folder}")
            return None

        # 2. Read the header from the first file
        # We assume the entire series shares the same patient and isotope parameters.
        # stop_before_pixels=True speeds up reading by skipping image data.
        target_dcm = dcm_files[0]
        ds = pydicom.dcmread(target_dcm, stop_before_pixels=True)

        # 3. Retrieve Patient Weight
        # Tag (0010,1030) PatientWeight is in kilograms, convert to grams.
        if not hasattr(ds, 'PatientWeight') or not ds.PatientWeight:
            print("   [SUV ERROR] Missing PatientWeight in DICOM header.")
            return None
        weight_g = float(ds.PatientWeight) * 1000.0

        # 4. Retrieve Radiopharmaceutical Information (Isotope data)
        # Sequence (0054,0016) RadiopharmaceuticalInformationSequence
        if not hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
            print("   [SUV ERROR] Missing RadiopharmaceuticalInformationSequence.")
            return None

        radio_seq = ds.RadiopharmaceuticalInformationSequence[0]

        # Total injected dose in Becquerels (Bq)
        inj_dose_bq = float(radio_seq.RadionuclideTotalDose)
        # Half-life in seconds
        half_life_s = float(radio_seq.RadionuclideHalfLife)
        # Injection time (format HHMMSS.ffffff)
        inj_time_str = radio_seq.RadiopharmaceuticalStartTime

        # 5. Determine Scan (Acquisition) Time
        # Priority: AcquisitionTime, fallback: SeriesTime
        scan_time_str = ds.AcquisitionTime if hasattr(ds, 'AcquisitionTime') else ds.SeriesTime

        # Helper function to parse DICOM time strings (handles fractional seconds)
        def parse_tm(t_str):
            t_str = str(t_str).strip()
            if '.' in t_str:
                return datetime.strptime(t_str, "%H%M%S.%f")
            return datetime.strptime(t_str, "%H%M%S")

        try:
            t_inj = parse_tm(inj_time_str)
            t_scan = parse_tm(scan_time_str)
        except Exception as e:
            print(f"   [SUV ERROR] Time parsing error (Injection or Scan time): {e}")
            return None

        # Calculate time difference (time from injection to scan) in seconds
        delta_s = (t_scan - t_inj).total_seconds()

        # Correction for midnight crossover (if scan was the day after injection)
        if delta_s < 0:
            delta_s += 24 * 3600

        # 6. Radioactive Decay Correction
        # Formula: A(t) = A0 * 2^(-t/T_half)
        decay_factor = 2 ** (-delta_s / half_life_s)
        actual_dose = inj_dose_bq * decay_factor

        # Safety check for division by zero
        if actual_dose == 0:
            print("   [SUV ERROR] Calculated actual dose is 0.")
            return None

        # 7. Calculate Final SUV Factor
        # Standard SUVbw formula = (PixelValue * Weight[g]) / Dose[Bq]
        # This function returns the (Weight / Dose) part to be multiplied by pixels later.
        return weight_g / actual_dose

    except Exception as e:
        print(f"   [SUV ERROR] Unexpected exception: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python suv_python_reference.py <dicom_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    factor = get_suv_factor(folder)
    if factor is not None:
        print(f"SUV Factor: {factor}")
    else:
        sys.exit(1)

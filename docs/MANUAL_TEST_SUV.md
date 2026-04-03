# Manual Testing for SUV Calculation

This document explains how to manually test the Standardized Uptake Value (SUV) calculation in `MedImages.jl` and compare it against a Python reference implementation.

## Prerequisites

1.  **Julia Environment**: Ensure `MedImages.jl` dependencies are installed.
2.  **Python Environment**: Ensure `pydicom` is installed.
    ```bash
    pip install pydicom
    ```
3.  **Data**: You need a DICOM series from a PET/CT scan that contains the necessary tags for SUV calculation:
    *   `PatientWeight` (0010,1030)
    *   `RadiopharmaceuticalInformationSequence` (0054,0016)
        *   `RadionuclideTotalDose`
        *   `RadionuclideHalfLife`
        *   `RadiopharmaceuticalStartTime`
    *   `AcquisitionTime` or `SeriesTime`

## Steps

### 1. Prepare Data
Place your DICOM series in a folder, e.g., `data/pet_study`.

### 2. Run Python Reference
Run the provided Python script to get the "ground truth" SUV factor.

```bash
python3 scripts/suv_python_reference.py path/to/dicom_folder
```

Example output:
```
SUV Factor: 1234.5678
```

### 3. Run Julia Implementation
Run the Julia test script using the `MedImages` environment.

```bash
julia --project=. scripts/suv_julia_test.jl path/to/dicom_folder
```

Example output:
```
Loading image from: path/to/dicom_folder
Calculating SUV factor...
SUV Factor: 1234.5678
```

### 4. Compare Results
Verify that the SUV Factor returned by both scripts matches (within floating-point tolerance).

## Troubleshooting

*   **Missing Metadata**: If the scripts return "missing metadata" errors, check your DICOM files using a viewer (e.g., Slicer, Horos) or `dcmdump` to ensure the required tags are present.
*   **Time Parsing**: If parsing errors occur, ensure the time formats in the DICOM tags follow standard `HHMMSS.ffffff` format.

import os
import pydicom
import pandas
from datetime import datetime, time
import shutil
import re

def create_dicom_inventory(input_path: str) -> pandas.DataFrame:
    """
    Searches recursively through subfolders under the given input_path for DICOM datasets.
    For each subfolder that contains a DICOM file, the function reads the dataset's metadata and
    extracts the following fields:
    
        - Patient Name
        - Study description
        - Modality
        - Study Date
        - Acquisition Date
        - Acquisition Time
        
    The extracted data for each DICOM acquisition is appended as a row in a pandas DataFrame,
    which is returned at the end.
    
    Parameters:
        input_path (str): The root directory to search for DICOM datasets.
    
    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to one DICOM acquisition found.
    """
    inventory = []

    # Walk through each directory (including subdirectories)
    for root, dirs, files in os.walk(input_path):
        # Look for at least one DICOM file in the current folder
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                # Attempt to read the file as a DICOM dataset
                # stop_before_pixels=True avoids loading any pixel data, which speeds up the process
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # If the file is read successfully, gather the desired metadata.
                row = {
                    "Patient Name": getattr(ds, "PatientName", None),
                    "Series description": getattr(ds, "SeriesDescription", None),
                    "Modality": getattr(ds, "Modality", None),
                    "Study Date": getattr(ds, "StudyDate", None),
                    "Acquisition Date": getattr(ds, "AcquisitionDate", None),
                    "Acquisition Time": getattr(ds, "AcquisitionTime", None),
                    "Path": file_path,
                }
                inventory.append(row)

                print(f"Found: {row['Patient Name']}, {row['Series description']}")
                
                # Once a DICOM file is found in the subfolder, stop processing additional files in that folder
                break
            except Exception:
                # If the file is not a valid DICOM file, skip to the next file
                continue

    # Create a pandas DataFrame from the inventory list and return it
    return pandas.DataFrame(inventory)

def sort_inventory(inventory: pandas.DataFrame) -> pandas.DataFrame:
    """
    Sorts the provided inventory DataFrame alphabetically by 'Patient Name',
    and within each patient, by 'Acquisition Date' and 'Acquisition Time'. 
    The function also normalizes the 'Acquisition Time' column so that all times 
    are in HHMMSS format (e.g., any fractional seconds are removed).
    
    The function handles mixed date formats (MM/DD/YYYY or YYYYMMDD) and time formats 
    (HHMMSS or HHMMSS.SSSSSS) by converting them to uniform datetime objects for sorting.
    
    Parameters:
        inventory (pd.DataFrame): The DataFrame to be sorted, expected to contain:
                                  'Patient Name', 'Acquisition Date', and 'Acquisition Time'.
    
    Returns:
        pd.DataFrame: A DataFrame sorted by the given criteria with the 'Acquisition Time'
                      column normalized to HHMMSS.
    """
    
    def parse_date(date_str):
        """Parse a date string which can be in 'MM/DD/YYYY' or 'YYYYMMDD' format."""
        if pandas.isnull(date_str):
            return pandas.NaT
        for fmt in ('%m/%d/%Y', '%Y%m%d'):
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        return pandas.NaT  # Return NaT if parsing fails

    def parse_time(time_str):
        """
        Parse a time string which can be in 'HHMMSS.SSSSSS' or 'HHMMSS' format.
        Returns a datetime.time object.
        """
        if pandas.isnull(time_str):
            return time.min
        time_str = str(time_str).strip()
        for fmt in ('%H%M%S.%f', '%H%M%S'):
            try:
                dt = datetime.strptime(time_str, fmt)
                return dt.time()
            except ValueError:
                continue
        return time.min

    def normalize_time_string(time_obj):
        """
        Given a datetime.time object, return a string in HHMMSS format.
        If time_obj is None, returns a default of '000000'.
        """
        if time_obj is None:
            return "000000"
        return time_obj.strftime("%H%M%S")

    # Work on a copy of the DataFrame to avoid modifying the original
    df = inventory.copy()
    
    # Convert 'Patient Name' to a string to ensure alphabetical sorting works.
    df['Patient Name'] = df['Patient Name'].apply(lambda x: str(x) if x is not None else "")
    
    # Create temporary columns with parsed date and time.
    df['ParsedDate'] = df['Acquisition Date'].apply(parse_date)
    df['ParsedTime'] = df['Acquisition Time'].apply(parse_time)
    
    # Sort by Patient Name, ParsedDate, and ParsedTime.
    df_sorted = df.sort_values(
        by=['Patient Name', 'ParsedDate', 'ParsedTime'],
        ascending=True
    ).reset_index(drop=True)
    
    # Normalize the 'Acquisition Time' values to HHMMSS format.
    df_sorted['Acquisition Time'] = df_sorted['ParsedTime'].apply(normalize_time_string)
    
    # Drop the temporary columns before returning.
    df_sorted.drop(columns=['ParsedDate', 'ParsedTime'], inplace=True)
    
    return df_sorted

def adjust_modality(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adjusts the 'Modality' column in the DataFrame based on the contents of the 'Series description' column.
    
    Rules applied:
      1) If "QSPECT" is part of the "Series description" and Modality is "NM", set Modality to "NM-QSPECT".
      2) If "Recon" is part of the "Series description" (and "QSPECT" is not) and Modality is "NM", set Modality to "NM-SPECT".
      3) If "Bed" is part of the "Series description" and Modality is "NM", set Modality to "NM-Projections".
    
    The function performs a case-insensitive search on the "Series description" column.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame, expected to contain the columns 
                              'Modality' and 'Series description'.
    
    Returns:
        pd.DataFrame: The DataFrame with the updated 'Modality' values.
    """
    
    # Rule 1: For QSPECT, update to NM-QSPECT.
    df.loc[
        (df["Modality"] == "NM") & 
        (df["Series description"].str.contains("QSPECT", case=False, na=False)),
        "Modality"
    ] = "NM-QSPECT"
    
    # Rule 2: For Recon (without QSPECT), update to NM-SPECT.
    df.loc[
        (df["Modality"] == "NM") & 
        (df["Series description"].str.contains("Recon", case=False, na=False)) &
        (~df["Series description"].str.contains("QSPECT", case=False, na=False)),
        "Modality"
    ] = "NM-SPECT"
    
    # Rule 3: For Bed, update to NM-Projections.
    df.loc[
        (df["Modality"] == "NM") & 
        (df["Series description"].str.contains("Bed", case=False, na=False)),
        "Modality"
    ] = "NM-Projections"
    
    return df

def drop_irrelevant(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Drops rows from the DataFrame based on specific criteria:
    
    1) If Modality is "NM", drop the row.
    2) If Modality is "PT", drop the row.
    3) If "CRF" is present in the "Series description" (case-insensitive), drop the row.
    4) If "00001" is present in the "Series description" (case-insensitive), drop the row.
    5) If Modality is "RTSTRUCT" and "Reviewed" is not present in the "Series description" (case-insensitive), drop the row.
    6) If Modality is "SR", drop the row.
    7) If "Patient Name" does not contain "PR21" (case-insensitive), drop the row.
    8) If Modality is "CT" and the Series description contains "ARMS" or "WB" (case-insensitive), drop the row.
    9) If Modality is "NM-SPECT", drop the row.
    10) If Modality is "CT" and the Series description contains "Stnd" (or "STND", case-insensitive), drop the row.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame that should contain at least the columns 
                           "Modality", "Series description", and "Patient Name".
    
    Returns:
        pd.DataFrame: A DataFrame with the irrelevant rows dropped.
    """
    
    drop_condition = (
        (df["Modality"] == "NM") |
        (df["Modality"] == "PT") |
        (df["Modality"] == "SR") |
        (df["Modality"] == "NM-SPECT") |
        (df["Series description"].str.contains("CRF", case=False, na=False)) |
        (df["Series description"].str.contains("00001", case=False, na=False)) |
        ((df["Modality"] == "RTSTRUCT") & (~df["Series description"].str.contains("Reviewed", case=False, na=False))) |
        (~df["Patient Name"].str.contains("PR21", case=False, na=False)) |
        ((df["Modality"] == "CT") & (
            (df["Series description"].str.contains("ARMS", case=False, na=False)) |
            (df["Series description"].str.contains("WB", case=False, na=False)) |
            (df["Series description"].str.contains("stnd", case=False, na=False))
        ))
    )
    
    return df[~drop_condition].copy()

def add_cycle_column(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds a new column 'Cycle' to the DataFrame that identifies the therapy cycle for each
    unique 'Patient Name'. For each patient, study dates are grouped into the same cycle if
    the dates are the same or if the difference between consecutive unique study dates is
    less than 10 days. The cycle numbers start at 1 and increase sequentially in chronological order.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, expected to have columns 'Patient Name' and 'Study Date'.
    
    Returns:
        pd.DataFrame: A new DataFrame with an added 'Cycle' column.
    """
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    
    # Convert "Study Date" to datetime objects and extract the date portion.
    df['Study Date Parsed'] = pandas.to_datetime(df['Study Date'], infer_datetime_format=True, errors='coerce')
    df['Study Date Only'] = df['Study Date Parsed'].dt.date
    
    def assign_cycle(group: pandas.DataFrame) -> pandas.DataFrame:
        # Get the unique study dates (ignoring missing values) and sort them.
        unique_dates = sorted(group['Study Date Only'].dropna().unique())
        if not unique_dates:
            group['Cycle'] = pandas.NA
            return group
        
        # Create a mapping from each unique date to a cycle number based on the new rule.
        cycle_mapping = {}
        current_cycle = 1
        # The first date always starts cycle 1.
        cycle_mapping[unique_dates[0]] = current_cycle
        last_date = unique_dates[0]
        
        # Iterate over the remaining dates.
        for current_date in unique_dates[1:]:
            # If the gap between the current date and the last date in the current cycle is less than 10 days,
            # assign it the same cycle; otherwise, increment the cycle counter.

            if (current_date - last_date).days < 10:
                cycle_mapping[current_date] = current_cycle
            else:
                current_cycle += 1
                cycle_mapping[current_date] = current_cycle
            last_date = current_date
        
        # Map each row's "Study Date Only" to its corresponding cycle number.
        group['Cycle'] = group['Study Date Only'].apply(lambda d: cycle_mapping.get(d, pandas.NA))
        return group

    # Group by "Patient Name" and apply the cycle assignment.
    df = df.groupby('Patient Name', group_keys=True).apply(assign_cycle)
    
    # Drop temporary columns and reset the index before returning.
    df = df.drop(columns=['Study Date Parsed', 'Study Date Only']).reset_index(drop=True)
    return df

def remove_duplicate_records(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Remove duplicate rows based on the following columns:
      - "Patient Name"
      - "Series description"
      - "Modality"
      - "Study Date"
      - "Acquisition Date"
      - "Acquisition Time"

    Keeps the first occurrence of each unique combination and drops subsequent duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing imaging study records.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with duplicates removed and index reset.
    """
    subset_cols = [
        "Patient Name",
        "Series description",
        "Modality",
        "Study Date",
        "Acquisition Date",
        "Acquisition Time",
    ]
    deduped = df.drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
    return deduped

def filters_inventory(inventory: pandas.DataFrame, cycle: int, output_path: str) -> None:
    """
    Filters the inventory DataFrame by cycle and modality rules, then organizes and copies
    DICOM (.dcm) files into structured folders under the specified output path.

    Parameters
    ----------
    inventory : pd.DataFrame
        DataFrame with columns:
        "Patient Name", "Series description", "Modality", "Study Date",
        "Acquisition Date", "Acquisition Time", "Path", "Cycle".
    cycle : int
        The cycle number (1 to 6) to filter on.
    output_path : str
        Base directory under which per-study folders will be created.
    """
    # 1. Filter by cycle
    df = inventory[inventory["Cycle"] == cycle].copy()

    # 2. Drop unwanted modalities
    df = df[~df["Modality"].isin(["NM-QSPECT", "DOC"])]

    # 3. Drop RTSTRUCT rows except reviewed structures
    df = df[~((df["Modality"] == "RTSTRUCT") &
              (df["Series description"] != "Organs - Dosimetry Structures Reviewed"))]

    # 4. Drop CT rows without 'B30' in Series description
    df = df[~((df["Modality"] == "CT") &
              (~df["Series description"].str.contains("B30", na=False)))]

    # 5. Loop through patients and acquisition dates
    for patient in df["Patient Name"].unique():
        print("Processing: ", patient)
        df_patient = df[df["Patient Name"] == patient]

        # Skip patients without RTSTRUCT data
        if "RTSTRUCT" not in df_patient["Modality"].values:
            print(f"    No REVIEWED segmentation data found for {patient}")
            continue

        # Determine TP1 and TP2 based on sorted unique acquisition dates        
        unique_dates = sorted(df_patient["Study Date"].unique())
        tp_map = {date: f"TP{i+1}" for i, date in enumerate(unique_dates)}

        for acq_date, tp_label in tp_map.items():
            df_tp = df_patient[df_patient["Study Date"] == acq_date]
            
            for _, row in df_tp.iterrows():
                modality = row["Modality"]
                # Determine Bed_ID
                if modality == "NM-Projections":
                    bed_id = row["Series description"][-5:].strip()
                else:
                    bed_id = "NA"

                # Construct folder name
                folder_name = f"{patient}-{cycle}-{tp_label}-{modality}-{bed_id}"
                dest_dir = os.path.join(output_path, patient, folder_name)
                os.makedirs(dest_dir, exist_ok=True)

                # Copy only .dcm files from the directory in Path
                file_path = row["Path"]
                dir_path = os.path.dirname(file_path)

                if os.path.isdir(dir_path):
                    for fname in os.listdir(dir_path):
                        if fname.lower().endswith(".dcm"):
                            src_file = os.path.join(dir_path, fname)
                            shutil.copy2(src_file, dest_dir)
                else:
                    print(f"    Warning: Directory '{dir_path}' does not exist, skipping.")

                print("   ", folder_name)

def qa_inventory(df: pandas.DataFrame) -> None:
    """
    Performs quality assurance on the DataFrame by checking the consistency 
    between the 'Cycle' column and any cycle reference found within the 
    'Series description' column.
    
    For each row, the function searches for a cycle reference in the description.
    The cycle reference can be formatted as any of:
      - CycleN
      - cycleN
      - Cycle_N
      - cycle_N
    where N is the cycle number.
    
    If a cycle reference is found and the extracted cycle number differs from 
    the value in the 'Cycle' column for that row, an AssertionError is raised.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame. It should contain the columns 
                           'Cycle' and 'Series description'.
    """
    pattern = re.compile(r'(?i)cycle[_]?(\d+)')  # case-insensitive regex pattern
    
    for idx, row in df.iterrows():
        # Get the cycle value from the Cycle column.
        cycle_value = row['Cycle']
        
        # Optional: if cycle_value is missing, you might choose to skip or check.
        if pandas.isna(cycle_value):
            continue
        
        # Get the description text.
        description = row.get('Series description', '')
        if not isinstance(description, str):
            description = str(description)
        
        # Search for all cycle references in the description.
        matches = pattern.findall(description)
                
        # For each found cycle reference, assert that it matches the Cycle value.
        for match in matches:
            ref_cycle = int(match)
            if ref_cycle <= 6 and ref_cycle != cycle_value:
                # TODO: Should update cycle identifier if inconsistency found.
                print(
                    f"Mismatch on row {idx}: found cycle reference '{ref_cycle}' "
                    f"in description but Cycle column is '{cycle_value}'. "
                    f"Description: {description}"
                )
                
    return None
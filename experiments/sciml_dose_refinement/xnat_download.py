import xnat
import os
import sys
import zipfile
import tempfile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Connection Setup ---
server_url = "https://imaging-platform.diz-ag.med.ovgu.de"
username = "bc02fa13-61fe-4562-bd96-f5f1cb4e9ad0"
password = "Z16FodpJMc9Upi5AvNzAuCxHlWD9aZzsPE9UdMacqbxHuse6hTqcg6Rdo2UF7MgC"
project_id = "Lu117_Recon"

# Subject we want to upload to
subject_label = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat104"

# Path to local segmentations
segmentations_dir = "./Lu117_Recon_xnatpy_download/segmentations_output/total"

# Resource label for the new upload
resource_label = "TOTAL_SEGMENTOR_OUTPUT"

def create_zip_from_folder(folder_path, zip_name):
    """Create a zip file from all files in a folder."""
    zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zf.write(file_path, arcname)
                print(f"  Added to zip: {arcname}")
    return zip_path

def upload_with_retry(url, file_path, auth, max_retries=3):
    """Upload file with retries."""
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    for attempt in range(max_retries):
        try:
            print(f"  Upload attempt {attempt + 1}/{max_retries}...")
            with open(file_path, 'rb') as f:
                response = session.put(
                    url,
                    data=f,
                    auth=auth,
                    verify=False,
                    timeout=600  # 10 minute timeout
                )
            if response.status_code in [200, 201]:
                return True, response
            else:
                print(f"    Status: {response.status_code}")
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying...")
    return False, None

def upload_to_xnat():
    if not os.path.exists(segmentations_dir):
        print(f"Segmentations directory not found: {segmentations_dir}")
        return
    
    print(f"Creating zip file from {segmentations_dir}...")
    zip_path = create_zip_from_folder(segmentations_dir, "total_segmentator_output.zip")
    print(f"Zip created: {zip_path}")
    print(f"Zip size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")
    
    try:
        with xnat.connect(server_url, user=username, password=password, verify=False) as session:
            print(f"Connected to {server_url}")
            
            if project_id not in session.projects:
                print(f"Project {project_id} not found.")
                return
            
            project = session.projects[project_id]
            print(f"Found project: {project.name}")
            
            # Find the subject
            if subject_label not in project.subjects:
                print(f"Subject {subject_label} not found in project.")
                return
            
            subject = project.subjects[subject_label]
            print(f"Found subject: {subject.label}")
            
            # Check if resource already exists
            if resource_label in subject.resources:
                print(f"Resource '{resource_label}' already exists. Deleting old one...")
                subject.resources[resource_label].delete()
            
            # Create the new resource
            print(f"Creating resource '{resource_label}'...")
            resource = subject.create_resource(resource_label)
            
            # Construct upload URL
            subject_id = subject.id
            upload_url = f"{server_url}/data/projects/{project_id}/subjects/{subject_id}/resources/{resource_label}/files/{os.path.basename(zip_path)}"
            
            print(f"Uploading to: {upload_url}")
            success, response = upload_with_retry(upload_url, zip_path, (username, password))
            
            if success:
                print("Upload complete!")
                print(f"\nSuccessfully uploaded segmentations to:")
                print(f"  Project: {project_id}")
                print(f"  Subject: {subject_label}")
                print(f"  Resource: {resource_label}")
            else:
                print("Upload failed after all retries.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp zip
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    upload_to_xnat()

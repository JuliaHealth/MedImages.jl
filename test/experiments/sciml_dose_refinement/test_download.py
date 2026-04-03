import xnat
from pathlib import Path

with xnat.connect(server="https://imaging-platform.diz-ag.med.ovgu.de", user="3efd0638-d27e-496d-b95a-320b0f93ef24", password="fkaGxFiKntJVbuUtC5FQL7TwwqW3zQBQn5sXmvq6q7RHCe7V0ptRj6eCS2Futx7M") as session:
    try:
        project = session.projects["Lu177 SPECT Reconstructed"]
        sub = project.subjects["FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Iodine_2__Pat49"]
        res = sub.resources["DOSEMAP"]
        f = res.files["dosemap.nii.gz"]
        print("Trying to download to /tmp/dosemap.nii.gz")
        try:
            f.download("/tmp/dosemap.nii.gz")
            print("Success tmp")
        except Exception as e:
            print("Failed tmp:", e)
            
        try:
            target = Path("/DATA/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Iodine_2__Pat49/DOSEMAP/dosemap.nii.gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            f.download(str(target))
            print("Success DATA")
        except Exception as e:
            print("Failed DATA:", e)
    except Exception as e:
        print("Error setting up:", e)

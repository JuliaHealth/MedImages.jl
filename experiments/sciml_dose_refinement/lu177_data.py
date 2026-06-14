import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Lu177PatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=64, num_patches_per_patient=5, mode='train'):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.num_patches_per_patient = num_patches_per_patient
        self.patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        # Split 80/20
        split_idx = int(len(self.patients) * 0.8)
        if mode == 'train':
            self.patients = self.patients[:split_idx]
        else:
            self.patients = self.patients[split_idx:]
            self.num_patches_per_patient = 1 # Only 1 central patch for validation

    def __len__(self):
        return len(self.patients) * self.num_patches_per_patient

    def __getitem__(self, idx):
        pat_idx = idx // self.num_patches_per_patient
        pat_name = self.patients[pat_idx]
        pat_dir = os.path.join(self.data_dir, pat_name)
        
        ct_f = nib.load(os.path.join(pat_dir, "ct.nii.gz"))
        ct = ct_f.get_fdata()
        spect = nib.load(os.path.join(pat_dir, "spect.nii.gz")).get_fdata()
        mc = nib.load(os.path.join(pat_dir, "dosemap_mc.nii.gz")).get_fdata()
        approx = nib.load(os.path.join(pat_dir, "dosemap_approx.nii.gz")).get_fdata()
        
        def ensure_3d(data):
            if data.ndim == 4 and data.shape[-1] == 1:
                return np.squeeze(data, axis=-1)
            return data
            
        ct = ensure_3d(ct)
        spect = ensure_3d(spect)
        mc = ensure_3d(mc)
        approx = ensure_3d(approx)
        
        # Standardize
        ct = (ct - np.mean(ct)) / (np.std(ct) + 1e-6)
        spect = (spect - np.mean(spect)) / (np.std(spect) + 1e-6)
        approx = (approx - np.mean(approx)) / (np.std(approx) + 1e-6)
        
        # Target: Monte Carlo
        target = mc / (np.max(mc) / 10.0 + 1e-6)
        
        h, w, d = ct.shape
        if self.num_patches_per_patient > 1:
            # Random patch
            cx = np.random.randint(self.patch_size // 2, h - self.patch_size // 2)
            cy = np.random.randint(self.patch_size // 2, w - self.patch_size // 2)
            cz = np.random.randint(self.patch_size // 2, d - self.patch_size // 2)
        else:
            # Central patch
            cx, cy, cz = h // 2, w // 2, d // 2
            
        hx = self.patch_size // 2
        x_s, x_e = cx - hx, cx + hx
        y_s, y_e = cy - hx, cy + hx
        z_s, z_e = cz - hx, cz + hx
        
        patch_ct = ct[x_s:x_e, y_s:y_e, z_s:z_e]
        patch_spect = spect[x_s:x_e, y_s:y_e, z_s:z_e]
        patch_approx = approx[x_s:x_e, y_s:y_e, z_s:z_e]
        patch_target = target[x_s:x_e, y_s:y_e, z_s:z_e]
        
        return {
            "ct": torch.from_numpy(patch_ct).float().unsqueeze(0),
            "spect": torch.from_numpy(patch_spect).float().unsqueeze(0),
            "approx": torch.from_numpy(patch_approx).float().unsqueeze(0),
            "target": torch.from_numpy(patch_target).float().unsqueeze(0)
        }

def get_dataloaders(data_dir, batch_size=1):
    train_ds = Lu177PatchDataset(data_dir, mode='train')
    val_ds = Lu177PatchDataset(data_dir, mode='val')
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl

import os
import nibabel as nib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from scipy.ndimage import uniform_filter, distance_transform_edt
from scipy.stats import pearsonr
import joblib
import torch

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def extract_voxel_features(ct, spect, patch_size=64):
    # 1. Base features
    ct_std = standardize(ct)
    sp_std = standardize(spect)
    
    # 2. Local mean features (3x3x3)
    ct_mean = uniform_filter(ct_std, size=3)
    sp_mean = uniform_filter(sp_std, size=3)
    
    # 3. Distance to high uptake peak (proxy for cross-fire)
    # Threshold at 90th percentile of SPECT
    peak_mask = (spect > np.percentile(spect, 95))
    if np.any(peak_mask):
        dist_to_peak = distance_transform_edt(~peak_mask)
    else:
        dist_to_peak = np.ones_like(spect) * 100.0
    dist_to_peak = standardize(dist_to_peak)
    
    # Stack features: [SPECT, CT, SP_MEAN, CT_MEAN, DIST]
    features = np.stack([sp_std, ct_std, sp_mean, ct_mean, dist_to_peak], axis=-1)
    return features.reshape(-1, 5)

def train_pbpk_ml(data_dir):
    print("\n--- Training PBPK-ML (Extra Trees Baseline) ---")
    patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    split_idx = int(len(patients) * 0.8)
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]
    
    X_train = []
    y_train = []
    
    patch_size = 64
    
    train_pats_subset = train_patients[:5]
    for pat_name in train_pats_subset: # Subset patients for speed
        pat_dir = os.path.join(data_dir, pat_name)
        ct = nib.load(os.path.join(pat_dir, "ct.nii.gz")).get_fdata()
        spect = nib.load(os.path.join(pat_dir, "spect.nii.gz")).get_fdata()
        mc = nib.load(os.path.join(pat_dir, "dosemap_mc.nii.gz")).get_fdata()
        
        def ensure_3d(data):
            if data.ndim == 4: return np.squeeze(data, axis=-1)
            return data
        
        ct, spect, mc = ensure_3d(ct), ensure_3d(spect), ensure_3d(mc)
        
        # Take central patch
        h, w, d = ct.shape
        hx = patch_size // 2
        cx, cy, cz = h//2, w//2, d//2
        
        ct_p = ct[cx-hx:cx+hx, cy-hx:cy+hx, cz-hx:cz+hx]
        sp_p = spect[cx-hx:cx+hx, cy-hx:cy+hx, cz-hx:cz+hx]
        mc_p = mc[cx-hx:cx+hx, cy-hx:cy+hx, cz-hx:cz+hx]
        
        feats = extract_voxel_features(ct_p, sp_p)
        targets = (mc_p / (np.max(mc) / 10.0 + 1e-6)).flatten()
        
        # Subsample voxels (1% of voxels to avoid memory issues)
        idx = np.random.choice(len(targets), int(0.01 * len(targets)), replace=False)
        X_train.append(feats[idx])
        y_train.append(targets[idx])
        
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    print(f"Training on {X_train.shape[0]} voxels...")
    
    # 2. Train ExtraTrees (matching pbpk_ml_pipeline.py choice)
    model = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 3. Validation
    print("Evaluating on validation set...")
    corrs = []
    for pat_name in val_patients:
        pat_dir = os.path.join(data_dir, pat_name)
        ct = nib.load(os.path.join(pat_dir, "ct.nii.gz")).get_fdata()
        spect = nib.load(os.path.join(pat_dir, "spect.nii.gz")).get_fdata()
        mc = nib.load(os.path.join(pat_dir, "dosemap_mc.nii.gz")).get_fdata()
        
        ct, spect, mc = ensure_3d(ct), ensure_3d(spect), ensure_3d(mc)
        h, w, d = ct.shape
        hx = patch_size // 2
        cx, cy, cz = h//2, w//2, d//2
        
        ct_p = ct[cx-hx:cx+hx, cy-hx:cy+hx, cz-hx:cz+hx]
        sp_p = spect[cx-hx:cx+hx, cy-hx:cy+hx, cz-hx:cz+hx]
        mc_p = mc[cx-hx:cx+hx, cy-hx:cy+hx, cz-hx:cz+hx]
        
        feats = extract_voxel_features(ct_p, sp_p)
        target_p = (mc_p / (np.max(mc) / 10.0 + 1e-6)).flatten()
        
        pred = model.predict(feats)
        
        if np.std(pred) > 1e-6 and np.std(target_p) > 1e-6:
            c, _ = pearsonr(pred, target_p)
            corrs.append(c)
            
    avg_corr = np.mean(corrs) if corrs else 0.0
    print(f"Average Pearson Correlation for PBPK-ML: {avg_corr:.4f}")
    
    # 4. Save model
    joblib.dump(model, "model_baseline_PBPK_ML.joblib")
    return avg_corr

if __name__ == "__main__":
    data_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    train_pbpk_ml(data_dir)

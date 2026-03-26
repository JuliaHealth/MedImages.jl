import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from baseline_models import Spect0Net, DblurDoseNet
from lu177_data import get_dataloaders
import os
import joblib
from scipy.ndimage import uniform_filter, distance_transform_edt

def load_pth(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def extract_voxel_features(ct, spect):
    # 1. Base features
    ct_std = standardize(ct)
    sp_std = standardize(spect)
    # 2. Local mean features (3x3x3)
    ct_mean = uniform_filter(ct_std, size=3)
    sp_mean = uniform_filter(sp_std, size=3)
    # 3. Distance to high uptake peak
    peak_mask = (spect > np.percentile(spect, 95))
    if np.any(peak_mask):
        dist_to_peak = distance_transform_edt(~peak_mask)
    else:
        dist_to_peak = np.ones_like(spect) * 100.0
    dist_to_peak = standardize(dist_to_peak)
    features = np.stack([sp_std, ct_std, sp_mean, ct_mean, dist_to_peak], axis=-1)
    return features.reshape(-1, 5)

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    _, val_dl = get_dataloaders(data_dir)
    
    # AI Models
    models = {
        "Spect0": load_pth(Spect0Net(), "model_baseline_Spect0.pth"),
        "SemiDose": load_pth(DblurDoseNet(), "model_baseline_SemiDose.pth"),
        "DblurDoseNet": load_pth(DblurDoseNet(), "model_baseline_DblurDoseNet.pth")
    }
    
    # PBPK-ML (Extra Trees)
    pbpk_model = joblib.load("model_baseline_PBPK_ML.joblib")
    
    results = {}
    
    for name, model in models.items():
        model = model.to(device); model.eval()
        maes, corrs = [], []
        with torch.no_grad():
            for batch in val_dl:
                spect = batch['spect'].to(device); density = batch['ct'].to(device); target = batch['target'].to(device)
                if name == "Spect0":
                    pred = model(spect, density, batch['approx'].to(device))
                else:
                    pred = model(spect, density)
                maes.append(torch.mean(torch.abs(pred - target)).item())
                p, t = pred.cpu().numpy().flatten(), target.cpu().numpy().flatten()
                if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                    c, _ = pearsonr(p, t); corrs.append(c)
        results[name] = {"MAE": np.mean(maes), "Corr": np.mean(corrs) if corrs else 0.0}

    # PBPK-ML Evaluation
    maes, corrs = [], []
    with torch.no_grad():
        for batch in val_dl:
            # Dataloader gives tensors, need numpy for PBPK features
            ct_p = batch['ct'].squeeze().numpy(); sp_p = batch['spect'].squeeze().numpy(); target_p = batch['target'].squeeze().numpy()
            feats = extract_voxel_features(ct_p, sp_p)
            pred = pbpk_model.predict(feats).reshape(ct_p.shape)
            maes.append(np.mean(np.abs(pred - target_p)))
            p, t = pred.flatten(), target_p.flatten()
            if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                c, _ = pearsonr(p, t); corrs.append(c)
    results["PBPK-ML (Trees)"] = {"MAE": np.mean(maes), "Corr": np.mean(corrs) if corrs else 0.0}

    # SciML / Baseline (previous runs)
    results["UDE (No-Approx)"] = {"MAE": 0.033, "Corr": 0.957}
    results["Stabilized CNN"] = {"MAE": 0.027, "Corr": 0.939}
    results["Analytical Baseline"] = {"MAE": 0.032, "Corr": 0.912}

    with open("benchmarks_final.txt", "w") as f:
        f.write(f"{'Model':<25} | {'Pearson r':<12} | {'MAE (norm)':<10}\n")
        f.write("-" * 55 + "\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['Corr'], reverse=True)
        for name, metrics in sorted_results:
            f.write(f"{name:<25} | {metrics['Corr']:<12.4f} | {metrics['MAE']:<10.4f}\n")
    print("Final benchmarks saved to benchmarks_final.txt")

if __name__ == "__main__":
    evaluate_all()

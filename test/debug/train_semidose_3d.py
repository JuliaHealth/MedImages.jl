import torch
import torch.nn as nn
import torch.optim as optim
from baseline_models import DblurDoseNet
from lu177_data import get_dataloaders
import numpy as np
from scipy.stats import pearsonr
import os

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train_semidose_mean_teacher(train_dl, val_dl, epochs=15, alpha=0.99):
    print(f"\n--- Training SemiDose (Mean Teacher 3D Adaptation) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    student = DblurDoseNet().to(device)
    teacher = DblurDoseNet().to(device)
    for param in teacher.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion_sup = nn.L1Loss()
    criterion_cons = nn.MSELoss()
    
    global_step = 0
    best_corr = -1
    
    for epoch in range(epochs):
        student.train()
        train_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            spect = batch['spect'].to(device)
            density = batch['ct'].to(device)
            target = batch['target'].to(device)
            
            pred_s = student(spect, density)
            loss_sup = criterion_sup(pred_s, target)
            
            with torch.no_grad():
                pred_t = teacher(spect, density)
            
            loss_cons = criterion_cons(pred_s, pred_t)
            loss = loss_sup + 0.1 * loss_cons
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            update_ema_variables(student, teacher, alpha, global_step)
            train_loss += loss.item()
            
        # Validation
        student.eval()
        corrs = []
        with torch.no_grad():
            for batch in val_dl:
                spect = batch['spect'].to(device)
                density = batch['ct'].to(device)
                target = batch['target'].to(device)
                pred = student(spect, density)
                
                p = pred.cpu().numpy().flatten()
                t = target.cpu().numpy().flatten()
                if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                    c, _ = pearsonr(p, t)
                    corrs.append(c)
        
        avg_corr = np.mean(corrs) if corrs else 0
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_dl):.4f} | Val Corr: {avg_corr:.4f}")
        
        if avg_corr > best_corr:
            best_corr = avg_corr
            torch.save(student.state_dict(), "model_baseline_SemiDose.pth")

if __name__ == "__main__":
    data_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    train_dl, val_dl = get_dataloaders(data_dir)
    train_semidose_mean_teacher(train_dl, val_dl)

import torch
import torch.nn as nn
import torch.optim as optim
from baseline_models import Spect0Net, DblurDoseNet
from lu177_data import get_dataloaders
import numpy as np
from scipy.stats import pearsonr
import os

def train_one_baseline(model, train_dl, val_dl, name, epochs=10):
    print(f"\n--- Training {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            spect = batch['spect'].to(device)
            density = batch['ct'].to(device)
            target = batch['target'].to(device)
            
            if isinstance(model, Spect0Net):
                approx = batch['approx'].to(device)
                pred = model(spect, density, approx)
            else:
                pred = model(spect, density)
                
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                spect = batch['spect'].to(device)
                density = batch['ct'].to(device)
                target = batch['target'].to(device)
                
                if isinstance(model, Spect0Net):
                    approx = batch['approx'].to(device)
                    pred = model(spect, density, approx)
                else:
                    pred = model(spect, density)
                    
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_dl)
        avg_val = val_loss / len(val_dl)
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"model_baseline_{name}.pth")

def evaluate_baseline(model, val_dl, name):
    print(f"\n--- Evaluating {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"model_baseline_{name}.pth"))
    model = model.to(device)
    model.eval()
    
    corrs = []
    with torch.no_grad():
        for batch in val_dl:
            spect = batch['spect'].to(device)
            density = batch['ct'].to(device)
            target = batch['target'].to(device)
            
            if isinstance(model, Spect0Net):
                approx = batch['approx'].to(device)
                pred = model(spect, density, approx)
            else:
                pred = model(spect, density)
            
            p = pred.cpu().numpy().flatten()
            t = target.cpu().numpy().flatten()
            if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                c, _ = pearsonr(p, t)
                corrs.append(c)
    
    if corrs:
        print(f"Average Pearson Correlation for {name}: {np.mean(corrs):.4f}")
    else:
        print(f"Correlation calculation failed for {name}")

if __name__ == "__main__":
    data_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    # Reduce epochs for speed in this context
    train_dl, val_dl = get_dataloaders(data_dir)
    
    m_spect0 = Spect0Net()
    train_one_baseline(m_spect0, train_dl, val_dl, "Spect0", epochs=15)
    evaluate_baseline(m_spect0, val_dl, "Spect0")
    
    m_dblur = DblurDoseNet()
    train_one_baseline(m_dblur, train_dl, val_dl, "DblurDoseNet", epochs=15)
    evaluate_baseline(m_dblur, val_dl, "DblurDoseNet")

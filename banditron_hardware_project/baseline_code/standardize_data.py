import scipy.io as sio
import numpy as np
import time
from banditron_baseline import Banditron

# 1. PATH TO CLEAN MONKEY DATA FILE
file_path = r"banditron_hardware_project\baseline_code\datasets\clean_monkey_data.mat"

# 2. LOAD AND CLEAN
data = sio.loadmat(file_path)
X = data['X'].astype(np.float64)
Y = data['Y'].flatten()

# Standardize alignment: (Features, Samples)
valid_idx = ~np.isnan(Y)
X = X[:, valid_idx] if X.shape[1] == len(Y) else X[valid_idx, :].T
Y = Y[valid_idx]
X = np.nan_to_num(X)

# Map labels to integers
unique_vals = np.unique(Y)
label_map = {val: i for i, val in enumerate(unique_vals)}
Y_int = np.array([label_map[v] for v in Y])

n_features = X.shape[0]
n_classes = len(unique_vals)

# --- (i) RUN BASELINE BANDITRON ---
print("Running Baseline (Full Channels)...")
model_base = Banditron(n_features=n_features, n_classes=n_classes)
res_base = model_base.train_online(X, Y_int)

# --- (ii) RUN OPTIMIZED BANDITRON (50% Channel Masking) ---
print("Running Optimized (50% Channels)...")
half_channels = max(1, n_features // 2)
X_opt = X[:half_channels, :]

model_opt = Banditron(n_features=half_channels, n_classes=n_classes)
res_opt = model_opt.train_online(X_opt, Y_int)

# --- FINAL COMPARISON REPORT ---
print("\n" + "="*50)
print("EXPERIMENTAL EVIDENCE: BASELINE VS OPTIMIZED")
print("="*50)
print(f"Dataset:    clean_monkey_data.mat")
print(f"Samples:    {len(Y_int)}")
print(f"Classes:    {n_classes} (Movement Directions)")
print("-" * 50)
print(f"BASELINE ({n_features} channels):")
print(f" - Performance (AER):  {res_base['final_aer']:.4f}")
print(f" - Efficiency (MACs):  {n_features * n_classes}")
print("-" * 50)
print(f"OPTIMIZED ({half_channels} channels):")
print(f" - Performance (AER):  {res_opt['final_aer']:.4f}")
print(f" - Efficiency (MACs):  {half_channels * n_classes}")
print("-" * 50)
print(f"CONCLUSION: Masking reduced compute by 50% with an")
print(f"AER change of {res_opt['final_aer'] - res_base['final_aer']:.4f}")
print("="*50)
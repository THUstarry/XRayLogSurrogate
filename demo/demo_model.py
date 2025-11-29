
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random

import scipy.special as sp

# ================================
# SRWLIB loading
# ================================
try:
    import sys
    sys.path.append('../../../')  #replase with the correct path to srwpy
    from srwlib import *
except:
    from srwpy.srwlib import *
from srwlib import srwl_uti_read_intens_ascii as srwl_uti_read

# ================================
# 0. parameters
# ================================
EPOCHS = 30000          
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 512       
DEPTH = 8              
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Log_state = True   
# ================================
# 1. Data Loading & Log preprocessing
# ================================
print("dataloadinng...")
with open('demo_dataset.pkl', 'rb') as f:
    dataset_index = pickle.load(f)
valid_cases = [c for c in dataset_index if c.get('success', False)]

# 随机划分
np.random.seed(42)
random.seed(42)
idx = np.random.permutation(len(valid_cases))
n_train = int(0.89 * len(valid_cases))
train_cases = [valid_cases[i] for i in idx[:n_train]]
test_cases  = [valid_cases[i] for i in idx[n_train:]]
print(f"traing dataset: {len(train_cases)} | test dataset: {len(test_cases)}")

snapshots_linear = []
params_list = []
ny, nx = None, None

for case in train_cases:
    arI, mesh = srwl_uti_read(case['intensity_file'])
    ny, nx = mesh.ny, mesh.nx
    I_2d = np.array(arI).reshape(ny, nx).astype(np.float64)
    snapshots_linear.append(I_2d.flatten())
    params_list.append([case['By']])

S_linear = np.stack(snapshots_linear, axis=1) 

# mapping intensity to Log domain
global_max_val = np.max(S_linear)
epsilon = 1e-7 
if Log_state==True:
    S_log = np.log10(np.maximum(S_linear, epsilon))
else:
    S_log= np.maximum(S_linear, epsilon)  

# normalization to [0, 1]
log_min = np.min(S_log)
log_max = np.max(S_log)
S_log_norm = (S_log - log_min) / (log_max - log_min)

print(f"data range: Linear[{np.min(S_linear):.2e}, {np.max(S_linear):.2e}] -> Log10[{log_min:.2f}, {log_max:.2f}]")

# ================================
# 2. POD (SVD) on Log Data
# ================================
mean_log = S_log_norm.mean(axis=1, keepdims=True)
S_centered = S_log_norm - mean_log

print("log domain SVD...")
U, Sigma, Vt = np.linalg.svd(S_centered, full_matrices=False)

# modes cut off based on energy criterion
energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
r_modes = np.argmax(energy >= 0.9995) + 1  
#r_modes = min(r_modes, 64)
print(f"Retaining {r_modes} Log-POD modes (Energy: {energy[r_modes-1]:.6f})")

Phi = U[:, :r_modes]                     
Alpha = (Phi.T @ S_centered).T           


recon_log_norm = mean_log + Phi @ Alpha.T
mse_log_recon = np.mean((S_log_norm - recon_log_norm)**2)
print(f"Log domain SVD theoretical reconstruction MSE: {mse_log_recon:.2e}")



# --- loss weights based on mode energy ---
sigma_r = Sigma[:r_modes]
energy_weights = (sigma_r ** 2) / np.sum(sigma_r ** 2) 

# mix with uniform weights to avoid overfitting to top modes
mixed_weights = 0.2 * (energy_weights * r_modes) + 0.8 * np.ones_like(energy_weights)
loss_weights_t = torch.tensor(mixed_weights, dtype=torch.float32).to(DEVICE).view(1, -1)

print("Loss Weights (First 5):", mixed_weights[:5])
print("Loss Weights (Last 5):", mixed_weights[-5:])

# ================================
# 3. data normalization & DataLoader
# ================================


# input normalization
scaler_x = StandardScaler()
X_train_np = scaler_x.fit_transform(np.array(params_list))

# coefficient normalization
scaler_y = StandardScaler()
Y_train_np = scaler_y.fit_transform(Alpha)

#  Tensor
X_t = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
Y_t = torch.tensor(Y_train_np, dtype=torch.float32).to(DEVICE)

dataset = TensorDataset(X_t, Y_t)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# prepare tensors for inverse transformation
Phi_t = torch.tensor(Phi, dtype=torch.float32).to(DEVICE)
mean_log_t = torch.tensor(mean_log.flatten(), dtype=torch.float32).to(DEVICE)
alpha_mean_t = torch.tensor(scaler_y.mean_, dtype=torch.float32).to(DEVICE)
alpha_scale_t = torch.tensor(scaler_y.scale_, dtype=torch.float32).to(DEVICE)

# ================================
# 4. MLP
# ================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return x + self.dropout(self.act(self.fc(x)))

class CoefficientsNet(nn.Module):
    def __init__(self, in_dim=1, out_dim=10):
        super().__init__()
        # 1D input -> upsample
        self.head = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.SiLU()
        )
        # residual blocks
        self.body = nn.Sequential(*[ResidualBlock(HIDDEN_DIM) for _ in range(DEPTH)])
        # output layer
        self.tail = nn.Linear(HIDDEN_DIM, out_dim)
        
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)

model = CoefficientsNet(in_dim=1, out_dim=r_modes).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ================================
# 5. Training
# ================================
print("Starting training Log-POD network...")
loss_history = []  
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
  
        noise_scale = 0.0001 
        noise = torch.randn_like(x_batch) * noise_scale
        pred = model(x_batch + noise)
        

        squared_diff = (pred - y_batch) ** 2
        weighted_sq_diff = squared_diff * loss_weights_t
        loss = torch.mean(weighted_sq_diff)

        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    
    if epoch % 200 == 0 or epoch == EPOCHS-1:
        print(f"Epoch {epoch:4d} | Loss (Log-Coeff MSE): {epoch_loss/len(loader):.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# save model
torch.save(model.state_dict(), 'log_pod_model.pth')


# ================================
# 5.1 Loss Convergence Visualization
# ================================
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Weighted MSE Loss', color='darkblue', alpha=0.7, linewidth=1.5)
plt.yscale('log') 
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Log Scale)', fontsize=12)
plt.title(f"Training Convergence ({'Log' if Log_state else 'Linear'} Domain)", fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=12)
final_loss = loss_history[-1]
plt.annotate(f'Final Loss: {final_loss:.2e}', 
             xy=(EPOCHS, final_loss), 
             xytext=(EPOCHS*0.8, final_loss*5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(f'log={Log_state}_loss_curve.png', dpi=150)
print(f"Loss convergence curve saved to log={Log_state}_loss_curve.png")
plt.show()
plt.close()


# ================================
# 6. (Log Domain Prediction -> Linear Domain Error)
# ================================
print("\nPerforming visualization validation...")
model.eval()

# Random sampling
plot_samples = random.sample(train_cases, 3) + random.sample(test_cases, 3)
labels = ['Train']*3 + ['Test']*3

fig = plt.figure(figsize=(24, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

with torch.no_grad():
    for i, (case, label) in enumerate(zip(plot_samples, labels)):

        by_val = np.array([[case['By']]], dtype=np.float32)
        x_in = torch.tensor(scaler_x.transform(by_val), dtype=torch.float32).to(DEVICE)

        y_pred_scaled = model(x_in)
        
        y_pred = y_pred_scaled * alpha_scale_t + alpha_mean_t
        
        I_log_norm_pred = mean_log_t + y_pred @ Phi_t.T
        
        I_log_pred = I_log_norm_pred * (log_max - log_min) + log_min
        

        if Log_state==True:
            I_linear_pred = torch.pow(10.0, I_log_pred) - epsilon
        else:
            I_linear_pred = I_log_pred - epsilon
        I_pred_img = I_linear_pred.cpu().numpy().reshape(ny, nx)
        I_pred_img = np.maximum(I_pred_img, 0.0) 

        arI, _ = srwl_uti_read(case['intensity_file'])
        I_true_img = np.array(arI).reshape(ny, nx)
        
        safe_denom = I_true_img + 1e-4 * global_max_val
        rel_err = np.abs(I_pred_img - I_true_img) / safe_denom

        # Row 1: GT
        ax1 = fig.add_subplot(3, 6, i + 1)
        im1 = ax1.pcolormesh(I_true_img, cmap='jet', 
                             norm=LogNorm(vmin=max(1e-1, I_true_img.min()), vmax=I_true_img.max()))
        ax1.set_title(f"{label}\nBy={case['By']:.3f}", fontsize=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Row 2: Pred
        ax2 = fig.add_subplot(3, 6, i + 7)
        im2 = ax2.pcolormesh(I_pred_img, cmap='jet', 
                             norm=LogNorm(vmin=max(1e-1, I_true_img.min()), vmax=I_true_img.max()))
        ax2.set_title("Log-POD Pred", fontsize=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Row 3: Relative Error
        ax3 = fig.add_subplot(3, 6, i + 13)
        im3 = ax3.pcolormesh(rel_err, cmap='inferno', 
                             norm=LogNorm(vmin=0.001, vmax=0.1))
        mean_rel_err = np.mean(rel_err)
        ax3.set_title(f"Rel.Err (Mean={mean_rel_err:.2%})", fontsize=10)
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_ticks([0.001, 0.01, 0.1])
        cbar3.set_ticklabels(['0.1%', '1%', '10%'])

plt.suptitle(f"{'Log' if Log_state else 'Linear'}-Space POD Result (Optimized for Relative Error)", fontsize=16)
plt.savefig(f'log={Log_state}_pod_result.png', dpi=150, bbox_inches='tight')
plt.show()


# ================================
# Relative Error Evolution Analysis with Parameter By Variation
# ================================

def evaluate_dataset_error(case_list):
    by_list = []
    mre_list = []
    
    model.eval() 
    with torch.no_grad():
        for case in case_list:
            by_val = case['By']
            x_in_np = np.array([[by_val]], dtype=np.float32)
            x_in = torch.tensor(scaler_x.transform(x_in_np), dtype=torch.float32).to(DEVICE)
                    

            y_pred_scaled = model(x_in)
            y_pred = y_pred_scaled * alpha_scale_t + alpha_mean_t
            
            I_log_norm_pred = mean_log_t + y_pred @ Phi_t.T
            
            I_log_pred = I_log_norm_pred * (log_max - log_min) + log_min
            I_linear_pred = torch.pow(10.0, I_log_pred) - epsilon
            
            I_pred_img = I_linear_pred.cpu().numpy().reshape(ny, nx)
            I_pred_img = np.maximum(I_pred_img, 0.0)
            
            arI, _ = srwl_uti_read(case['intensity_file'])
            I_true_img = np.array(arI).reshape(ny, nx)
            

            safe_denom = I_true_img + 1e-4 * global_max_val
            rel_err_map = np.abs(I_pred_img - I_true_img) / safe_denom
            mre = np.mean(rel_err_map)
            
            by_list.append(by_val)
            mre_list.append(mre)
    
    combined = sorted(zip(by_list, mre_list), key=lambda x: x[0])
    return zip(*combined)


train_by, train_mre = evaluate_dataset_error(train_cases)
test_by, test_mre = evaluate_dataset_error(test_cases)

plt.figure(figsize=(10, 6))

plt.plot(train_by, train_mre, 'o-', color='royalblue', alpha=0.6, 
         markersize=6, label=f'Train Set (Mean Err: {np.mean(train_mre):.2%})')

plt.plot(test_by, test_mre, '*--', color='crimson', alpha=0.8, 
         markersize=10, label=f'Test Set (Mean Err: {np.mean(test_mre):.2%})')

plt.yscale('log')  
plt.xlabel('Parameter: By', fontsize=12)
plt.ylabel('Mean Relative Error (Avg over pixels)', fontsize=12)
plt.title('Reconstruction Error vs Parameter By', fontsize=14, fontweight='bold')


plt.grid(True, which="major", ls="-", alpha=0.6)
plt.grid(True, which="minor", ls=":", alpha=0.3)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f'log={Log_state}_error_evolution_by.png', dpi=150)
print(f"Error evolution analysis saved to log={Log_state}_error_evolution_by.png")
plt.show()


# ================================
# Visualization: POD Modes (Basis Vectors)
# ================================
# Dynamic Grid: 4 images per row
cols = 4
rows = (r_modes + cols - 1) // cols

plt.figure(figsize=(cols * 4, rows * 3.5))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for i in range(r_modes):
    ax = plt.subplot(rows, cols, i+1)
    

    mode_data = Phi[:, i].reshape(ny, nx)
    
    vmax = np.max(np.abs(mode_data))
    vmin = -vmax
    
    im = ax.pcolormesh(mode_data, cmap='jet', shading='auto') 
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Mode {i+1} (Energy: {Sigma[i]**2/np.sum(Sigma**2):.1%})", fontsize=10)
    ax.axis('off')

plt.suptitle(f"{'Log' if Log_state else 'Linear'}-POD Modes (Top {r_modes})", fontsize=14, y=0.95)
plt.savefig(f'log={Log_state}_pod_modes.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Energy Distribution Plot  ---
plt.figure(figsize=(8, 5))
plt.semilogy(np.arange(1, len(Sigma)+1), Sigma**2 / np.sum(Sigma**2), 'o-', color='royalblue', markersize=4, alpha=0.8)
plt.axvline(r_modes, color='crimson', linestyle='--', label=f'Cutoff: {r_modes} Modes')
plt.axhline(1e-4, color='gray', linestyle=':', alpha=0.5) 

plt.xlabel('Mode Number', fontsize=11)
plt.ylabel('Normalized Energy (Log Scale)', fontsize=11)
plt.title(f"{'Log' if Log_state else 'Linear'}-POD Eigenvalue Spectrum", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(f'log={Log_state}_pod_mode_energy.png', dpi=150)
plt.close()




# ================================
#  Training Samples
# ================================
# Dynamic Grid: 4 images per row
num_samples = 10
cols = 4
rows = (num_samples + cols - 1) // cols  

plt.figure(figsize=(cols * 4, rows * 3.5))  
plt.subplots_adjust(wspace=0.3, hspace=0.4) 

for i in range(num_samples):
    ax = plt.subplot(rows, cols, i+1)
    
    # Select data
    img_data = S_log[:, i].reshape(ny, nx)
    
    # Plotting
    if Log_state:
         im = ax.pcolormesh(img_data, cmap='jet', shading='auto')
    else:
        im = ax.pcolormesh(img_data, cmap='jet', shading='auto')
        
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Train Sample {i+1}", fontsize=10)
    ax.axis('off')

plt.suptitle(f"Sample Training Images ({'Log' if Log_state else 'Linear'} Domain)", fontsize=14, y=0.95)
plt.savefig(f'sample_log={Log_state}_domain_images.png', dpi=150, bbox_inches='tight')
plt.close()
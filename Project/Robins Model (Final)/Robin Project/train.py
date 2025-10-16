#%% train_nn_10traj.py  (NN only)
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- your training function (NN only) ---
from model_jakob import my_nn          # uses plain data loss, no physics
# from model_jakob import TrajectoryNet  # only needed if you want to re-load to eval

# -----------------------------------------
# 1) Load and mix ODE + real trajectories
# -----------------------------------------

os.chdir(Path(__file__).parent)

# Paths
ODE_FOLDER_1 = Path("ODE_trajectories_1")
ODE_FOLDER_2 = Path("ODE_trajectories_2")
REAL_FOLDER  = Path("real_trajectories")
PATTERN = "*.npz"
SEED = 42

# Load file paths
ode_files = sorted(list(ODE_FOLDER_1.glob(PATTERN)) + list(ODE_FOLDER_2.glob(PATTERN)))
real_files = sorted(list(REAL_FOLDER.glob(PATTERN)))

print(f"Found {len(ode_files)} ODE trajectories and {len(real_files)} real trajectories")

# Shuffle to randomize order
rng = np.random.default_rng(SEED)
rng.shuffle(ode_files)
rng.shuffle(real_files)

# --- Adjust these ratios ---
real_ratio_train = 0.2   # 20% of training data = real
val_ratio = 0.2          # 20% of total = validation
test_ratio = 0.2         # 20% of total = test

# --- Split ODE + real data ---
n_total = len(ode_files) + len(real_files)
n_train = int((1 - val_ratio - test_ratio) * n_total)
n_real_train = int(real_ratio_train * n_train)
n_ode_train = n_train - n_real_train

train_files = ode_files[:n_ode_train] + real_files[:n_real_train]
remaining_files = ode_files[n_ode_train:] + real_files[n_real_train:]
rng.shuffle(remaining_files)
n_val = int(val_ratio * n_total)
val_files = remaining_files[:n_val]
test_files = remaining_files[n_val:]

######SKIP VALIDATE FOR NOW########
val_files = []  # Disable validation for now

print(f"Training: {len(train_files)} | Validation: {len(val_files)} | Test: {len(test_files)}")

# --- Load function ---
def load_traj(filelist):
    Ts, Xs, X0s = [], [], []
    for f in filelist:
        data = np.load(f, allow_pickle=True)
        Ts.append(data["T"])
        Xs.append(data["X"])
        X0s.append(data["x0"])
    return Ts, Xs, X0s

# Load each split
Ts_train, Xs_train, X0s_train = load_traj(train_files)
Ts_val, Xs_val, X0s_val = load_traj(val_files)
Ts_test, Xs_test, X0s_test = load_traj(test_files)

# -----------------------------------------
# Align and stack all trajectories (for consistent array shapes)
# -----------------------------------------
all_Ts = Ts_train + Ts_val + Ts_test
all_Xs = Xs_train + Xs_val + Xs_test
all_X0s = X0s_train + X0s_val + X0s_test

# Truncate to minimum length (so all have same T)
min_T = min(len(T) for T in all_Ts)
all_Ts = [T[:min_T] for T in all_Ts]
all_Xs = [X[:min_T, :] for X in all_Xs]

# Stack arrays
T_mat = np.stack(all_Ts, axis=0)              # [N, T]
X_mat = np.stack(all_Xs, axis=0)              # [N, T, D]
x0_mat = np.stack(all_X0s, axis=0)            # [N, D]

N, T, D = X_mat.shape
print(f"Loaded {N} total trajectories (after mixing), length T={T}, dim D={D}.")


# -----------------------------------------
# Define indices for train/val/test
# -----------------------------------------
idx_train = np.arange(len(Ts_train))
idx_val   = np.arange(len(Ts_train), len(Ts_train) + len(Ts_val))
idx_test  = np.arange(len(Ts_train) + len(Ts_val),
                      len(Ts_train) + len(Ts_val) + len(Ts_test))

print(f"idx_train: {idx_train}, idx_val: {idx_val}, idx_test: {idx_test}")

# -----------------------------------------
# 3) Normalize using TRAIN stats only
# We'll standardize X (over time & traj) and x0 (over traj).
# -----------------------------------------
X_train = X_mat[idx_train]            # [n_train, T, D]
x0_train = x0_mat[idx_train]          # [n_train, D]

X_mean = X_train.reshape(-1, D).mean(axis=0)
X_std  = X_train.reshape(-1, D).std(axis=0) + 1e-8

x0_mean = x0_train.mean(axis=0)
x0_std  = x0_train.std(axis=0) + 1e-8

def norm_X(X):
    return (X - X_mean[None, None, :]) / X_std[None, None, :]

def norm_x0(x0):
    return (x0 - x0_mean[None, :]) / x0_std[None, :]

Xn = norm_X(X_mat)
x0n = norm_x0(x0_mat)

#%%
# -----------------------------------------
# 4) Train with TRAIN set only (you can minibatch inside my_nn)
# -----------------------------------------
lambda_l1   = 0
optim_alg   = 'Adam'
epochs      = 5000
batch_size  = None   # all at once; set to e.g. 4 if you want mini-batches

model, train_loss = my_nn(
    x0n[idx_train],      # [n_train, D]
    Xn[idx_train],       # [n_train, T, D]
    T_mat[idx_train],    # [n_train, T]
    lambda_l1=lambda_l1,
    optim_alg=optim_alg,
    epochs=epochs,
    batch_size=batch_size
)

# -----------------------------------------
# 5) Evaluate on VAL and TEST
# -----------------------------------------
import torch.nn as nn
criterion = nn.MSELoss()

def evaluate_mse(model, x0n_split, Xn_split, T_split):
    # my_nn returns a PyTorch model; use its forward directly
    with torch.no_grad():
        t_t = torch.from_numpy(T_split).float().unsqueeze(-1)   # [N,T,1]
        x0_t = torch.from_numpy(x0n_split).float()              # [N,D]
        X_true_t = torch.from_numpy(Xn_split).float()           # [N,T,D]
        X_pred_t = model(t_t, x0_t)                             # [N,T,D]
        return float(criterion(X_pred_t, X_true_t).item())

val_mse  = evaluate_mse(model, x0n[idx_val],  Xn[idx_val],  T_mat[idx_val])
test_mse = evaluate_mse(model, x0n[idx_test], Xn[idx_test], T_mat[idx_test])

# -----------------------------------------
# 6) Plot training loss
# -----------------------------------------
plt.figure()
plt.plot(train_loss, label='Train loss (MSE + L1)')
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
plt.title('NN training loss')
plt.tight_layout()
plt.show()

# -----------------------------------------
# 7) Save model + normalizer + split
# -----------------------------------------
save_dir = Path(".")
torch.save(model.state_dict(), save_dir / "trained_model_ADAM5000.pt")
np.savez(save_dir / "nn_normalizer_and_split.npz",
         X_mean=X_mean, X_std=X_std, x0_mean=x0_mean, x0_std=x0_std,
         idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
         files=np.array([str(f) for f in (train_files + val_files + test_files)], dtype=object))
print("✅ Saved: trained_nn_model.pt and nn_normalizer_and_split.npz")



# -----------------------------------------
# 8) Print parameeters
# -----------------------------------------

print("\n==========================================")
print("✅ TRAINING SUMMARY")
print("==========================================")
print(f"Total trajectories:     {len(train_files) + len(val_files) + len(test_files)}")
print(f"  → Training:           {len(train_files)}")
print(f"  → Validation:         {len(val_files)}")
print(f"  → Test:               {len(test_files)}")
print("------------------------------------------")
print(f"Real ratio in training: {real_ratio_train*100:.0f}%")
print(f"Validation ratio:       {val_ratio*100:.0f}%")
print(f"Test ratio:             {test_ratio*100:.0f}%")
print("------------------------------------------")
print(f"Optimizer:              {optim_alg}")
print(f"L1 regularization:      {lambda_l1}")
print(f"Epochs:                 {epochs}")
print(f"Batch size:             {batch_size if batch_size else 'Full batch'}")
print("------------------------------------------")
print(f"Val MSE (normalized):   {val_mse:.3e}")
print(f"Test MSE (normalized):  {test_mse:.3e}")
print("==========================================\n")

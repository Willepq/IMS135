#%% data_split.py
import numpy as np
from pathlib import Path

# -----------------------------------------
# Paths
# -----------------------------------------
ROOT = Path(__file__).parent
ODE_FOLDER_1 = ROOT / "ODE_trajectories_1"
ODE_FOLDER_2 = ROOT / "ODE_trajectories_2"
REAL_FOLDER  = ROOT / "real_trajectories"
PATTERN = "*.npz"

# -----------------------------------------
# Load all file paths
# -----------------------------------------
ode_files = sorted(list(ODE_FOLDER_1.glob(PATTERN)) + list(ODE_FOLDER_2.glob(PATTERN)))
real_files = sorted(list(REAL_FOLDER.glob(PATTERN)))

# -----------------------------------------
# Manual or controlled split
# -----------------------------------------
# Example: fix which real ones you always keep for validation
fixed_val_files = [
    REAL_FOLDER / "trajectory_3.npz",
    REAL_FOLDER / "trajectory_4.npz",
]

# remove those from training pool
ode_train = ode_files.copy()
real_train = [f for f in real_files if f not in fixed_val_files]

# define ratios
real_ratio_train = 0.2   # 20% of training = real data
val_ratio = 0.2
test_ratio = 0.2

# random shuffle but deterministic
rng = np.random.default_rng(42)
rng.shuffle(ode_train)
rng.shuffle(real_train)

# split ODE/real into train/test (validation fixed)
n_total = len(ode_train) + len(real_train)
n_train = int((1 - val_ratio - test_ratio) * n_total)
n_real_train = int(real_ratio_train * n_train)
n_ode_train = n_train - n_real_train

train_files = ode_train[:n_ode_train] + real_train[:n_real_train]
remaining_files = ode_train[n_ode_train:] + real_train[n_real_train:]
rng.shuffle(remaining_files)

# half of remaining = test
n_test = len(remaining_files) // 2 + 2
test_files = remaining_files[:n_test]
val_files = fixed_val_files + remaining_files[n_test:]

print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# Optional: save split for reproducibility
np.savez(ROOT / "data_split_fixed.npz",
         train=np.array([str(f) for f in train_files], dtype=object),
         val=np.array([str(f) for f in val_files], dtype=object),
         test=np.array([str(f) for f in test_files], dtype=object))

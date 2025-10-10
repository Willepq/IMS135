# %% Initialize and train model

import numpy as np
import torch
from PINN_model import train_pinn, simulate  # import our new training & sim functions
import matplotlib.pyplot as plt
import glob
import os

# Folder with training .npz files
folder = '/Users/william_g/Documents/VScode/IMS135/Project/trajectories'

files = sorted(glob.glob(os.path.join(folder, 'trajectory_*.npz')))
print("Found files:", files)

x0_list = []
x_list = []
steps = None

for f in files:
    data = np.load(f)

    x0_data = data['x0']     # shape (4,)
    x_data  = data['X']      # shape (T, 4)
    steps_i = data['T']      # shape (T,)

    if steps is None:
        steps = steps_i

    x0_list.append(x0_data[np.newaxis, :])
    x_list.append(x_data[np.newaxis, :, :])

x0_all = np.concatenate(x0_list, axis=0)  # (N, 4)
x_all  = np.concatenate(x_list, axis=0)   # (N, T, 4)

print("x0_all shape:", x0_all.shape)
print("x_all shape:", x_all.shape)
print("steps shape:", steps.shape)

lambda_phys = 1e-6
optim_alg = 'Adam'
epochs = 20

# --- Train PINN ---
pinn_model = train_pinn(x0_all, x_all, steps, lambda_phys, optim_alg, epochs)


# %%
def validate_plot(x0, x_all, phi:int, steps, batch:int):
    dt = float(steps[1] - steps[0])
    T = len(steps)
    x0_tensor = x0[batch, :].reshape(1, 4)
    traj_pred = simulate(pinn_model, x0_tensor, T, dt).squeeze(1)
    traj_pred = traj_pred[1:]  # drop initial state so length = T

    print(traj_pred[:, phi].shape)
    print(steps.shape)

    plt.figure()
    plt.plot(steps, x_all[batch, :, phi], label='Real Data')
    plt.plot(steps, traj_pred[:, phi], '--r', label='Model Data')
    plt.title(f'Validation for Batch {batch}, variable {phi}')
    plt.legend()



# %% Validation example
validate_plot(x0_all, x_all, phi=0, steps=steps, batch=0)


# %% Load and test on unseen validation data
val_path = '/Users/william_g/Documents/VScode/IMS135/Project/Data/validation trajectories/validation_trajectories_0.npz'
val_data = np.load(val_path)
x_valid = val_data['X']
steps_valid = val_data['T']

# new test initial condition
test = np.array([np.pi/2.2, np.pi/6, 0.0, 0.0]).reshape(1, 4)
dt = float(steps_valid[1] - steps_valid[0])
T = len(steps_valid)
traj_test = simulate(pinn_model, test, T, dt).squeeze(1)

plt.figure()
plt.plot(steps_valid, traj_test[:, 0], label='Model')
plt.plot(steps_valid, x_valid[:, 0], label='Real')
plt.title('Model test on untrained data')
plt.legend()
plt.show()

# %%

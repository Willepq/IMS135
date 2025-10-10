
# %% Initialize and prepare the data ===========================================

import numpy as np
import torch
from PINN_model import my_nn  # import the training function
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Folder where your .npz files are stored
folder = '/Users/william_g/Documents/VScode/IMS135/Project/trajectories'

# Find all trajectory files
files = sorted(glob.glob(os.path.join(folder, 'trajectory_*.npz')))
print("Found files:", files)

x0_list = []
x_list = []
steps = None

import numpy as np

x0_list, x_list = [], []
steps = None

for f in files:
    data = np.load(f)

    x0_data = data['x0']     # shape (4,)
    x_data  = data['X']      # shape (T, 4)
    steps_i = data['T']      # shape (T,)

    # Store steps from the first file only (should be same for all)
    if steps is None:
        steps = steps_i

    # Add batch dims for stacking later
    x0_list.append(x0_data[np.newaxis, :])   # (1, 4)
    x_list.append(x_data[np.newaxis, :, :])  # (1, T, 4)

# Stack along batch dimension -> (N, 4) and (N, T, 4)
x0_all = np.concatenate(x0_list, axis=0)
x_all  = np.concatenate(x_list, axis=0)

# ---- Transform (N,*,4) -> (N,*,6) with [sin t1, sin t2, cos t1, cos t2, dth1, dth2]
def to_sin_cos6(x):
    # x shape: (..., 4) = [theta1, theta2, dtheta1/dt, dtheta2/dt]
    t1 = x[..., 0]
    t2 = x[..., 1]
    d1 = x[..., 2]
    d2 = x[..., 3]
    out = np.stack([np.sin(t1), np.sin(t2),
                    np.cos(t1), np.cos(t2),
                    d1, d2], axis=-1)
    return out

# Apply to initial states and sequences
x0_all_6 = to_sin_cos6(x0_all)   # shape (N, 6)
x_all_6  = to_sin_cos6(x_all)    # shape (N, T, 6)

print("x0_all_6 shape:", x0_all_6.shape)   # (N, 6)
print("x_all_6 shape:", x_all_6.shape)     # (N, T, 6)
print("steps shape:", steps.shape)

#%% Train the model ===========================================

lambda_l1 = 1e-6
optim_alg = 'Adam'

mymodel,loss_history = my_nn(x0_all_6, x_all_6, steps, lambda_l1, optim_alg, epochs=2000)


#%%
def validate_plot(x0, x, phi:int, steps, batch:int):
    x0_tensor = torch.from_numpy(x0[phi,:]).float().unsqueeze(0)
    y_pred = mymodel(x0_tensor).squeeze(0).detach().numpy()  # shape becomes (5000, 4)
    
    plt.figure()
    plt.plot(steps,x[batch,:,phi], label='Real Data')
    plt.plot(steps,y_pred[:,phi], '--r', label='Model Data')
    plt.title(fr'Validation for Batch {batch}, $\theta_{+1}$')
    plt.legend()


# %% Plot results ===========================================

validate_plot(x0_all_6,x_all_6, 0, steps, 0)

# load and define validation data for testing
val_data = np.load('/Users/william_g/Documents/VScode/IMS135/Project/Data/validation trajectories/validation_trajectories_0.npz')
x_valid      = val_data['X']


test = np.array([np.sin(np.pi/2.2), np.sin(np.pi/6), np.cos(np.pi/2.2), np.cos(np.pi/6), 0.0, 0.0]).reshape(1,6)
test_tensor = torch.from_numpy(test).float().unsqueeze(0)
y_test = mymodel(test_tensor).squeeze(0).detach().numpy()

plt.figure()
plt.plot(steps, y_test[:,0])
plt.plot(steps,x_valid[:,0])
plt.title('Model test, untrained data')



# %%

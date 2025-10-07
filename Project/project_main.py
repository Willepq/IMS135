
# %% Initialize and train model

import numpy as np
import torch
from model import my_nn  # import the training function
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

for f in files:
    data = np.load(f)
    
    x0_data = data['x0']     # shape (4,)
    x_data  = data['X']      # shape (T, 4)
    steps_i = data['T']      # shape (T,)
    
    # Store steps from the first file only (should be same for all)
    if steps is None:
        steps = steps_i
    
    # Add batch dimensions for stacking later
    x0_list.append(x0_data[np.newaxis, :])   # shape (1, 4)
    x_list.append(x_data[np.newaxis, :, :])  # shape (1, T, 4)

# Stack along batch dimension
x0_all = np.concatenate(x0_list, axis=0)  # (N, 4)
x_all  = np.concatenate(x_list, axis=0)   # (N, T, 4)

print("x0_all shape:", x0_all.shape)
print("x_all shape:", x_all.shape)
print("steps shape:", steps.shape)

lambda_l1 = 1e-9
optim_alg = 'Adam'

mymodel = my_nn(x0_all, x_all, steps, lambda_l1, optim_alg, epochs=2000)


# %% 

test = np.array([np.pi/2.2, np.pi/6, 0.0,0.0]).reshape(1,4)
test_tensor = torch.from_numpy(test).float().unsqueeze(0)
y_test = mymodel(test_tensor).squeeze(0).detach().numpy()

plt.plot(steps, y_test[:,0])



def validate_plot(x0, x, phi:int, steps, batch:int):
    x0_tensor = torch.from_numpy(x0[phi,:]).float().unsqueeze(0)
    y_pred = mymodel(x0_tensor).squeeze(0).detach().numpy()  # shape becomes (5000, 4)
    
    plt.figure()
    plt.plot(steps,x[batch,:,phi], label='Real Data')
    plt.plot(steps,y_pred[:,phi], '--r', label='Model Data')
    plt.title(fr'Validation for Batch {batch}, $\theta_{+1}$')
    plt.legend()


# %%

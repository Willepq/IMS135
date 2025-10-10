#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file trains and runs the model. 
"""
#%% 
import numpy as np
from model_jakob import my_nn  # import the training function
import torch
import matplotlib.pyplot as plt
import glob
import os
import time

#%% Fetch the training data

# Folder where your .npz files are stored
folder = '/Users/william_g/Documents/VScode/IMS135/Project/trajectories'

# Find all trajectory files
files = sorted(glob.glob(os.path.join(folder, 'trajectory_*.npz')))
print(f"Found {len(files)} trajectory files")

# Number of trajectories to include in training
n_train = min(8, len(files))
print(f"Training on {n_train} trajectory files")

x0_list, x_list, t_list = [], [], []

for f in files[:n_train]:
    data = np.load(f)
    x0_list.append(data['x0'][np.newaxis, :])    # (1, 4)
    x_list.append(data['X'][np.newaxis, :, :])   # (1, T, 4)
    t_list.append(data['T'][np.newaxis, :])      # (1, T)

# Stack along batch dimension
x0_train = np.concatenate(x0_list, axis=0)  # (N, 4)
x_train  = np.concatenate(x_list, axis=0)   # (N, T, 4)
t_train  = np.concatenate(t_list, axis=0)   # (N, T)


"""Test using sin and cos angles instead of raw angles"""
# Encode angles as sine and cosine
# x0 = [theta1, theta2, theta1dot, theta2dot]
theta1_0, theta2_0, theta1dot_0, theta2dot_0 = x0_train.T

x0_train_encoded = np.stack([
    np.sin(theta1_0), np.cos(theta1_0),
    np.sin(theta2_0), np.cos(theta2_0),
    theta1dot_0, theta2dot_0
], axis=1)

# For full trajectories (N, T, 4)
theta1 = x_train[:, :, 0]
theta2 = x_train[:, :, 1]
theta1dot = x_train[:, :, 2]
theta2dot = x_train[:, :, 3]

x_train_encoded = np.stack([
    np.sin(theta1), np.cos(theta1),
    np.sin(theta2), np.cos(theta2),
    theta1dot, theta2dot
], axis=-1)  # Shape: (N, T, 6)

#%% Train the network

# Start timer
start_time = time.time()

lambda_l1 = 0.0
lambda_phys = 0.1
optim_alg = 'Adam'
epochs = 1000

# Choose NN or PINN model 
mymodel, train_loss = my_nn(x0_train_encoded, x_train_encoded, t_train, lambda_l1, optim_alg, epochs)
# mymodel, train_loss, train_loss_data, train_loss_phys = my_nn_PINN(x0_train_encoded, x_train_encoded, t_train, lambda_l1, lambda_phys, optim_alg, epochs)

# End timer
end_time = time.time()

# Compute and print total training time
elapsed_time = end_time - start_time
print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# plot the training results vs actual trajectory
def training_plot(model, x0, x, t, variable, trajectory_nr):
    """
    model: trained TrajectoryNet
    x0: (N, 4)
    x:  (N, T, 4)
    t:  (N, T)
    variable: 0..3 (which state to plot)
    trajectory_nr: which trajectory to visualize
    """
    model.eval()

    # Select the trajectory and add the batch/time-channel dims
    t_tensor  = torch.from_numpy(t[trajectory_nr, :]).float().unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    x0_tensor = torch.from_numpy(x0[trajectory_nr, :]).float().unsqueeze(0)               # [1, 4]

    with torch.no_grad():
        y_pred = model(t_tensor, x0_tensor).cpu().numpy()[0]  # [T, 4]

    # Plot
    plt.figure()
    plt.plot(t[trajectory_nr, :], x[trajectory_nr, :, variable], label='Real Data')
    plt.plot(t[trajectory_nr, :], y_pred[:, variable], 'r--', label='Model Prediction')
    labels = [r'$\theta_1$', r'$\theta_2$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
    plt.title(f'Validation for training trajectory {trajectory_nr}, variable: {labels[variable]}', fontsize=17)
    plt.xlabel('t', fontsize=15)
    plt.ylabel(labels[variable], fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()
    
    ## Comment out the 2nd plot definition if running with PINN and the 1st if regular NN
    
    # plt.figure()
    # plt.semilogy(train_loss, label = "total loss") 
    # plt.semilogy(train_loss_data, label = "Loss from trajectory")
    # plt.semilogy(train_loss_phys, label ="Loss from ODE")
    # plt.xlabel('Epoch', fontsize=15)
    # plt.ylabel('Average Training Loss', fontsize=14)
    # plt.title('Training Loss Curve', fontsize=17)
    # plt.legend(fontsize=15)
    # plt.grid(True)
    # plt.show()
    
    plt.figure()
    plt.semilogy(train_loss, label = "total loss") 
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Average Training Loss', fontsize=14)
    plt.title('Training Loss Curve', fontsize=17)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()
    
    return y_pred

y_predict = training_plot(mymodel, x0_train_encoded, x_train_encoded, t_train, variable=0, trajectory_nr=0)
    

#%% Test the model on a real, unseen trajectory

# load and define validation data for testing
val_data = np.load('/Users/william_g/Documents/VScode/IMS135/Project/Data/validation trajectories')
x_valid  = val_data['X']
x0_valid = val_data['x0']
# x0_valid = np.array(x0_valid).reshape(1,4)                       # not needed
x0_valid_tensor = torch.from_numpy(x0_valid).float().unsqueeze(0)

y_test = mymodel(x0_valid_tensor).squeeze(0).detach().numpy()

plt.figure()
plt.plot(t_train, y_test[:,0])
plt.plot(t_train,x_valid[:,0])
plt.title('Model test, untrained data')


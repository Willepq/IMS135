import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ------------------------------
# Neural net that predicts accelerations
# ------------------------------
class AccelNet(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=2):
        super(AccelNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        return self.net(x)  # outputs accelerations [ddθ1, ddθ2]


# ------------------------------
# Training function for PINN
# ------------------------------
def train_pinn(x0_all, x_all, steps, lambda_phys, optim_alg='Adam', epochs=2000):
    """
    Trains the PINN on multiple trajectories.

    x0_all : (N, 4)
    x_all  : (N, T, 4)  -> [θ1, θ2, dθ1, dθ2]
    steps  : (T,)       -> time array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = float(steps[1] - steps[0])
    timesteps = len(steps)

    # Prepare training data
    x0_tensor = torch.tensor(x0_all, dtype=torch.float32).to(device)
    x_tensor = torch.tensor(x_all, dtype=torch.float32).to(device)

    pos_true = x_tensor[:, :, :2]
    vel_true = x_tensor[:, :, 2:]

    model = AccelNet().to(device)
    mse = nn.MSELoss()

    if optim_alg == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=0.01)
    elif optim_alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optim_alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optim_alg}")

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        # Loop over all trajectories
        for i in range(x0_tensor.shape[0]):
            state = x0_tensor[i:i+1, :]  # shape (1, 4)
            pos_pred = []
            vel_pred = []

            for t in range(timesteps):
                # NEW — safe for autograd
                acc = model(state)
                new_vel = state[:, 2:] + acc * dt
                new_pos = state[:, :2] + new_vel * dt
                state = torch.cat([new_pos, new_vel], dim=1)
                pos_pred.append(state[:, :2])
                vel_pred.append(state[:, 2:])

            pos_pred = torch.cat(pos_pred, dim=0).view(timesteps, -1)
            vel_pred = torch.cat(vel_pred, dim=0).view(timesteps, -1)

            loss_data = mse(pos_pred, pos_true[i]) + mse(vel_pred, vel_true[i])
            phys_reg = torch.mean(model(state) ** 2)
            loss_total += loss_data + lambda_phys * phys_reg

        loss_total.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_total.item():.6f}")

    return model


# ------------------------------
# Simulation using trained model
# ------------------------------
def simulate(model, x0, timesteps, dt):
    model.eval()
    x0 = torch.tensor(x0, dtype=torch.float32)
    state = x0.clone()
    traj = [state.detach().cpu().numpy()]
    for _ in range(timesteps):
        acc = model(state)
        state[:, 2:] += acc * dt
        state[:, :2] += state[:, 2:] * dt
        traj.append(state.detach().cpu().numpy())
    return np.stack(traj)

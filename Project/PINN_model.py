import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------------
# Neural network definition
# ----------------------------------
class TrajectoryNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, timesteps):
        super(TrajectoryNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_outputs * timesteps)
        self.tanh = nn.Tanh()
        self.num_outputs = num_outputs
        self.timesteps = timesteps

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.timesteps, self.num_outputs)  # (B, T, S)


# ----------------------------------
# Generalized training function (now logs + plots loss)
# ----------------------------------
def my_nn(x0, x, steps, lambda_l1, optim_alg, epochs):
    """
    Trains a neural network to predict state trajectories x(t) from initial conditions x0.
    Returns the trained model and the loss history.
    """

    # Convert numpy arrays to PyTorch tensors
    x0_tensor = torch.from_numpy(x0).float()   # (N, num_inputs)
    x_tensor  = torch.from_numpy(x).float()    # (N, T, num_outputs)

    # Sizes
    num_inputs  = x0_tensor.shape[1]
    timesteps   = x_tensor.shape[1]            # safer than len(steps) if they ever differ
    num_outputs = x_tensor.shape[2]

    # Model, loss, optimizer
    torch.manual_seed(0)
    model = TrajectoryNet(num_inputs, num_outputs, timesteps)
    criterion = nn.MSELoss()

    if optim_alg == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=20, history_size=100)
    elif optim_alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif optim_alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optim_alg}")

    # L1 helper
    def add_l1(loss):
        if lambda_l1 > 0.0:
            l1_reg = torch.tensor(0., device=loss.device)
            for p in model.parameters():
                l1_reg += p.abs().sum()
            loss = loss + lambda_l1 * l1_reg
        return loss

    loss_history = []

    # --- Training loop ---
    model.train()
    if optim_alg == 'LBFGS':
        # LBFGS needs a closure; we still run for 'epochs' outer loops
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                outputs = model(x0_tensor)
                loss = criterion(outputs, x_tensor)
                loss = add_l1(loss)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Log current loss value
            with torch.no_grad():
                outputs = model(x0_tensor)
                loss_val = criterion(outputs, x_tensor)
                loss_val = add_l1(loss_val).item()
            loss_history.append(loss_val)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss_val:.6e}")
    else:
        # Standard loop for Adam/SGD
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x0_tensor)
            loss = criterion(outputs, x_tensor)
            loss = add_l1(loss)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss_history[-1]:.6e}")

    # --- Plot loss after training ---
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE + L1)')
    plt.title(f'Training Loss – {optim_alg}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, loss_history

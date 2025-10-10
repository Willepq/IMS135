import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch


# ----------------------------------
# Neural network definition
# ----------------------------------
class TrajectoryNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, timesteps):
        super(TrajectoryNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_outputs * timesteps)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.num_outputs = num_outputs
        self.timesteps = timesteps

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # reshape into (batch_size, timesteps, num_outputs)
        return x.view(-1, self.timesteps, self.num_outputs)


# ----------------------------------
# Generalized training function
# ----------------------------------
def my_nn(x0, x, steps, lambda_l1, optim_alg, epochs):
    """
    Trains a neural network to predict state trajectories x(t)
    from initial conditions x0.

    Parameters
    ----------
    x0 : np.ndarray
        Shape (N, num_inputs) – starting values for each simulation
    x : np.ndarray
        Shape (N, timesteps, num_outputs) – target trajectories
    steps : np.ndarray
        Shape (timesteps, 1) or (timesteps,) – time or step values
    lambda_l1 : float
        L1 regularization strength
    optim_alg : str
        Optimizer type ('Adam', 'LBFGS', or 'SGD')
    epochs : int
        Number of training iterations
    """

    # Convert numpy arrays to PyTorch tensors
    x0_tensor = torch.from_numpy(x0).float()  # shape (1, 4)
    x_tensor = torch.from_numpy(x).float()    # shape (1, T, 4)


    # Automatically determine sizes
    num_inputs = x0_tensor.shape[1]
    timesteps = len(steps)
    num_outputs = x_tensor.shape[2]
    
    # Initialize model
    torch.manual_seed(0)
    model = TrajectoryNet(num_inputs, num_outputs, timesteps)

    # Define loss and optimizer
    criterion = nn.MSELoss()

    if optim_alg == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=0.01)
    elif optim_alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optim_alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optim_alg}")

    # Closure for LBFGS
    def closure():
        optimizer.zero_grad()
        outputs = model(x0_tensor)
        loss = criterion(outputs, x_tensor)
        l1_reg = torch.tensor(0.)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        loss += lambda_l1 * l1_reg
        loss.backward()
        return loss

    # Training loop
    for epoch in range(epochs):
        optimizer.step(closure)
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                loss_val = criterion(model(x0_tensor), x_tensor).item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_val:.6f}")

    return model



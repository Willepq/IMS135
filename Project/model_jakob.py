"""
This code sets up the architecture and defines the NN.
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

# ----------------------------------
# Neural network definition
# ----------------------------------
class TrajectoryNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, timesteps):
        super(TrajectoryNet, self).__init__()
        # 2 hidden layers and 1 output layer. 
        self.fc1 = nn.Linear(num_inputs+1, 128)  # add 1 due to time being an input variable now as well
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, num_outputs)
        
        # defined attributes that are used in the forward() method
        self.num_outputs = num_outputs
        # Activation functions
        self.relu = nn.ReLU() # This is initialized but not used 
        self.tanh = nn.Tanh()
        
    
    def forward(self, t, x0):
        """
        t: tensor of shape [N, T, 1] — time values for each trajectory
        x0: tensor of shape [N, num_inputs]
        Returns: tensor of shape [N, T, num_outputs]
        """
        N, T, _ = t.shape
        x0_expanded = x0.unsqueeze(1).repeat(1, T, 1)   # [N, T, num_inputs]
        inputs = torch.cat([t, x0_expanded], dim=2)     # [N, T, num_inputs + 1]
    
        # Pass each timepoint of each trajectory through the same MLP
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return self.fc5(x)                                      # [N, T, num_outputs]



# ----------------------------------
# Generalized training function
# ----------------------------------
def my_nn(x0, x, t, lambda_l1, optim_alg, epochs, batch_size=None):
    """
    Vectorized training for the TrajectoryNet model.
    
    Parameters
    ----------
    x0 : np.ndarray
        Shape (N, num_inputs) – initial conditions for N trajectories.
    x : np.ndarray
        Shape (N, T, num_outputs) – true state trajectories.
    t : np.ndarray
        Shape (N, T) – time values for each trajectory.
    lambda_l1 : float
        L1 regularization strength.
    optim_alg : str
        Optimizer type ('Adam', 'LBFGS', 'SGD').
    epochs : int
        Number of training iterations.
    batch_size : int or None
        Optional minibatch size. If None, all trajectories are used at once.
    """

    # Convert numpy arrays to torch tensors
    t_tensor = torch.from_numpy(t).float().unsqueeze(-1)  # [N, T, 1]
    x0_tensor = torch.from_numpy(x0).float()              # [N, num_inputs]
    x_tensor  = torch.from_numpy(x).float()               # [N, T, num_outputs]

    # Infer dimensions
    num_inputs = x0.shape[1]
    num_outputs = x.shape[2]
    N = x0.shape[0]

    # Initialize model
    torch.manual_seed(0)
    model = TrajectoryNet(num_inputs, num_outputs, timesteps=None)

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

    # Prepare batching
    if batch_size is None or batch_size > N:
        batch_size = N

    num_batches = int(np.ceil(N / batch_size))
    train_loss = []

    # Training loop
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0

        for b in range(num_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            t_b = t_tensor[idx]    # [B, T, 1]
            x0_b = x0_tensor[idx]  # [B, num_inputs]
            x_b = x_tensor[idx]    # [B, T, num_outputs]

            def closure():
                optimizer.zero_grad()
                outputs = model(t_b, x0_b)      # [B, T, num_outputs]
                loss = criterion(outputs, x_b)
                l1_reg = sum(torch.norm(p, 1) for p in model.parameters())
                loss += lambda_l1 * l1_reg
                loss.backward()
                return loss

            # Step optimizer
            if optim_alg == 'LBFGS':
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        train_loss.append(avg_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

    return model, train_loss





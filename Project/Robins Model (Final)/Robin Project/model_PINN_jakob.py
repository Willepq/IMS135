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
        self.fc4 = nn.Linear(128, num_outputs)
        
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
        return self.fc4(x)                                      # [N, T, num_outputs]


# ---------------------------------------------------------------
# Physics: double pendulum accelerations (from your Jupyter notebook)
# ---------------------------------------------------------------
def double_pendulum_acc(th1, th2, th1dot, th2dot, params):
    m1, m2 = params['m1'], params['m2']
    l1, r1, r2 = params['l1'], params['r1'], params['r2']
    I1, I2 = params['I1'], params['I2']
    b, mu, g = params['b'], params['mu'], params['g']

    s1, c1 = torch.sin(th1), torch.cos(th1)
    s2, c2 = torch.sin(th2), torch.cos(th2)
    s12 = torch.sin(th1 + th2)

    # --- Mass matrix ---
    M11 = I1 + I2 + l1**2 * m2 + 2*l1*m2*r2*c2
    M12 = I2 + l1*m2*r2*c2
    M21 = M12
    M22 = I2

    # --- Coriolis matrix ---
    C11 = -2*th2dot*l1*m2*r2*s2
    C12 = -th2dot*l1*m2*r2*s2
    C21 = th1dot*l1*m2*r2*s2
    C22 = torch.zeros_like(th1)

    # --- Gravity vector ---
    G1 = -g*m1*r1*s1 - g*m2*(l1*s1 + r2*s12)
    G2 = -g*m2*r2*s12

    # --- Friction vector ---
    F1 = b*th1dot + mu*torch.atan(100*th1dot)
    F2 = b*th2dot + mu*torch.atan(100*th2dot)

    # Inverse of 2×2 matrix (done analytically)
    detM = M11*M22 - M12*M21
    Minv11 =  M22 / detM
    Minv12 = -M12 / detM
    Minv21 = -M21 / detM
    Minv22 =  M11 / detM

    rhs1 = G1 - (C11*th1dot + C12*th2dot) - F1
    rhs2 = G2 - (C21*th1dot + C22*th2dot) - F2

    th1ddot = Minv11*rhs1 + Minv12*rhs2
    th2ddot = Minv21*rhs1 + Minv22*rhs2

    return th1ddot, th2ddot



# ----------------------------------
# Generalized training function
# ----------------------------------
def my_nn_PINN(x0, x, t, lambda_l1, lambda_phys, optim_alg, epochs, batch_size=None):
    """
    Training function for the PINN version of the double pendulum network.
    
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

    # Convert numpy arrays to tensors
    t_tensor  = torch.from_numpy(t).float().unsqueeze(-1)  # [N, T, 1]
    x0_tensor = torch.from_numpy(x0).float()               # [N, num_inputs]
    x_tensor  = torch.from_numpy(x).float()                # [N, T, num_outputs]

    num_inputs  = x0.shape[1]
    num_outputs = x.shape[2]
    N = x0.shape[0]

    # Initialize model
    torch.manual_seed(0)
    model = TrajectoryNet(num_inputs, num_outputs, timesteps=None)

    # Define optimizer
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
    # Add at start of training loop
    train_loss, train_loss_data, train_loss_phys = [], [], []

    # Physical parameters (constants)
    params = {
        'm1': 0.10548, 'm2': 0.07619,
        'l1': 0.05, 'r1': 0.05, 'r2': 0.0367,
        'I1': 0.0004616, 'I2': 0.0002370,
        'b': 0.00051, 'mu': 0.00305, 'g': 9.81
    }

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
    
        # For batch logging
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
    
        for b in range(num_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            t_b  = t_tensor[idx].clone().requires_grad_(True)
            x0_b = x0_tensor[idx]
            x_b  = x_tensor[idx]
    
            def closure():
                optimizer.zero_grad()
                outputs = model(t_b, x0_b)
    
                l1_reg = sum(torch.norm(p, 1) for p in model.parameters())
                data_loss = criterion(outputs, x_b)
    
                # --- Physics loss computation ---
                sin_th1, cos_th1 = outputs[..., 0], outputs[..., 1]
                sin_th2, cos_th2 = outputs[..., 2], outputs[..., 3]
                th1dot, th2dot   = outputs[..., 4], outputs[..., 5]
                th1 = torch.atan2(sin_th1, cos_th1)
                th2 = torch.atan2(sin_th2, cos_th2)
    
                th1dotdot = torch.autograd.grad(
                    th1dot, t_b, grad_outputs=torch.ones_like(th1dot),
                    create_graph=True, retain_graph=True
                )[0].squeeze(-1)
                th2dotdot = torch.autograd.grad(
                    th2dot, t_b, grad_outputs=torch.ones_like(th2dot),
                    create_graph=True, retain_graph=True
                )[0].squeeze(-1)
    
                ## This is where the acceleration, and thus, the ODE function is used
                th1ddot_phys, th2ddot_phys = double_pendulum_acc(
                    th1, th2, th1dot, th2dot, params
                )
                physics_res = (th1dotdot - th1ddot_phys)**2 + (th2dotdot - th2ddot_phys)**2
                physics_loss = physics_res.mean()
    
                loss = data_loss + lambda_phys * physics_loss + lambda_l1 * l1_reg
    
                # --- Track losses for monitoring ---
                nonlocal epoch_data_loss, epoch_phys_loss
                epoch_data_loss += data_loss.detach().item()
                epoch_phys_loss += physics_loss.detach().item()
    
                loss.backward()
                return loss
    
            if optim_alg == 'LBFGS':
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
    
            total_loss += loss.item()
    
        avg_loss = total_loss / num_batches
        train_loss.append(avg_loss)
        train_loss_data.append(epoch_data_loss / num_batches)
        train_loss_phys.append(epoch_phys_loss / num_batches)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Total: {avg_loss:.6f} | "
                  f"Data: {train_loss_data[-1]:.6f} | "
                  f"Physics: {train_loss_phys[-1]:.6f}")

    return model, train_loss, train_loss_data, train_loss_phys





#%% plot_trained_vs_real.py
import numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from model_jakob import TrajectoryNet
from data_split import val_files

ROOT = Path(__file__).parent
meta = np.load(ROOT / "nn_normalizer_and_split.npz", allow_pickle=True)
X_mean, X_std = meta["X_mean"], meta["X_std"]
x0_mean, x0_std = meta["x0_mean"], meta["x0_std"]
D = len(X_mean)

model = TrajectoryNet(D, D, timesteps=None)
model.load_state_dict(torch.load(ROOT / "trained_model_ADAMS000.pt", map_location="cpu"))
model.eval()


for f in val_files:
    data = np.load(f, allow_pickle=True)
    T, X_true, x0 = data["T"], data["X"][:, :D], data["x0"][:D]

    def norm_X(X):  return (X - X_mean[None, :]) / (X_std[None, :] + 1e-8)
    def norm_x0(x0): return (x0 - x0_mean) / (x0_std + 1e-8)

    t = torch.tensor(T)[None, :, None].float()
    x0_t = torch.tensor(norm_x0(x0))[None, :].float()

    with torch.no_grad():
        X_pred_n = model(t, x0_t).squeeze(0).numpy()
    X_pred = X_pred_n * X_std + X_mean

    # --- Create figure with subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    labels = [r"$\theta_1$", r"$\theta_2$", r"$\omega_1$", r"$\omega_2$"]

    for i in range(D):
        ax = axes[i]
        ax.plot(T, X_true[:, i], label="True", linewidth=1.2)
        ax.plot(T, X_pred[:, i], "--", label="Pred", linewidth=1.2)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(labels[i])
        ax.set_title(f"{labels[i]} vs time", fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Main title for this figure
    fig.suptitle(f"Validation trajectory [{Path(f).parent.name}]: {Path(f).name}", fontsize=13, y=0.995)
    fig.tight_layout()
    plt.show()
# %%

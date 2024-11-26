import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def spectral_norm_conv(conv_layer, device="cpu"):
    """
    Compute the spectral norm for Conv2d layers.
    """
    weight = conv_layer.weight.data.to(device)
    reshaped_weight = weight.view(weight.size(0), -1)  # Reshape to 2D
    spectral_norm = torch.linalg.norm(reshaped_weight, ord=2)
    return spectral_norm.item()


def lipschitz_constant_linear_layers(weights, c=1.0, device="cpu"):
    """
    Compute the Lipschitz constant for linear layers using eigenvalues.
    """
    m = 0.5  # ReLU
    weights = [W.to(device) for W in weights]
    n0 = weights[0].shape[1]
    X_prev = c * torch.eye(n0, device=device)
    for i in range(len(weights) - 1):
        W_i = weights[i]
        X_prev_inv = torch.linalg.pinv(X_prev)
        A_i = W_i @ X_prev_inv @ W_i.T
        eig_max = torch.linalg.eigvalsh(A_i).max().item()
        lambda_i = 1 / (2 * m**2 * eig_max)
        n_i = W_i.shape[0]
        X_i = lambda_i * torch.eye(n_i, device=device) - lambda_i**2 * m**2 * A_i
        X_prev = X_i
    eig_min = torch.linalg.eigvalsh(X_prev).min().item()
    L_bar = (1 / eig_min) ** 0.5
    W_l = weights[-1]
    norm_W_l = torch.linalg.norm(W_l, 2).item()
    L0 = (c**0.5) * L_bar * norm_W_l
    return L0


# Combined Lipschitz constant computation
def compute_combined_lipschitz_constant(model, device="cpu"):
    """
    Combines the spectral norm for Conv2d layers and eigenvalue-based method for Linear layers.
    """
    L = 1.0
    linear_weight_list = []

    for layer in model.network:
        if isinstance(layer, nn.Conv2d):
            spectral_norm = spectral_norm_conv(layer, device=device)
            print(f"Spectral Norm (Conv2d): {spectral_norm}")
            L *= spectral_norm
        elif isinstance(layer, nn.Linear):
            linear_weight_list.append(layer.weight.data)

    # Compute the Lipschitz constant for the linear layers
    if linear_weight_list:
        linear_lipschitz = lipschitz_constant_linear_layers(
            linear_weight_list, device=device
        )
        print(f"Lipschitz Constant (Linear Layers): {linear_lipschitz}")
        L *= linear_lipschitz

    return L


def compute_simple_spectral_norm_lipschitz(model, device="cpu"):
    """
    Compute the Lipschitz constant using only the spectral norm for Conv2d and Linear layers.
    """
    L = 1.0

    for layer in model.network:
        if isinstance(layer, nn.Conv2d):
            # Compute spectral norm for Conv2d
            weight = layer.weight.data.to(device)
            reshaped_weight = weight.view(weight.size(0), -1)  # Reshape to 2D
            spectral_norm = torch.linalg.norm(reshaped_weight, ord=2)
            print(f"Spectral Norm (Conv2d): {spectral_norm.item()}")
            L *= spectral_norm.item()
        elif isinstance(layer, nn.Linear):
            # Compute spectral norm for Linear
            weight = layer.weight.data.to(device)
            spectral_norm = torch.linalg.norm(weight, ord=2)
            print(f"Spectral Norm (Linear): {spectral_norm.item()}")
            L *= spectral_norm.item()

    return L


if __name__ == "__main__":
    env = gym.make("PongNoFrameskip-v4")
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStack(env, 4)

    model = QNetwork(env)

    # Compute the Lipschitz constant using combined method
    device = "cpu"
    combined_lipschitz_constant = compute_combined_lipschitz_constant(
        model, device=device
    )
    print(f"Combined Lipschitz Constant: {combined_lipschitz_constant}")

    # Compute the Lipschitz constant using simple spectral norm method
    simple_lipschitz_constant = compute_simple_spectral_norm_lipschitz(
        model, device=device
    )
    print(f"Simple Spectral Norm Lipschitz Constant: {simple_lipschitz_constant}")

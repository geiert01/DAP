import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


# Define the QNetwork class for Atari environment
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Similar to DQN architecture
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # Adjusted for 84x84 input size
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)  # Normalize inputs to [0, 1]


# Method 1: Compute Lipschitz constant using spectral norms
def compute_lipschitz_constant(model, input_shape):
    lipschitz_constant = 1.0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # Compute spectral norm for Conv2d (mocked for simplicity here)
            weight = layer.weight.data.cpu()
            spectral_norm = torch.linalg.norm(weight.view(weight.size(0), -1), ord=2)
            lipschitz_constant *= spectral_norm.item()
            print(f"Spectral Norm (Conv2d): {spectral_norm.item()}")
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.data.cpu()
            spectral_norm = torch.linalg.norm(weight, ord=2)
            lipschitz_constant *= spectral_norm.item()
            print(f"Spectral Norm (Linear): {spectral_norm.item()}")
    return lipschitz_constant


# Method 2: Compute Lipschitz constant using lipschitz_constant_linear_layers
def lipschitz_constant_linear_layers(weights, c=1.0, device="cpu"):

    m = 0.5  # ReLU

    weights = [W.to(device) for W in weights]

    n0 = weights[0].shape[1]
    X_prev = c * torch.eye(n0, device=device)

    lambda_list = []
    for i in range(len(weights) - 1):
        W_i = weights[i]
        try:
            X_prev_inv = torch.linalg.inv(X_prev)
        except RuntimeError:
            X_prev_inv = torch.linalg.pinv(X_prev)

        A_i = W_i @ X_prev_inv @ W_i.T
        eigvals = torch.linalg.eigvalsh(A_i)
        eig_max = eigvals.max().item()

        lambda_i = 1 / (2 * m**2 * eig_max)
        lambda_list.append(lambda_i)

        n_i = W_i.shape[0]
        X_i = lambda_i * torch.eye(n_i, device=device) - lambda_i**2 * m**2 * A_i
        X_prev = X_i

    eigvals = torch.linalg.eigvalsh(X_prev)
    eig_min = eigvals.min().item()
    L_bar = (1 / eig_min) ** 0.5

    W_l = weights[-1]
    norm_W_l = torch.linalg.norm(W_l, 2).item()
    print(f"Spectral Norm (Final Linear): {norm_W_l}")

    L0 = (c**0.5) * L_bar * norm_W_l
    return L0


# Test the methods
if __name__ == "__main__":
    # Create a Gymnasium Atari environment
    env = gym.make("ALE/Pong-v5")  # Example Atari environment
    env = gym.wrappers.GrayScaleObservation(env)  # Grayscale
    env = gym.wrappers.ResizeObservation(env, (84, 84))  # Resize to 84x84
    env = gym.wrappers.FrameStack(env, 4)  # Stack 4 frames

    # Instantiate the QNetwork
    model = QNetwork(env)

    # Compute Lipschitz constant using Method 1
    input_shape = (1, 4, 84, 84)
    spectral_norm_lipschitz = compute_lipschitz_constant(model, input_shape)

    # Prepare weights for Method 2
    weights = [
        layer.weight.data for layer in model.network if isinstance(layer, nn.Linear)
    ]
    for weight in weights:
        print(weight.shape)

    # Compute Lipschitz constant using Method 2
    lipschitz_constant_l0 = lipschitz_constant_linear_layers(weights)

    # Print the results
    print(f"Lipschitz constant using spectral norms: {spectral_norm_lipschitz}")
    print(
        f"Lipschitz constant using lipschitz_constant_linear_layers: {lipschitz_constant_l0}"
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_norm_conv(conv_layer, input_shape, n_iterations=100, device='cpu'):
    """
    Estimates the spectral norm of a convolutional layer using the power iteration method.

    Args:
        conv_layer (nn.Conv2d): The convolutional layer.
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).
        n_iterations (int): Number of power iterations.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        float: Estimated spectral norm of the convolutional layer.
    """
    # Initialize a random input tensor with requires_grad=True
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    x = x / x.norm()

    for _ in range(n_iterations):
        # Forward pass
        y = conv_layer(x)
        y_norm = y.norm()

        # Compute gradients
        grad_x = torch.autograd.grad(y_norm, x, retain_graph=True, create_graph=False)[0]

        # Normalize the gradient to get the next iteration input
        x = grad_x / (grad_x.norm() + 1e-12)
        x = x.detach().requires_grad_(True)  # Detach to prevent gradient accumulation

    # After iterations, compute the final norm
    with torch.no_grad():
        y = conv_layer(x)
        sigma = y.norm().item()

    return sigma

def spectral_norm_linear(linear_layer, n_iterations=100, device='cpu'):
    """
    Estimates the spectral norm of a linear layer using the power iteration method.

    Args:
        linear_layer (nn.Linear): The linear layer.
        n_iterations (int): Number of power iterations.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        float: Estimated spectral norm of the linear layer.
    """
    # Initialize a random vector for power iteration
    u = torch.randn(linear_layer.weight.size(0), device=device)
    u = u / u.norm()

    for _ in range(n_iterations):
        # v = W^T u
        v = torch.mv(linear_layer.weight.t(), u)
        v = v / (v.norm() + 1e-12)

        # u = W v
        u = torch.mv(linear_layer.weight, v)
        u = u / (u.norm() + 1e-12)

    # Estimated spectral norm
    sigma = torch.dot(u, torch.mv(linear_layer.weight, v)).item()
    return sigma

def compute_lipschitz_constant(model, input_shape):
    """
    Computes an upper bound on the Lipschitz constant of the given model.

    Args:
        model (nn.Module): The neural network model.
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).

    Returns:
        float: Upper bound on the Lipschitz constant of the model.
    """
    lipschitz_constants = []
    modules = list(model.network)

    current_input_shape = input_shape

    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            sigma = spectral_norm_conv(layer, current_input_shape, device=layer.weight.device)
            lipschitz_constants.append(sigma)

            # Compute output shape after convolution
            # Using the formula: ((W - F + 2P) / S) + 1
            batch_size, in_channels, H_in, W_in = current_input_shape
            F = layer.kernel_size[0]
            P = layer.padding[0]
            S = layer.stride[0]
            H_out = (H_in + 2 * P - F) // S + 1

            F = layer.kernel_size[1]
            P = layer.padding[1]
            S = layer.stride[1]
            W_out = (W_in + 2 * P - F) // S + 1

            current_input_shape = (batch_size, layer.out_channels, H_out, W_out)

        elif isinstance(layer, nn.Linear):
            sigma = spectral_norm_linear(layer, device=layer.weight.device)
            lipschitz_constants.append(sigma)
            # After linear layer, the shape is (batch_size, out_features)
            current_input_shape = (current_input_shape[0], layer.out_features)

        elif isinstance(layer, nn.ReLU):
            # ReLU is 1-Lipschitz
            lipschitz_constants.append(1.0)

        elif isinstance(layer, nn.Flatten):
            # Flatten doesn't change Lipschitz constant
            lipschitz_constants.append(1.0)
            # After flatten, the shape is (batch_size, -1)
            current_input_shape = (current_input_shape[0], -1)

        else:
            # For other layers (e.g., pooling), assume Lipschitz constant of 1
            lipschitz_constants.append(1.0)

    # Multiply the Lipschitz constants
    total_lipschitz_constant = 1.0
    for L in lipschitz_constants:
        total_lipschitz_constant *= L

    # Account for input scaling (dividing by 255.0)
    total_lipschitz_constant *= (1.0 / 255.0)

    return total_lipschitz_constant

# Define your QNetwork with a known action space size
class QNetwork(nn.Module):
    def __init__(self, action_space_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# Example usage:
if __name__ == "__main__":
    # Assuming input images are of size (4, 84, 84) as in Atari games
    input_shape = (1, 4, 84, 84)  # Batch size of 1

    # Define the action space size
    action_space_n = 4  # Replace with env.single_action_space.n

    # Initialize the model
    model = QNetwork(action_space_n)

    # Move model to appropriate device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set model to evaluation mode to disable dropout, batchnorm, etc.
    model.eval()

    # Compute the Lipschitz constant
    with torch.no_grad():
        lipschitz_constant = compute_lipschitz_constant(model, input_shape)
    print(f"Upper bound on the Lipschitz constant of the network: {lipschitz_constant}")

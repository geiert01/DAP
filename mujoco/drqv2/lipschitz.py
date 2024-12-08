import torch
import torch.nn as nn

def spectral_norm_linear(linear_layer, n_iterations=20, device="cpu"):
    # Provided spectral norm calculation for linear layers
    u = torch.randn(linear_layer.weight.size(0), device=device)
    u = u / u.norm()

    for _ in range(n_iterations):
        v = torch.mv(linear_layer.weight.t(), u)
        v = v / (v.norm() + 1e-12)
        u = torch.mv(linear_layer.weight, v)
        u = u / (u.norm() + 1e-12)

    sigma = torch.dot(u, torch.mv(linear_layer.weight, v)).item()
    return sigma

def spectral_norm_conv(conv_layer, input_shape, n_iterations=20, device="cpu"):
    """
    Estimates the spectral norm of a convolutional layer using the power iteration method.
    """
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    x = x / x.norm()
    for _ in range(n_iterations):
        y = conv_layer(x)
        y_norm = y.norm()
        grad_x = torch.autograd.grad(y_norm, x, retain_graph=True, create_graph=False)[0]
        x = grad_x / (grad_x.norm() + 1e-12)
        x = x.detach().requires_grad_(True)

    with torch.no_grad():
        y = conv_layer(x)
        sigma = y.norm().item()

    return sigma

def lipschitz_constant_linear_layers(weights, c=1.0, device="cpu"):
    """
    Compute the Lipschitz constant for a chain of linear layers using eigenvalues.
    """
    m = 0.5  # For ReLU
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
    L_bar = (1 / eig_min)**0.5
    W_l = weights[-1]
    norm_W_l = torch.linalg.norm(W_l, 2).item()
    L0 = (c**0.5) * L_bar * norm_W_l
    return L0

def compute_linear_lipschitz(model, device="cpu"):
    """
    Compute lipschitz constant for a sequence of linear layers in a model.
    Extracts all nn.Linear layers and computes using lipschitz_constant_linear_layers.
    """
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data)
    if not weights:
        return 1.0  # No linear layers
    return lipschitz_constant_linear_layers(weights, device=device)

def compute_encoder_lipschitz(encoder, obs_shape, device="cpu"):
    """
    Compute the Lipschitz constant of the encoder.
    The encoder consists of conv layers (DrQV2).
    We'll multiply the spectral norms of all convolutional layers.
    obs_shape: (C,H,W)
    We'll assume a batch_size=1 for spectral norm estimation.
    """
    L = 1.0
    input_shape = (1, ) + obs_shape  # (1, C, H, W)
    for layer in encoder.convnet:
        if isinstance(layer, nn.Conv2d):
            L *= spectral_norm_conv(layer, input_shape, device=device)
            # Update input_shape after conv?
            # After each Conv2d, output shape changes. Let's do a forward pass 
            # to track shape:
            with torch.no_grad():
                dummy_in = torch.randn(*input_shape, device=device)
                dummy_out = layer(dummy_in)
                input_shape = dummy_out.shape
    return L

def compute_actor_lipschitz(actor, device="cpu"):
    """
    Compute Lipschitz constant of the Actor:
    Actor has:
        trunk: linear + layernorm + Tanh (we consider linear for spectral norm)
        policy: multiple linear layers

    Steps:
    - Compute spectral norm of the first linear layer in trunk.
    - Compute lipschitz_constant_linear_layers for the policy part.
    - Multiply them.
    """
    # trunk spectral norm (just the linear in trunk)
    trunk_linear = None
    for layer in actor.trunk:
        if isinstance(layer, nn.Linear):
            trunk_linear = layer
            break
    trunk_sn = spectral_norm_linear(trunk_linear, device=device) if trunk_linear is not None else 1.0

    # policy linear layers lipschitz
    policy_weights = []
    for layer in actor.policy:
        if isinstance(layer, nn.Linear):
            policy_weights.append(layer.weight.data)
    if policy_weights:
        policy_lipschitz = lipschitz_constant_linear_layers(policy_weights, device=device)
    else:
        policy_lipschitz = 1.0

    return trunk_sn * policy_lipschitz

def compute_critic_lipschitz(critic, device="cpu"):
    """
    Compute Lipschitz constants for Critic Q1 and Q2:
    Critic has:
        trunk: linear + layernorm + Tanh
        Q1: linear layers
        Q2: linear layers

    Steps:
    - Compute spectral norm for trunk's initial linear layer.
    - For Q1 and Q2, extract linear weights and compute lipschitz.
    - Multiply trunk_spectral_norm * Q1_lipschitz and trunk_spectral_norm * Q2_lipschitz respectively.
    """
    # trunk linear spectral norm
    trunk_linear = None
    for layer in critic.trunk:
        if isinstance(layer, nn.Linear):
            trunk_linear = layer
            break
    trunk_sn = spectral_norm_linear(trunk_linear, device=device) if trunk_linear else 1.0

    # Q1 linear layers
    q1_weights = []
    for layer in critic.Q1:
        if isinstance(layer, nn.Linear):
            q1_weights.append(layer.weight.data)
    if q1_weights:
        q1_lipschitz = lipschitz_constant_linear_layers(q1_weights, device=device)
    else:
        q1_lipschitz = 1.0

    # Q2 linear layers
    q2_weights = []
    for layer in critic.Q2:
        if isinstance(layer, nn.Linear):
            q2_weights.append(layer.weight.data)
    if q2_weights:
        q2_lipschitz = lipschitz_constant_linear_layers(q2_weights, device=device)
    else:
        q2_lipschitz = 1.0

    return trunk_sn * q1_lipschitz, trunk_sn * q2_lipschitz

def compute_total_lipschitz(agent, obs_shape, device="cpu"):
    """
    Compute combined Lipschitz constants:
    1) Lipschitz_Encoder
    2) Lipschitz_Actor
    3) Lipschitz_Critic_Q1
    4) Lipschitz_Critic_Q2
    5) Combined:
       - Encoder * Actor
       - Encoder * Critic_Q1
       - Encoder * Critic_Q2
    """
    L_encoder = compute_encoder_lipschitz(agent.encoder, obs_shape, device=device)
    L_actor = compute_actor_lipschitz(agent.actor, device=device)
    L_q1, L_q2 = compute_critic_lipschitz(agent.critic, device=device)

    return {
        "L_encoder": L_encoder,
        "L_actor": L_actor,
        "L_critic_q1": L_q1,
        "L_critic_q2": L_q2,
        "L_encoder_actor": L_encoder * L_actor,
        "L_encoder_critic_q1": L_encoder * L_q1,
        "L_encoder_critic_q2": L_encoder * L_q2,
    }

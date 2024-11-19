# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "atari"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "ALE/DemonAttack-v5"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e4)
    """the replay memory buffer size"""
    gamma: float = 0.90
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 5000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.001
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)  # wait a rand num of frames until env starts (max 30)
        env = MaxAndSkipEnv(env, skip=4)  # only consider every 4th frame (fix action for 4 frames)
        env = EpisodicLifeEnv(env)  # treats losing life as failing, but env continues until all lifes are gone
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)  # if one needs to press FIRE to start the game, this is handled here
        env = ClipRewardEnv(env)  # don't adjust to reward size, just to good or bad: rewards = {-1, 0, 1}
        env = gym.wrappers.ResizeObservation(env, (84, 84))  # smaller representation of image by interpolation
        env = gym.wrappers.GrayScaleObservation(env)  # convert to greyscale
        env = gym.wrappers.FrameStack(env, 4)  # use 4 frames as one input to get temporal information

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            # nn.Conv2d(num input channels, num output channels/filters, size of conv filter, step size of filter)
            # nn.Conv2d(4 stacked images, output: 32 feature maps, 8x8-filter-size, filter moves 4 pixels)
            # nn.Conv2d(4 images, 32 images where filter was applied/each filter produces 1 op, size, stride)
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def spectral_norm_conv(conv_layer, input_shape, n_iterations=20, device='cpu'):
    """
    Estimates the spectral norm of a convolutional layer using the power iteration method.

    Args:
        conv_layer (nn.Conv2d): The convolutional layer whose spectral norm we want to estimate.
        input_shape (tuple): The shape of the input tensor (batch_size, channels, height, width).
        n_iterations (int): Number of iterations to perform in the power method.
        device (str): The device to perform computations on ('cpu' or 'cuda').

    Returns:
        float: Estimated spectral norm (largest singular value) of the convolutional layer.
    """

    # start with normalized x_0
    # we need x_{k+1} = \frac{A^T A x_k}{||A^T A x_k||}
    # in the end \lambda_\max = \frac{x_k^T A^T A x_k}{x_k^T x_k}

    # Initialize a random input tensor with requires_grad=True
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    x = x / x.norm()  # x_0

    for _ in range(n_iterations):
        # Forward pass
        y = conv_layer(x)  # A x_k
        y_norm = y.norm()  # ||A x_k||

        # Compute gradients
        grad_x = torch.autograd.grad(y_norm, x, retain_graph=True, create_graph=False)[0]  # \frac{A^T A x_k}{||A x_k||}

        # Normalize the gradient to get the next iteration input
        x = grad_x / (grad_x.norm() + 1e-12)  # \frac{||A^T A x_k||}{||A x_k||} / \frac{A^T A x_k}{||A x_k||}
        # this x is then the x_{k+1}
        x = x.detach().requires_grad_(True)  # Detach to prevent gradient accumulation
        # to stop comp. graph from growing, which stops redundant memory usage

    # After iterations, compute the final norm
    with torch.no_grad():  # lambda_max = ||A x||, since x is the eigenvector of the largest eigenvalue
        y = conv_layer(x)
        sigma = y.norm().item()

    return sigma


def spectral_norm_linear(linear_layer, n_iterations=20, device='cpu'):
    """
    Estimates the spectral norm of a linear layer using the power iteration method.

    Args:
        linear_layer (nn.Linear): The linear layer whose spectral norm we want to estimate.
        n_iterations (int): Number of iterations for the power method (default is 20).
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        float: Estimated spectral norm of the linear layer.
    """
    # Initialize a random vector for power iteration
    u = torch.randn(linear_layer.weight.size(0), device=device)
    u = u / u.norm()

    for _ in range(n_iterations):
        # v = W^T u
        v = torch.mv(linear_layer.weight.t(), u)  # torch.mv ... matrix-vector-mult.
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
    """
    device = next(model.parameters()).device
    lipschitz_constants = []
    modules = list(model.network)

    current_input_shape = input_shape

    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            sigma = spectral_norm_conv(layer, current_input_shape, device=device)
            lipschitz_constants.append(sigma)

            # Compute output shape after convolution
            # Using the formula: ((W - F + 2P) / S) + 1
            batch_size, in_channels, H_in, W_in = current_input_shape
            H_out = (H_in + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[
                0] + 1
            W_out = (W_in + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[
                1] + 1
            current_input_shape = (batch_size, layer.out_channels, H_out, W_out)

        elif isinstance(layer, nn.Linear):
            sigma = spectral_norm_linear(layer, device=device)
            lipschitz_constants.append(sigma)
            # After linear layer, the shape is (batch_size, out_features)
            current_input_shape = (current_input_shape[0], layer.out_features)

        elif isinstance(layer, nn.ReLU):
            # ReLU is 1-Lipschitz
            lipschitz_constants.append(1.0)

        elif isinstance(layer, nn.Flatten):
            # Flatten doesn't change Lipschitz constant
            lipschitz_constants.append(1.0)
            # After flatten, the shape is (current batch size, -1)
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


if __name__ == "__main__":

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")


    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        # most important methods:
        # add(): adds new transition with obs, act, r, done (Once buffer full, new entries replace the oldest ones)
        # sample(batch_size): samples from saved transitions
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,  # next observation stored in observations to save memory
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)  # qval of best act in next state
                    # = Q(S_{t+1}, A_{t+1})
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    # td_target = R_{t+1} + \gamma max_a Q(S_{t+1}, a); [(1 - data.dones.flatten()) gets rid of last
                    # term, which we don't need if we reached a terminal state
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()  # Q(s,a)
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        # Compute and log the Lipschitz constant every 1000 steps
        if global_step % 1000 == 0:
            lipschitz_constant = compute_lipschitz_constant(q_network, input_shape=(1, 4, 84, 84))
            writer.add_scalar("charts/lipschitz_constant", lipschitz_constant, global_step)
            print(f"Global step {global_step}: Lipschitz constant = {lipschitz_constant}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()

import os
import torch
import gymnasium as gym
import numpy as np
from torch import nn
from torch.distributions.categorical import Categorical


# Define the Agent class
class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Actor-critic network
        self.critic = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n),
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    # Set up the environment
    env_id = "CartPole-v1"  # Environment ID
    env = gym.make(env_id, render_mode="human")  # "human" mode for rendering

    # Device setup (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the agent
    agent = Agent(env).to(device)

    # Load the pre-trained model
    model_path = "runs/CartPole-v1__ppo__1__1729787670/ppo_model.pth"  # Path to the saved model
    agent.load_state_dict(torch.load(model_path))
    agent.eval()  # Set the model to evaluation mode

    # Evaluate the agent for a few episodes
    num_episodes = 2  # Number of episodes to evaluate
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Reset the environment
        obs = torch.Tensor(obs).to(device)  # Convert observation to PyTorch tensor
        done = False
        total_reward = 0

        while not done:
            # No gradient calculation needed during evaluation
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)

            # Take action in the environment
            obs, reward, done, truncated, info = env.step(action.cpu().numpy())

            # Convert the new observation to PyTorch tensor
            obs = torch.Tensor(obs).to(device)

            total_reward += reward  # Accumulate reward for the episode

            if done or truncated:  # Check for episode termination
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")  # Print total reward for the episode

    # Close the environment
    env.close()

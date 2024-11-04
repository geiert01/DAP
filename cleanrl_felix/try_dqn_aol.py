import os
import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import pygame
import time

# Define the custom AOLLinear layer
class AOLLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(AOLLinear, self).__init__()
        self.P = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        PTP = torch.abs(self.P.T @ self.P)
        D = torch.diag(1.0 / torch.sqrt(PTP.sum(dim=0) + 1e-10))
        W = self.P @ D
        return x @ W.T + self.bias

# Define the QNetwork class with AOLLinear layers
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            AOLLinear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            AOLLinear(120, 84),
            nn.ReLU(),
            AOLLinear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

# Initialize device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment
env = gym.make('MountainCar-v0', render_mode='rgb_array')
obs, _ = env.reset()

# Initialize QNetwork with the correct AOL structure
q_network = QNetwork(env).to(device)

# Load the saved model
model_path = 'runs/MountainCar-v0__dqn_aol__1__1730384390/dqn_aol.cleanrl_model'
q_network.load_state_dict(torch.load(model_path))
q_network.eval()  # Set the model to evaluation mode

# Initialize pygame display
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("MountainCar Visualization")

# Evaluate the model
total_rewards = 0
for _ in range(1000):  # Run for 1000 timesteps
    with torch.no_grad():
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        q_values = q_network(obs_tensor)
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]

    # Step the environment and get the next observation
    obs, reward, done, truncated, info = env.step(action)
    total_rewards += reward

    # Render the frame and display it using pygame
    frame = env.render()
    frame = np.transpose(frame, (1, 0, 2))  # Transpose for pygame
    surface = pygame.surfarray.make_surface(frame)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    time.sleep(0.02)

    if done or truncated:
        obs, _ = env.reset()

print(f"Total reward during evaluation: {total_rewards}")
env.close()
pygame.quit()

import os
import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import pygame
import time


# Define the QNetwork class
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


# Initialize device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the CartPole environment
#env = gym.make('CartPole-v1', render_mode='rgb_array')
env = gym.make('MountainCar-v0', render_mode='rgb_array')
obs, _ = env.reset()

# Initialize QNetwork
q_network = QNetwork(env).to(device)

# Load the saved model (Update the path to the actual model path)
model_path = 'runs/MountainCar-v0__dqn_aol__1__1730384390/dqn_aol.cleanrl_model'
#model_path = 'runs/MountainCar-v0__dqn__1__1729613159/dqn.cleanrl_model'
#model_path = 'runs/CartPole-v1__dqn__1__1728998507/dqn.cleanrl_model'
q_network.load_state_dict(torch.load(model_path))
q_network.eval()  # Set the model to evaluation mode

# Initialize pygame display
pygame.init()
screen = pygame.display.set_mode((600, 400))  # Adjust the size if needed
pygame.display.set_caption("CartPole Visualization")

# Evaluate the model in the CartPole environment
total_rewards = 0
for _ in range(1000):  # Run for 1000 timesteps
    with torch.no_grad():
        # Add an extra dimension to obs to make it 2D (batch size of 1)
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)

        q_values = q_network(obs_tensor)
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]

    # Step the environment and get the next observation
    obs, reward, done, truncated, info = env.step(action)
    total_rewards += reward

    # Render the frame and display it using pygame
    frame = env.render()  # Render the environment as an image
    frame = np.transpose(frame, (1, 0, 2))  # new
    surface = pygame.surfarray.make_surface(frame)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    time.sleep(0.02)

    if done or truncated:
        obs, _ = env.reset()

print(f"Total reward during evaluation: {total_rewards}")
env.close()
pygame.quit()

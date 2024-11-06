import argparse
import os
from distutils.util import strtobool
import time
import random
import numpy as np
import torch
import gymnasium as gym


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                        help='the name of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, torch.backend.cudnn.deterministic=False')
    parser.add_argument('--mps', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, mps will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked by weights and biases')
    parser.add_argument('--wandb-project-name', type=str, default='cleanL_yt_tutorial',
                        help='the name of the wandb project')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='the name of the wandb entity')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.gym_id}__{args.env_name}__{args.seed}__{int(time.time())}"
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    observation = env.reset()
    #episodic_return = 0


    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        #episodic_return += reward
        if terminated:
            observation = env.reset()
            print(f"episodic return: {info['episode']['r']}")
            #print(f"episodic return: {episodic_return}")
            #episodic_return = 0
    env.close()

    def make_env(gym_id, seed):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    observation = envs.reset()
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)
        for item in info:
            print(f"episodic return {item['episode']['r']}")
            # NOTE: there is no observation=env.reset() anymore
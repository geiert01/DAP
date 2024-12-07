# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import time
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    print(cfg.obs_shape)
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

def lipschitz_constant_linear_layers(weights, c=1.0, device="cpu"):
    # Compute the Lipschitz constant for linear layers using eigenvalues.

    m = 0.5  # Slope approximation for ReLU
    weights = [W.to(device) for W in weights]
    n0 = weights[0].shape[1]  # Input dimension of the first layer

    # Initialize matrix for eigenvalue computations
    X_prev = c * torch.eye(n0, device=device)

    for i in range(len(weights) - 1):
        W_i = weights[i]
        X_prev_inv = torch.linalg.pinv(X_prev)  # Compute pseudo-inverse of X_prev
        A_i = W_i @ X_prev_inv @ W_i.T  # Construct symmetric matrix A_i

        eig_max = torch.linalg.eigvalsh(A_i).max().item()  # Largest eigenvalue of A_i
        lambda_i = 1 / (2 * m**2 * eig_max)  # Scaling factor for this layer
        n_i = W_i.shape[0]  # Output dimension of the current layer

        # Update X_i based on the current layer
        X_i = lambda_i * torch.eye(n_i, device=device) - lambda_i**2 * m**2 * A_i
        X_prev = X_i

    eig_min = torch.linalg.eigvalsh(X_prev).min().item()  # Smallest eigenvalue of the final matrix
    L_bar = (1 / eig_min) ** 0.5  # Intermediate Lipschitz bound

    W_l = weights[-1]  # Weight matrix of the last layer
    norm_W_l = torch.linalg.norm(W_l, 2).item()  # Spectral norm of the last weight matrix

    # Final Lipschitz constant computation
    L0 = (c**0.5) * L_bar * norm_W_l
    return L0


def spectral_norm_conv(conv_layer, input_shape, n_iterations=20, device="cpu"):
    # Estimate the spectral norm of a Conv2d layer using the power iteration method.

    # Initialize a random input tensor and normalize it
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    x = x / x.norm()

    for _ in range(n_iterations):
        # Apply the Conv2d layer
        y = conv_layer(x)
        y_norm = y.norm()

        # Compute the gradient of the norm w.r.t. the input
        grad_x = torch.autograd.grad(y_norm, x, retain_graph=True, create_graph=False)[0]

        # Normalize the gradient to prepare for the next iteration
        x = grad_x / (grad_x.norm() + 1e-12)
        x = x.detach().requires_grad_(True)  # Detach to avoid graph growth

    # Final spectral norm estimation
    with torch.no_grad():
        y = conv_layer(x)
        sigma = y.norm().item()
    return sigma


def compute_combined_lipschitz_constant(network, input_shape, device="cpu"):
    """
    Compute the combined Lipschitz constant for a network.

    Args:
        network: The PyTorch nn.Module to analyze.
        input_shape: The shape of the input tensor.
        device: The device to perform computations on.

    Returns:
        Combined Lipschitz constant.
    """
    L = 1.0
    linear_weight_list = []

    for layer in network.modules():
        if isinstance(layer, nn.Conv2d):
            # Compute spectral norm for Conv2d layers
            spectral_norm = spectral_norm_conv(layer, input_shape, device=device)
            L *= spectral_norm
        elif isinstance(layer, nn.Linear):
            # Collect Linear layer weights
            linear_weight_list.append(layer.weight.data)

    if linear_weight_list:
        # Compute Lipschitz constant for Linear layers
        linear_lipschitz = lipschitz_constant_linear_layers(linear_weight_list, device=device)
        L *= linear_lipschitz

    # Log the combined Lipschitz constant
    return L





class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = f"{cfg.exp_name}_{cfg.seed}__{int(time.time())}"
            proj_name = "mujoco"
        
            wandb.init(
            project=proj_name,
            name=exp_name# ,
            # config=cfg
            )
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                
                # if "obs_shape" not in self.cfg:
                #    self.cfg.obs_shape = self.train_env.observation_spec().shape

                
                # Compute and log Lipschitz constants every 1000 steps
                if self.global_step % 1000 == 0:
                    actor_lipschitz = compute_combined_lipschitz_constant(
                        network=self.agent.actor,
                        input_shape=(1, 9, 84, 84),  # Replace `state_dim` with the actual state dimensionality
                        device=self.device
                    )
                    critic_lipschitz = compute_combined_lipschitz_constant(
                        network=self.agent.critic,
                        input_shape=(1, 9, 84, 84),  # Replace `state_dim` accordingly
                        device=self.device
                    )

                    combined_lipschitz = actor_lipschitz * critic_lipschitz
                    
                else:
                   combined_lipschitz = None  # Skip computation if not a logging step
                
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        
                        if combined_lipschitz is not None:
                            log("lipschitz_constant", combined_lipschitz)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
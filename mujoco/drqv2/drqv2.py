# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import wandb


# Class DrQV2Agent is defined puts everything together.


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad  # how many pixels to pad before randomly shifting the image

    def forward(self, x):  # x...batch of images
        n, c, h, w = x.size()  # batch-size, channels, height, width
        # e.g., channels=3 for RGB or 1 for grayscale (or 3 for stacked grayscale; or 9 for stacked RGB)
        assert h == w
        padding = tuple([self.pad] * 4)  # padding tuple: left, right, top, bottom
        x = F.pad(x, padding, 'replicate')  # pads by replicating the border pixels, so a random crop/shift is possible
        eps = 1.0 / (h + 2 * self.pad)  # (h+2*pad) is the new height/width after padding
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)  # vertical stack of h rows, each like "arange"
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)  # pair it with transposed
        # version to get (x,y) coordinates
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # repeat batch-size "n" (when working w/ batch of images)

        # creating random offset
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)  # normalized coordinates

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',  # if for whatever reason you sample outside the padded image
                             align_corners=False)


class Encoder(nn.Module):
    '''
    Processes pixels and outputs feature vector
    '''
    def __init__(self, obs_shape):  # obs_shape: (channels, height, width)
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35  # dim of encoders output (for fully connected layers)
        # Output-size formula: [(W−K+2P)/S]+1, W:Width, K:Kernel, P:Padding, S:Stride
        # Here: Image is 84x84
        # First Layer: (84-3+0)/2 + 1 = 41
        # Second Layer: (41-3+0)/1 + 1 = 39
        # Third Layer: (39-3+0)/1 + 1 = 37
        # Fourth Layer: (37-3+0)/1 + 1 = 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)  # apply (custom) weight initialization

    def forward(self, obs):  # takes pixels and outputs feature vector
        obs = obs / 255.0 - 0.5  # normalize to [-0.5, 0.5] (stabilizing)
        h = self.convnet(obs)  # apply CNN and h has dim (N, 32, 35, 35), N...batch-size
        h = h.view(h.shape[0], -1)  # flatten to (N, 32*35*35)
        return h


class Actor(nn.Module):
    '''
    Takes encoded features and outputs action distribution
    '''
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        # (input-dim, shape-of-action, intermediate-dim, MLP-layer-dim)
        # action_shape: (dim-of-action, ), e.g., (2,) for steering_wheel and horn_on_off
        super().__init__()

        # map ops to smaller dimension
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        # Because of this input dependence, global Lipschitz constant might not exist.
        # Locally, if certain conditions are met (like bounded variance, which I think we have fue to clipping),
        # you could find a finite Lipschitz constant.

        # policy network
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)  # custom weight initialization

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)  # tanh squashes to [-1, 1], so we don't propose out-of-bounds actions
        std = torch.ones_like(mu) * std  # std is fixed scheduled

        dist = utils.TruncatedNormal(mu, std)  # truncated Normal Dist restricts to [-1, 1]
        return dist


class Critic(nn.Module):
    '''
    Defines Q-val functions (2 for stability).
    Input: encoded features and action
    Output: Q-value-A and Q-value-B
    '''
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    '''
    Puts everything together:
    .) Creates the Networks: Encoder, Actor, Critic
    .) Implement methods to act, update critic and update actor
    .) Uses data aug., schedules std and optimizers
    '''
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, use_wandb):  # TODO include wandb
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)  # create Encoder
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,  # create Actor
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,  # create Critic
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,  # create Critic Target
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # to have same weights at beginning as Critic

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # need separate optimizers, because we update each network differently

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()  # switch all trainable networks to training mode (switch it off when evaluating)
        # by now: encoder, actor, critic are in training mode; critic_target is not
        self.critic_target.train()  # to have this in training mode as well

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):  # how to act in the environment
        # returns action
        obs = torch.as_tensor(obs, device=self.device)  # transform to tensor
        obs = self.encoder(obs.unsqueeze(0))  # to fit the needed dimension: (1,C,H,W) instead of (C,H,W)
        stddev = utils.schedule(self.stddev_schedule, step)  # exploration noise schedule
        dist = self.actor(obs, stddev)  # get action distribution
        # Choose action:
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)  # for less overestimation
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        if self.use_wandb:
            wandb.log({"train/critic_target_q": target_Q.mean().item()}, step=step)
            wandb.log({"train/critic_q1": Q1.mean().item()}, step=step)
            wandb.log({"train/critic_q2": Q2.mean().item()}, step=step)
            wandb.log({"train/critic_loss": critic_loss.item()}, step=step)

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        if self.use_wandb:
            wandb.log({"train/actor_loss": actor_loss.item()}, step=step)
            wandb.log({"train/actor_logprob": log_prob.mean().item()}, step=step)
            wandb.log({"train/actor_ent": dist.entropy().sum(dim=-1).mean().item()}, step=step)

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        if self.use_wandb:
            wandb.log({"train/batch_reward": reward.mean().item()}, step=step)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

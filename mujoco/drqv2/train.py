# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'  # use Intel's MKL fro computation
os.environ['MUJOCO_GL'] = 'egl'  # headless rendering

from pathlib import Path  # filepath manipulations

import time
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True  # optimize GPU performance


def make_agent(obs_spec, action_spec, cfg):
    '''
    Instantiate the agent using the configuration.
    '''
    cfg.obs_shape = obs_spec.shape  # set obs. shape in config
    cfg.action_shape = action_spec.shape  # set action shape in config
    return hydra.utils.instantiate(cfg)  # dynamic instantiation of objects

def sanitize_config(cfg):
    '''
    Clean the configuration dictionary to ensure all values are serializable (e.g., when logging to W&B).
    '''
    sanitized = {}
    for key, value in cfg.items():
        # Skip placeholders and non-serializable values
        if isinstance(value, dict):
            sanitized[key] = sanitize_config(value)  # Recursively sanitize dicts
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value  # keep serializable types (e.g., str, int, float, bool)
        else:
            sanitized[key] = str(value)  # Convert unsupported types to strings
    return sanitized


class Workspace:
    '''
    Initialize workspace; setting up:
    .) directories (where what will be saved),
    .) seeds,
    .) device configurations (where CPU or GPU),
    .) W&B logging,
    .) environments (setting up training and evaluation environments),
    .) agent
    .) Replay buffer and loader (prep storage and retrieval of experiences),
    .) video recorder (setting up tools to record videos during training),
    '''
    def __init__(self, cfg):
        self.work_dir = Path.cwd()  # current working directory: base for saving logs, models, etc.
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        # wandb
        if self.cfg.use_wandb:
            exp_name = f"{cfg.exp_name}_{cfg.seed}__{int(time.time())}"
            proj_name = "mujoco"
        
            wandb.init(
            project=proj_name,
            name=exp_name,
            config=sanitize_config(cfg)  # maybe comment out (new)
            )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)  # device based on configuration
        self.setup()  # (function below) set up environments and other components
        self.agent = make_agent(self.train_env.observation_spec(),  # shape of obs. that agent receives
                                self.train_env.action_spec(),  # shape of actions the agent will output
                                self.cfg.agent)  # contains hyperparams specific to the agent
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        '''
        set up logger, environments, replay buffer, video recorder for training
        '''
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)

        # create envs
        # sets up DMC Suite tasks with param: name, frame_stack, action_repeat(num times), seed
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)


        # create replay buffer
        # def structure of data in buffer: (obs, action, reward, discount)
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        # storage mechanism for saving experiences to disk
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')  # where to save

        # provide batches of experiences
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None  # will be used to iterate over replay buffer

        # set up tools for recording videos
        self.video_recorder = VideoRecorder(  # record videos during evaluation
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(  # record videos during training
            self.work_dir if self.cfg.save_train_video else None)

    # easy access to global training counters and the replay iterator
    # cumulative steps taken in environment
    @property  # @property to access method as attribute [no "()" needed]
    def global_step(self):
        return self._global_step

    # num of completed episodes (from env reset to termination)
    @property
    def global_episode(self):
        return self._global_episode

    # number of frames the agent has seen (accounts for action repeat)
    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    # ensure rb iterator initialized only once
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):  # regularly monitor agents performance
        # step: num steps taken during eval
        # episode: num episodes completed during eval
        # total_reward: total reward accumulated during eval
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)  # True until condition
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()  # start new episode
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))  # only store 1 eval epis per eval-run
            while not time_step.last():
                # select action for current observation
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)  # applies action to receive next state
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')  # saves eval-episode video

        # log eval stats
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:  # handles logging
            log('episode_reward', total_reward / episode)  # avg episode reward
            log('episode_length', step * self.cfg.action_repeat / episode)  # avg episode length
            log('episode', self.global_episode)  # total num of episodes
            log('step', self.global_step)  # total num of training-steps up to this point

    def train(self):
        # predicates
        # train until num of frames reached
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        # seed until num of seed frames reached (random decisions at beginning)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        # eval every num of frames
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()  # initial step/state
        self.replay_storage.add(time_step)  # add initial step to replay buffer
        self.train_video_recorder.init(time_step.observation)   # init video recorder
        metrics = None  # will hold training metrics after update
        while train_until_step(self.global_step):  # main loop
            if time_step.last():  # if episode ended
                self._global_episode += 1
                
                self.train_video_recorder.save(f'{self.global_frame}.mp4')

                # wait until all the metrics schema is populated
                # log training metrics
                if metrics is not None:  # only log if agent started updating
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat  # num of frames in episode
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                    

                # prepare for next episode
                time_step = self.train_env.reset()  # reset env
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)

                # save snapshot if specified so in config.yaml
                if self.cfg.save_snapshot:
                    self.save_snapshot()  # saves current state of training (agent, optimizer, states, etc.)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):  # weather it's time for eval
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)  # logs total training time before eval
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,  # current obs from environment
                                        self.global_step,  # global training step
                                        eval_mode=False)  # agent includes exploration noise

            # try to update the agent
            if not seed_until_step(self.global_step):  # after seeding phase (random action phase)
                metrics = self.agent.update(self.replay_iter, self.global_step)  # dict: metrics of update, e.g. loss
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        # after each episode ends, the code chacks if snapshot saving is enabled in the config
        # if yes, it saves the current state of training
        # For that, set: save_snapshot: true in config.yaml
        '''
        Save current state of training:
        .) agent: NN-architecture, weights, biases, optimizer states, etc.
        .) timer: time spent on training
        .) _global_step: num of steps taken so far
        .) _global_episode: num of episodes completed so far
        Allows to stop training and later resume from where it was stopped (with load_snapshot)
        '''
        snapshot = self.work_dir / 'snapshot.pt'  # where to save snapshot
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']  # attrib. of Worksp. that need to be saved
        payload = {k: self.__dict__[k] for k in keys_to_save}  # dict with keys from above and val corresp. from Worksp.
        with snapshot.open('wb') as f:  # serialize payload(dict) and write it to the file "f"
            torch.save(payload, f)

    def load_snapshot(self):
        # In the main-func, before starting training, the code checks if a snapshot file exists in the working directory
        # If it does, it calls workspace.load_snapshot() to restore the training state.
        # If you wish to start training from scratch, delete the snapshot.pt file before training
        '''
        Loads the saved state from a file, restoring the agent's parameters, timers, and counters.
        This enables resuming training without losing progress.
        '''
        snapshot = self.work_dir / 'snapshot.pt'  # path to be loaded
        with snapshot.open('rb') as f:  # reads saved state form file
            payload = torch.load(f)
        for k, v in payload.items():  # updates the Workspace instance's attributes with the loaded values
            self.__dict__[k] = v


# Hydra looks for your configuration files inside the cfgs directory,
# load config.yaml (named “config”) from the specified path
@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W  #  In some Hydra configs, delaying imports after main is called prevents issues.
    root_dir = Path.cwd()
    workspace = W(cfg)  # instantiate workspace with all the params of cdg
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
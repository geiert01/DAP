# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


# Function to calculate the length of an episode
def episode_len(episode):
    # Subtract 1 because the first transition is a dummy transition
    return next(iter(episode.values())).shape[0] - 1


# Function to save an episode to disk in a compressed format
def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)  # Save episode as a compressed .npz file
        bs.seek(0)  # Reset buffer pointer
        with fn.open('wb') as f:  # Write the buffer content to the file
            f.write(bs.read())


# Function to load an episode from disk
def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        # Convert the loaded data to a dictionary
        episode = {k: episode[k] for k in episode.keys()}
        return episode


# Class for storing replay buffer episodes
class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs  # Specification of data to store (e.g., obs, action, reward)
        self._replay_dir = replay_dir  # Directory where episodes will be stored
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)  # Temporary storage for the current episode
        self._preload()  # Preload existing episodes from disk

    # Returns the total number of transitions stored
    def __len__(self):
        return self._num_transitions

    # Add a time step to the replay buffer
    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            # Ensure value is in the correct shape and type
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)  # Add value to the current episode
        if time_step.last():  # If this is the last step in the episode
            # Save the episode to disk
            episode = {spec.name: np.array(self._current_episode[spec.name], spec.dtype)
                       for spec in self._data_specs}
            self._current_episode = defaultdict(list)  # Reset the current episode storage
            self._store_episode(episode)  # Store the episode

    # Preload existing episodes from disk to count total episodes and transitions
    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')  # Extract episode length from the filename
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    # Save an episode to disk and update counters
    def _store_episode(self, episode):
        eps_idx = self._num_episodes  # Current episode index
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
       
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)  # Save episode to file


# Class for managing replay buffer and sampling data
class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir  # Directory containing replay data
        self._size = 0  # Current size of the buffer
        self._max_size = max_size  # Maximum size of the buffer
        self._num_workers = max(1, num_workers)  # Number of parallel workers
        self._episode_fns = []  # List of episode filenames
        self._episodes = dict()  # Loaded episodes
        self._nstep = nstep  # Number of steps for n-step returns
        self._discount = discount  # Discount factor for rewards
        self._fetch_every = fetch_every  # Fetch frequency for loading new episodes
        self._samples_since_last_fetch = fetch_every  # Counter for fetch
        self._save_snapshot = save_snapshot  # Whether to save episode snapshots

    # Sample a random episode from the loaded episodes
    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    # Store a new episode in memory and maintain buffer size
    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)  # Length of the episode
        # Ensure buffer size does not exceed max size
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)  # Remove the oldest episode
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)  # Delete the file from disk
        # Add the new episode
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()  # Sort by episode filename
        self._episodes[eps_fn] = episode
        self._size += eps_len

        # Optionally delete the saved file if snapshots are not required
        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    # Try fetching new episodes from disk
    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return  # Fetch only every `fetch_every` samples
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id  # Get worker ID for parallel fetching
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)  # Get episode filenames
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue  # Distribute fetching across workers
            if eps_fn in self._episodes.keys():
                break  # Stop if the episode is already loaded
            if fetched_size + eps_len > self._max_size:
                break  # Stop if adding this episode exceeds max size
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break  # Stop on failure to store the episode

    # Sample a single transition for training
    def _sample(self):
        try:
            self._try_fetch()  # Fetch new episodes if needed
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()  # Randomly sample an episode
        # Randomly sample an index for n-step return
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]  # Current observation
        action = episode['action'][idx]  # Action taken
        next_obs = episode['observation'][idx + self._nstep - 1]  # Next observation
        reward = np.zeros_like(episode['reward'][idx])  # Initialize reward
        discount = np.ones_like(episode['discount'][idx])  # Initialize discount
        for i in range(self._nstep):  # Compute n-step reward and discount
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    # Infinite iterator for DataLoader
    def __iter__(self):
        while True:
            yield self._sample()


# Worker initialization function for parallel data loading
def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id  # Seed based on worker ID
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set random seed


# Function to create a DataLoader for the replay buffer
def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)  # Divide max size among workers

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,  # Pin memory for faster GPU transfers
                                         worker_init_fn=_worker_init_fn)
    return loader

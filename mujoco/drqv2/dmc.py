# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Any, NamedTuple

import dm_env  # DeepMind environment interface
import numpy as np
from dm_control import manipulation, suite  # Control suite and manipulation tasks
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


# Custom NamedTuple to extend the default time step returned by the environment
class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    # Check if the current step is the first step of the episode
    def first(self):
        return self.step_type == StepType.FIRST

    # Check if the current step is a mid-episode step
    def mid(self):
        return self.step_type == StepType.MID

    # Check if the current step is the last step of the episode
    def last(self):
        return self.step_type == StepType.LAST

    # Allow accessing attributes via index or name
    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


# Wrapper to repeat an action for multiple steps
class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env  # Base environment
        self._num_repeats = num_repeats  # Number of times to repeat the action

    # Execute the same action for multiple steps
    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)  # Take a step in the environment
            reward += (time_step.reward or 0.0) * discount  # Accumulate discounted rewards
            discount *= time_step.discount  # Update the discount factor
            if time_step.last():  # If the episode ends, stop repeating
                break
        return time_step._replace(reward=reward, discount=discount)  # Return the modified time step

    # Forward observation specification from the base environment
    def observation_spec(self):
        return self._env.observation_spec()

    # Forward action specification from the base environment
    def action_spec(self):
        return self._env.action_spec()

    # Reset the base environment
    def reset(self):
        return self._env.reset()

    # Delegate other attributes to the base environment
    def __getattr__(self, name):
        return getattr(self._env, name)


# Wrapper to stack multiple consecutive frames in the observation
class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env  # Base environment
        self._num_frames = num_frames  # Number of frames to stack
        self._frames = deque([], maxlen=num_frames)  # Deque to store frames
        self._pixels_key = pixels_key  # Key to extract pixel observations

        # Validate that the environment provides pixel observations
        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        # Get the shape of the pixel observations
        pixels_shape = wrapped_obs_spec[pixels_key].shape
        if len(pixels_shape) == 4:  # Remove batch dimension if present
            pixels_shape = pixels_shape[1:]

        # Define the shape of the stacked observation
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate([[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )

    # Transform the observation by stacking the stored frames
    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames  # Ensure correct number of frames
        obs = np.concatenate(list(self._frames), axis=0)  # Stack frames
        return time_step._replace(observation=obs)

    # Extract pixel observations from the time step
    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        if len(pixels.shape) == 4:  # Remove batch dimension if present
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()  # Transpose to channel-first format

    # Reset the environment and fill the frame stack with the initial observation
    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)  # Fill the stack with the same initial frame
        return self._transform_observation(time_step)

    # Step through the environment and update the frame stack
    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)  # Add the new frame
        return self._transform_observation(time_step)

    # Forward observation specification
    def observation_spec(self):
        return self._obs_spec

    # Forward action specification
    def action_spec(self):
        return self._env.action_spec()

    # Delegate other attributes to the base environment
    def __getattr__(self, name):
        return getattr(self._env, name)


# Wrapper to change the data type of actions
class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env  # Base environment
        wrapped_action_spec = env.action_spec()  # Get the action spec
        # Define a new action spec with the desired data type
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            'action'
        )

    # Convert actions to the desired data type before passing to the environment
    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    # Forward observation specification
    def observation_spec(self):
        return self._env.observation_spec()

    # Return the modified action specification
    def action_spec(self):
        return self._action_spec

    # Reset the environment
    def reset(self):
        return self._env.reset()

    # Delegate other attributes to the base environment
    def __getattr__(self, name):
        return getattr(self._env, name)


# Wrapper to augment the time step with additional fields (e.g., action)
class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env  # Base environment

    # Reset the environment and augment the time step
    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    # Step through the environment and augment the time step
    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    # Add the action to the time step
    def _augment_time_step(self, time_step, action=None):
        if action is None:  # If no action is provided, use zeros
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0
        )

    # Forward observation specification
    def observation_spec(self):
        return self._env.observation_spec()

    # Forward action specification
    def action_spec(self):
        return self._env.action_spec()

    # Delegate other attributes to the base environment
    def __getattr__(self, name):
        return getattr(self._env, name)


# Function to create a wrapped environment
def make(name, frame_stack, action_repeat, seed):
    # Parse the domain and task from the environment name
    domain, task = name.split('_', 1)
    # Handle specific domain/task name adjustments
    domain = dict(cup='ball_in_cup').get(domain, domain)

    # Load tasks from the suite or manipulation environment
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            task_kwargs={'random': seed},
            visualize_reward=False
        )
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'

    # Apply wrappers
    env = ActionDTypeWrapper(env, np.float32)  # Ensure actions are float32
    env = ActionRepeatWrapper(env, action_repeat)  # Repeat actions
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)  # Scale actions

    # Add rendering for classic tasks
    if (domain, task) in suite.ALL_TASKS:
        camera_id = dict(quadruped=2).get(domain, 0)  # Select camera ID for rendering
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(
            env,
            pixels_only=True,
            render_kwargs=render_kwargs
        )

    # Add frame stacking
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    # Extend the time step with additional fields
    env = ExtendedTimeStepWrapper(env)
    return env

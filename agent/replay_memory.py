from collections import namedtuple

import numpy as np
import torch
from gym import Env, spaces

Batch = namedtuple("Batch", ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory:
    def __init__(self, max_size, state_size, device="cpu"):
        """Replay memory implemented as a circular buffer.

        Args:
            max_size: Maximum size of the buffer
            state_size: Size of the state-space features for the environment
            device: Torch device to which the memory will be allocated ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.max_size = max_size
        self.state_size = state_size

        # Initialize memory structures on the specified device
        self.states = torch.empty((max_size, state_size), device=self.device)
        self.actions = torch.empty((max_size, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((max_size, 1), device=self.device)
        self.next_states = torch.empty((max_size, state_size), device=self.device)
        self.dones = torch.empty((max_size, 1), dtype=torch.bool, device=self.device)

        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Convert inputs to tensors directly on the specified device if not already."""
        self.states[self.idx] = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.actions[self.idx] = torch.as_tensor(
            action, dtype=torch.long, device=self.device
        )
        self.rewards[self.idx] = torch.as_tensor(
            reward, dtype=torch.float32, device=self.device
        )
        self.next_states[self.idx] = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device
        )
        self.dones[self.idx] = torch.as_tensor(
            done, dtype=torch.bool, device=self.device
        )

        # Update circular buffer indices
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        actual_size = min(self.size, batch_size)
        sample_indices = torch.randperm(self.size)[:actual_size].to(self.device)

        batch = Batch(
            states=self.states[sample_indices].to(self.device),
            actions=self.actions[sample_indices].to(self.device),
            rewards=self.rewards[sample_indices].to(self.device),
            next_states=self.next_states[sample_indices].to(self.device),
            dones=self.dones[sample_indices].to(self.device),
        )

        return batch

    def populate(self, env: Env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """
        print("Populating replay memory")
        state, info = env.reset()
        for _ in range(num_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.add(state, action, reward, next_state, done)
            if not done:
                state = next_state
            else:
                state, info = env.reset()
        print("Populating completed")

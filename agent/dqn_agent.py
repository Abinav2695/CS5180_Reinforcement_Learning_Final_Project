import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Env, spaces
from tqdm import tqdm, trange

from .exponential_schedule import ExponentialSchedule
from .replay_memory import Batch, ReplayMemory


class DQN(nn.Module):
    def __init__(
        self, state_dim, action_dim, *, num_layers=3, hidden_dim=256, device="cpu"
    ):
        """Deep Q-Network PyTorch model with device specification.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
            - device: Computational device ('cpu' or 'cuda')
        """
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)  # Specify the device for computation

        # Define the network layers
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Identity())

        # Initialize nn.Sequential model
        self.model = nn.Sequential(*layers)
        self.to(device)  # Move the model to the specified device

    def forward(self, states):
        """Forward pass that handles states on the correct device."""
        return self.model(states)

    @classmethod
    def custom_load(cls, data):
        """Custom load for restoring model."""
        model = cls(*data["args"], **data["kwargs"]).to(data["kwargs"]["device"])
        model.load_state_dict(data["state_dict"])
        return model

    def custom_dump(self):
        """Custom dump for saving model."""
        return {
            "args": (self.state_dim, self.action_dim),
            "kwargs": {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "device": str(self.device),
            },
            "state_dict": self.state_dict(),
        }


def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma, device):
    states, actions, rewards, next_states, dones = batch

    # Ensure all tensors are on the correct device
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Compute the Q-values from the current model for all actions
    q_values = dqn_model(states)

    # Select the Q-value for the action taken in each state
    values = q_values.gather(1, actions)

    # Compute the Q-values from the target model for next states using no_grad
    with torch.no_grad():
        next_q_values = dqn_target(next_states)
        next_q_values_max = next_q_values.max(1, keepdim=True)[0]

    # Compute the target values
    target_values = rewards + gamma * (1 - dones.int()) * next_q_values_max

    # Compute MSE loss
    loss = F.mse_loss(values, target_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_dqn(
    env: Env,
    num_steps,
    *,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    batch_size,
    exploration: ExponentialSchedule,
    gamma,
):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # Initialize the DQN and DQN-target models
    state_size = env.observation_space.shape[1]
    dqn_model = DQN(state_size, env.action_space.n, device=device_name)
    dqn_target = DQN.custom_load(dqn_model.custom_dump())
    dqn_target.load_state_dict(dqn_model.state_dict())

    # Initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters())

    # Initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size, device=device_name)
    memory.populate(env, replay_prepopulate_steps)

    # Initialize lists to store returns, lengths, and losses
    returns = []
    lengths = []
    losses = []
    rewards = []

    # Initialize structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    # Begin training
    state, info = env.reset()
    t_episode = 0
    pbar = trange(num_steps)
    for t_total in pbar:
        # Epsilon for the current step
        eps = exploration.value(t_total)

        # Action selection via epsilon-greedy approach
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = dqn_model(state_tensor)
                action = q_values.argmax().item()

        # Environment step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.add(state, action, reward, next_state, done)

        # Update state
        state = next_state if not done else env.reset()[0]

        rewards.append(reward)
        t_episode += 1

        # Training the network
        if t_total % 4 == 0:
            batch = memory.sample(batch_size)
            # batch = Batch(*[item.to(device) for item in batch])
            loss = train_dqn_batch(
                optimizer, batch, dqn_model, dqn_target, gamma, device
            )
            losses.append(loss)

        # Periodically update the target network
        if t_total % 10_000 == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())

        # Logging progress
        if done:
            G = sum(rewards)
            returns.append(G)
            lengths.append(t_episode)
            rewards = []
            t_episode = 0
            pbar.set_description(
                f"Episode {len(returns)}: Steps : {lengths[-1], }Return={G:.2f}, Epsilon={eps:.2f}, Loss={loss:.4f}"
            )

        # Save models periodically
        if t_total in t_saves:
            model_name = f"{100 * t_total / num_steps:04.1f}".replace(".", "_")
            saved_models[model_name] = copy.deepcopy(dqn_model)

    # Save the final model
    saved_models["final"] = copy.deepcopy(dqn_model)

    return saved_models, np.array(returns), np.array(lengths), np.array(losses)


def _test_dqn_forward(dqn_model, input_shape, output_shape, device):
    """Tests that the dqn returns the correctly shaped tensors."""
    inputs = torch.randn(input_shape).to(device)
    outputs = dqn_model(inputs)

    expected_type = (
        torch.FloatTensor if device == torch.device("cpu") else torch.cuda.FloatTensor
    )

    if not isinstance(outputs, expected_type):
        raise Exception(
            f"DQN.forward returned type {type(outputs)} instead of {expected_type}"
        )

    if outputs.shape != output_shape:
        raise Exception(
            f"DQN.forward returned tensor with shape {outputs.shape} instead of {output_shape}"
        )

    if not outputs.requires_grad:
        raise Exception(
            f"DQN.forward returned tensor which does not require a gradient (but it should)"
        )


if __name__ == "__main__":
    print("Testing ...")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dqn_model = DQN(10, 4, device=device_name)
    print(f"DQN model device: {next(dqn_model.parameters()).device}")
    _test_dqn_forward(dqn_model, (64, 10), (64, 4), device=device)
    _test_dqn_forward(dqn_model, (2, 3, 10), (2, 3, 4), device=device)
    del dqn_model

    dqn_model = DQN(64, 16, device=device_name)
    print(f"DQN model device: {next(dqn_model.parameters()).device}")
    _test_dqn_forward(dqn_model, (64, 64), (64, 16), device=device)
    _test_dqn_forward(dqn_model, (2, 3, 64), (2, 3, 16), device=device)
    del dqn_model

    # Testing custom dump / load
    dqn1 = DQN(10, 4, num_layers=10, hidden_dim=20, device=device_name)
    dqn2 = DQN.custom_load(dqn1.custom_dump())
    print(f"DQN model device: {next(dqn1.parameters()).device}")
    print(f"DQN model device of dqn2: {next(dqn2.parameters()).device}")

    assert dqn2.state_dim == 10
    assert dqn2.action_dim == 4
    assert dqn2.num_layers == 10
    assert dqn2.hidden_dim == 20

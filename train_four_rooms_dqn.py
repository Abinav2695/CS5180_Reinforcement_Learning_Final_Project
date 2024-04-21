import torch
from agent.exponential_schedule import ExponentialSchedule
from agent.dqn_agent import train_dqn, DQN
from envs.four_rooms_env import LargeFourRooms


def main(filepath):
    env = LargeFourRooms(max_steps_per_episode=500)

    # Hyperparameters
    ## Legend Parameters for training
    gamma = 0.99

    num_steps = 500_000
    num_saves = 5  # Save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 200_000
    replay_prepopulate_steps = 50_000

    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.05, 1_000_000)

    # Train the model
    dqn_models, returns, lengths, losses = train_dqn(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )

    # Check if the models have been saved correctly
    assert len(dqn_models) == num_saves
    assert all(isinstance(value, DQN) for value in dqn_models.values())

    # Save models to disk for later analysis or continued training
    checkpoint = {key: model.custom_dump() for key, model in dqn_models.items()}
    torch.save(checkpoint, filepath)

    print("Training completed successfully.")


if __name__ == "__main__":
    main(filepath = "models/train2_dqn.pt")

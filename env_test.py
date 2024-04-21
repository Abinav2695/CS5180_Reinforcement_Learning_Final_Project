from envs.four_rooms_env import LargeFourRooms


def main():
    env = LargeFourRooms(mode="human")
    print(env.observation_space.shape[1])
    state, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        if done:
            break
    env.close()


if __name__ == "__main__":
    main()

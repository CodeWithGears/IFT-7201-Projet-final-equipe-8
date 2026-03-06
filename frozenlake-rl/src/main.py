import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4")

state, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print("State:", state, "Reward:", reward)
    done = terminated or truncated

env.close()
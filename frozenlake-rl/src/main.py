from env import create_frozenlake_env
from maps import map_generator

# S, F, H, or G

MAP = map_generator(16, 0.07, 0.2, 2)

env = create_frozenlake_env(map_desc=MAP, render_mode="human")

state, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print("State:", state, "Reward:", reward)
    done = terminated or truncated

env.close()
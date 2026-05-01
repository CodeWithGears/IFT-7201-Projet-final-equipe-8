"""Simple GUI to interact with Lagrangian QL agent."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from main import cas_etude, make_env
from lagrangian_ql import LagrangianQLearning

# Path
src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(src_path)

stats_path = os.path.join(project_path, "training_stats")
model_path = os.path.join(project_path, "models")

# Configuration
difficulty = "hard"  # Change to "medium" or "hard"
model_path = os.path.join(model_path, f"lagrangian_ql_FrozenLake_{difficulty}.npz")  # Path to the saved model for the chosen difficulty

# Create environment and agent with rgb_array rendering
env = make_env(cas_etude[difficulty], render_mode="rgb_array")
agent = LagrangianQLearning(env, alpha=0.1, gamma=0.99, epsilon=0.0, cost_limit=3.0)

# Load model
if os.path.exists(model_path):
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
else:
    print(f"Model not found at {model_path}")
    exit(1)

# Run episodes and collect all paths
all_episodes = []
for episode in range(20):
    obs, _ = env.reset()
    state = np.argmax(obs) if isinstance(obs, np.ndarray) else int(obs)
    done = False
    total_reward = 0.0
    path = [state]
    frames = []
    
    while not done:
        action = agent.select_action(state, train=False)
        obs_next, reward, terminated, truncated, info = env.step(action)
        state = np.argmax(obs_next) if isinstance(obs_next, np.ndarray) else int(obs_next)
        total_reward += reward
        done = terminated or truncated
        path.append(state)
        frames.append(env.render())
    
    all_episodes.append({'path': path, 'reward': total_reward, 'frames': frames})
    print(f"Episode {episode + 1}: Reward={total_reward:.1f}, Success={total_reward > 0}")

# Superpose all paths on a single grid
grid_size = int(np.sqrt(len(agent.Q)))
cell_size = all_episodes[0]['frames'][0].shape[0] // grid_size

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(all_episodes[0]['frames'][0])

for ep_idx, episode_data in enumerate(all_episodes):
    path = episode_data['path']
    reward = episode_data['reward']
    color = 'blue'  # Same color for all episodes
    
    # Draw path
    for i, state_idx in enumerate(path):
        row, col = divmod(state_idx, grid_size)
        x, y = col * cell_size + cell_size // 2, row * cell_size + cell_size // 2
        
        if i == 0:
            ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.7)
        elif i == len(path) - 1:
            ax.plot(x, y, 'x', color=color, markersize=15, alpha=0.7)
        
        if i > 0:
            prev_row, prev_col = divmod(path[i-1], grid_size)
            prev_x = prev_col * cell_size + cell_size // 2
            prev_y = prev_row * cell_size + cell_size // 2
            ax.plot([prev_x, x], [prev_y, y], color=color, alpha=0.4, linewidth=1.5)

ax.set_title(f"{len(all_episodes)} trajectoires superposées")
ax.axis('off')
plt.tight_layout()
plt.show()

env.close()

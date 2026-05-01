"""Simple GUI to interact with Lagrangian QL agent."""
import os
import numpy as np
from main import cas_etude, make_env
from lagrangian_ql import LagrangianQLearning

# Configuration
difficulty = "easy"  # Change to "medium" or "hard"
model_path = f"lagrangian_ql_FrozenLake_{difficulty}_agent.npz"

# Create environment and agent
env = make_env(cas_etude[difficulty], render_mode="human")
agent = LagrangianQLearning(env, alpha=0.1, gamma=0.99, epsilon=0.0, cost_limit=5.0)

# Load model
if os.path.exists(model_path):
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
else:
    print(f"Model not found at {model_path}")
    exit(1)

# Run episodes
for episode in range(5):
    obs, _ = env.reset()
    state = np.argmax(obs) if isinstance(obs, np.ndarray) else int(obs)
    done = False
    total_reward = 0.0
    
    while not done:
        action = agent.select_action(state, train=False)
        obs_next, reward, terminated, truncated, info = env.step(action)
        state = np.argmax(obs_next) if isinstance(obs_next, np.ndarray) else int(obs_next)
        total_reward += reward
        done = terminated or truncated
        env.render()
    
    print(f"Episode {episode + 1}: Reward={total_reward:.1f}, Success={total_reward > 0}")

env.close()

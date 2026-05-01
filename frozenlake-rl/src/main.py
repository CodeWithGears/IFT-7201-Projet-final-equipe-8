import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from lagrangian_ql import LagrangianQLearning

cas_etude = {
    "easy": {
        "desc": [
            "SFFFFFHFFFFF",
            "FFFFFFFFFFHF",
            "FFFFFFFFFFFF",
            "FFFFFFFFFFFF",
            "FFFFFFFFFFFH",
            "HFFFFFFFFFFF",
            "FFFFFFFFFFFF",
            "FFFFFFFFFHFF",
            "FFFFFFFFFFFF",
            "FFFFFFFFFFFF",
            "FFFFFFFFFFFF",
            "FFHFFFFFFFFG",
        ],
        "is_slippery": True,
        "success_rate": 0.9,
        "reward_schedule": (50, -15, -0.1),
    },
    "medium": {
        "desc": [
            "SFFFFFFFFFFF",
            "FFFFFFFHFHFF",
            "HFFFFHHFFFFF",
            "HFFFFFFFFFFF",
            "FFFFFFFFHFFF",
            "FFFFFFFFHHFF",
            "FFFFFFFFFFFF",
            "FFFFFFFFFFFF",
            "FFHFFFFFFFFH",
            "HHFFFHHFFFFH",
            "HFFFFFFFFFFH",
            "FFFFFFFFFFFG",
        ],
        "is_slippery": True,
        "success_rate": 0.8,
        "reward_schedule": (50, -15, -0.1),
    },
    "hard": {
        "desc": [
            "SFFFFFFFFFFH",
            "FFHHFFFFFFFH",
            "FFFFFFFFFFFF",
            "HFFFFFHHFFFF",
            "HFFFFFFFFHFF",
            "HFFFHFFFFHFF",
            "FFFHFFFFHHFF",
            "FFHFHFFFFFFH",
            "FHFFFFFFFFFF",
            "FFHHFFFFHFHF",
            "FFFFFHFFFFHF",
            "FFFFFFFFFFFG",
        ],
        "is_slippery": True,
        "success_rate": 0.7,
        "reward_schedule": (50, -15, -0.1),
    },
}


class FrozenLakeHoleInfoWrapper(gym.Wrapper):
    """
    Adds `fell_in_hole` to info on terminal steps so the callback can count holes.
    """

    def __init__(self, env):
        super().__init__(env)
        raw_desc = self.unwrapped.desc
        self.desc = np.array([
            [cell.decode("utf-8") if isinstance(cell, bytes) else str(cell) for cell in row]
            for row in raw_desc
        ])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        state = int(self.unwrapped.s)
        n_cols = self.desc.shape[1]
        row, col = divmod(state, n_cols)
        tile = self.desc[row, col]

        info = dict(info)
        info["fell_in_hole"] = bool(terminated and tile == "H")
        return obs, reward, terminated, truncated, info


class SimpleTrainingStatsCallback(BaseCallback):
    """
    Records:
    - average total reward per batch of completed episodes
    - average hole falls per batch of completed episodes

    Saves only one .npz file at the end.
    """

    def __init__(self, save_path, episodes_per_batch=20, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episodes_per_batch = int(episodes_per_batch)

        self.current_rewards = None
        self.batch_rewards = []
        self.batch_holes = []

        self.timesteps = []
        self.avg_rewards = []
        self.avg_holes = []

    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs, dtype=np.float64)

    def _flush_batch(self):
        if not self.batch_rewards:
            return

        self.timesteps.append(int(self.num_timesteps))
        self.avg_rewards.append(float(np.mean(self.batch_rewards)))
        self.avg_holes.append(float(np.mean(self.batch_holes)))

        self.batch_rewards.clear()
        self.batch_holes.clear()

    def _on_step(self):
        rewards = np.array(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.array(self.locals["dones"], dtype=bool).reshape(-1)
        infos = self.locals["infos"]

        self.current_rewards += rewards

        for i, done in enumerate(dones):
            if not done:
                continue

            self.batch_rewards.append(float(self.current_rewards[i]))
            self.batch_holes.append(1.0 if infos[i].get("fell_in_hole", False) else 0.0)
            self.current_rewards[i] = 0.0

            if len(self.batch_rewards) >= self.episodes_per_batch:
                self._flush_batch()

        return True

    def _on_training_end(self):
        self._flush_batch()
        save_training_stats(
            self.save_path,
            np.array(self.timesteps, dtype=np.int64),
            np.array(self.avg_rewards, dtype=np.float64),
            np.array(self.avg_holes, dtype=np.float64),
        )
    

def DQN_training (env, retrain=False, name="dqn_FrozenLake", total_timesteps=100000, seed=6, stats_dir="training_stats", episodes_per_batch=20):
    stats_path = os.path.join(stats_dir, f"{name}.npz")
    if retrain or not os.path.exists(f"{name}.zip"):
        callback = SimpleTrainingStatsCallback(
            save_path=stats_path,
            episodes_per_batch=episodes_per_batch,
        )
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=5e-5,
            buffer_size=100000,
            learning_starts=20000,
            batch_size=128,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
            seed=seed,
        )
        model.learn(total_timesteps=total_timesteps, log_interval=4, callback=callback)
        model.save(name)
    else:
        model = DQN.load(name, env=env)
    return model

def PPO_training(env, retrain=False, name="ppo_FrozenLake", total_timesteps=500000, seed=6, stats_dir="training_stats", episodes_per_batch=20):
    stats_path = os.path.join(stats_dir, f"{name}.npz")
    if retrain or not os.path.exists(f"{name}.zip"):
        callback = SimpleTrainingStatsCallback(
            save_path=stats_path,
            episodes_per_batch=episodes_per_batch,
        )
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.1,
            vf_coef=0.5,
            clip_range=0.1,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
            seed=seed,
        )
        model.learn(total_timesteps=total_timesteps, log_interval=4, callback=callback)
        model.save(name)
    else:
        model = PPO.load(name, env=env)
    return model

def lagrangian_ql_training(env, 
                           retrain=False, 
                           name="lagrangian_ql_FrozenLake", 
                           num_timestamps=100000, 
                           cost_limit=5.0, 
                           stats_dir="training_stats", 
                           model_dir="models",
                           batch_size=20):
    
    stats_path = os.path.join(stats_dir, f"{name}.npz")
    agent_path = os.path.join(model_dir, f"{name}.npz")
    
    if retrain :
        print(f"\n{'='*60}")
        print(f"Starting Lagrangian Q-Learning: {name}")
        print(f"{'='*60}")
        
        agent = LagrangianQLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, cost_limit=cost_limit, lambda_lr=0.01)
        batch, costs, rewards = agent.train(num_timestamps, n_batches=batch_size)
        
        # Save agent weights
        agent.save(agent_path)  # New: save agent
        
        # Save training stats
        os.makedirs(stats_dir, exist_ok=True)
        np.savez(stats_path, batch_index=batch, avg_holes=costs, avg_rewards=rewards)
        
        print(f"Stats saved to: {stats_path}\n")
    
    return agent


def save_training_stats(path, timesteps, avg_rewards, avg_holes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        timesteps=timesteps,
        avg_rewards=avg_rewards,
        avg_holes=avg_holes,
    )


def plot_training_stats(path, output_dir="training_figures", title_prefix=""):
    data = np.load(path)

    if "timesteps" in data:
        x = data["timesteps"]
        xlabel = "Training timesteps"
    elif "batch_index" in data:
        x = data["batch_index"]
        xlabel = "Batch index"
    else:
        raise KeyError(
            f"Neither 'timesteps' nor 'batch_index' found in {path}. "
            f"Available keys: {list(data.keys())}"
        )

    avg_rewards = data["avg_rewards"]
    avg_holes = data["avg_holes"]

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]

    holes_fig_path = os.path.join(output_dir, f"{stem}_avg_holes.png")
    rewards_fig_path = os.path.join(output_dir, f"{stem}_avg_rewards.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, avg_holes)
    plt.xlabel(xlabel)
    plt.ylabel("Average hole falls")
    plt.title(f"{title_prefix} Average hole falls".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(holes_fig_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, avg_rewards)
    plt.xlabel(xlabel)
    plt.ylabel("Average total reward")
    plt.title(f"{title_prefix} Average total reward".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(rewards_fig_path, dpi=150)
    plt.close()

    return holes_fig_path, rewards_fig_path


def display_model(model, env, max_steps=1000):
    time_step = 0
    terminated, truncated = False, False

    state, _ = env.reset()

    while not (terminated or truncated) and time_step < max_steps:
        action, _ = model.predict(state, deterministic=True)
        action = int(action)
        state, reward, terminated, truncated, _ = env.step(action)
        time_step += 1
    env.close()

def make_env(config, render_mode=None):
    env = gym.make(
        "FrozenLake-v1",
        desc=config ["desc"],
        is_slippery=config ["is_slippery"],
        success_rate=config ["success_rate"],
        reward_schedule=config ["reward_schedule"],
        render_mode=render_mode,
    )
    env = FrozenLakeHoleInfoWrapper(env)

    n = env.observation_space.n
    env = TransformObservation(
        env,
        lambda obs: np.eye(n, dtype=np.float32)[int(obs)],
        observation_space=Box(0.0, 1.0, shape=(n,), dtype=np.float32),
    )
    return env

def to_one_hot(state, n_states):
    vec = np.zeros(n_states, dtype=np.float32)
    vec[int(state)] = 1.0
    return vec

def main():
    # Gestion des path
    src_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(src_path)

    stats_path = os.path.join(project_path, "training_stats")
    model_path = os.path.join(project_path, "models")


    #Entrainer/charger les modèles DQN et PPO pour chaque cas d'étude
    DQN_models = {}
    PPO_models = {}
    train_models = True
    total_timesteps_list = [250000, 500000, 1500000]
    episodes_per_batch = 20
    # for i, cas in enumerate(list(cas_etude.keys())):
    #     config = cas_etude[cas]
    #     total_timesteps = total_timesteps_list[i]

    #     print(f"Training DQN for {cas} case...")
    #     env_dqn = make_env(config)
    #     DQN_models[cas] = DQN_training(env_dqn, retrain=train_models, name=f"dqn_FrozenLake_{cas}", total_timesteps=total_timesteps, episodes_per_batch=episodes_per_batch)

    #     print(f"Training PPO for {cas} case...")
    #     env_ppo = make_env(config)
    #     PPO_models[cas] = PPO_training(env_ppo, retrain=train_models, name=f"ppo_FrozenLake_{cas}", total_timesteps=total_timesteps, episodes_per_batch=episodes_per_batch)

    #     dqn_stats = os.path.join("training_stats", f"dqn_FrozenLake_{cas}.npz")
    #     ppo_stats = os.path.join("training_stats", f"ppo_FrozenLake_{cas}.npz")

    #     if os.path.exists(dqn_stats):
    #         plot_training_stats(dqn_stats, title_prefix=f"DQN - {cas}")
    #     if os.path.exists(ppo_stats):
    #         plot_training_stats(ppo_stats, title_prefix=f"PPO - {cas}")
    
    # display_env = make_env(cas_etude["hard"], render_mode="human")
    # display_model(DQN_models["hard"], display_env)
    
    # Train Lagrangian Q-Learning
    print("\nTraining Lagrangian Q-Learning...")

    for i, cas in enumerate(list(cas_etude.keys())):
        config = cas_etude[cas]
        env_lql = make_env(config)
        lagrangian_ql_training(env_lql, 
                               retrain=train_models, 
                               name=f"lagrangian_ql_FrozenLake_{cas}",
                               stats_dir=stats_path,
                               model_dir=model_path,
                               num_timestamps= total_timesteps_list[i], 
                               batch_size=episodes_per_batch,
                               cost_limit=3.0)

if __name__ == "__main__":
    main()
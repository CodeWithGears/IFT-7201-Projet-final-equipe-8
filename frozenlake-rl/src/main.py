import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

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
    

def DQN_training (env, retrain=False, name="dqn_FrozenLake", total_timesteps=100000, seed=6):
    if retrain or not os.path.exists(f"{name}.zip"):
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
            exploration_final_eps=0.1,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
            seed=seed,
        )
        model.learn(total_timesteps=total_timesteps, log_interval=4)
        model.save(name)
    else:
        model = DQN.load(name, env=env)
    return model

def PPO_training(env, retrain=False, name="ppo_FrozenLake", total_timesteps=500000, seed=6):
    if retrain or not os.path.exists(f"{name}.zip"):
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=256,
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
        model.learn(total_timesteps=total_timesteps, log_interval=4)
        model.save(name)
    else:
        model = PPO.load(name, env=env)
    return model


def save_training_stats(path, batch_index, avg_rewards, avg_holes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        batch_index=batch_index,
        avg_rewards=avg_rewards,
        avg_holes=avg_holes,
    )


def plot_training_stats(path, output_dir="training_figures", title_prefix=""):
    data = np.load(path)
    x = data["batch_index"]
    avg_rewards = data["avg_rewards"]
    avg_holes = data["avg_holes"]

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]

    holes_fig_path = os.path.join(output_dir, f"{stem}_avg_holes.png")
    rewards_fig_path = os.path.join(output_dir, f"{stem}_avg_rewards.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, avg_holes, marker="o")
    plt.xlabel("Temps (numéro de la batch)")
    plt.ylabel("Nombre moyen de chutes dans les trous par batch")
    plt.title(f"{title_prefix} Nombre moyen de chutes dans les trous par batch en fonction du temps".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(holes_fig_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, avg_rewards, marker="o")
    plt.xlabel("Temps (numéro de la batch)")
    plt.ylabel("Récompenses totales moyennes par batch")
    plt.title(f"{title_prefix} Récompenses totales moyennes par batch en fonction du temps".strip())
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
    #Entrainer/charger les modèles DQN et PPO pour chaque cas d'étude
    DQN_models = {}
    PPO_models = {}
    train_models = True
    total_timesteps_list = [250000, 500000, 1500000]
    for i, cas in enumerate(list(cas_etude.keys())):
        config = cas_etude[cas]
        total_timesteps = total_timesteps_list[i]

        print(f"Training DQN for {cas} case...")
        env_dqn = make_env(config)
        DQN_models[cas] = DQN_training(env_dqn, retrain=train_models, name=f"dqn_FrozenLake_{cas}", total_timesteps=total_timesteps)

        print(f"Training PPO for {cas} case...")
        env_ppo = make_env(config)
        PPO_models[cas] = PPO_training(env_ppo, retrain=train_models, name=f"ppo_FrozenLake_{cas}", total_timesteps=total_timesteps)
    
    display_env = make_env(cas_etude["hard"], render_mode="human")
    display_model(DQN_models["hard"], display_env)

    #Évaluation des modèles DQN et PPO pour chaque cas d'étude
    for cas in cas_etude.keys():
        print(f"Evaluating DQN for {cas} case...")
        env_dqn = make_env(cas_etude[cas])

        print(f"Evaluating PPO for {cas} case...")
        env_ppo = make_env(cas_etude[cas])

if __name__ == "__main__":
    main()
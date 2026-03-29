import os
import numpy as np
import matplotlib.pyplot as plt

cas_etude = {
    "easy": {
        "name": "facile",
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
        "name": "moyenne",
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
        "name": "difficile",
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

def smooth_curve(y, window=8):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y.copy()

    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if y.size < window:
        window = y.size if y.size % 2 == 1 else max(1, y.size - 1)
    if window <= 1:
        return y.copy()

    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")

def generate_comparison_figures_for_map(
    map_name,
    stats_dir="training_stats",
    output_dir="training_figures",
    smooth_window=9,
):
    dqn_path = os.path.join(stats_dir, f"dqn_FrozenLake_{map_name}.npz")
    ppo_path = os.path.join(stats_dir, f"ppo_FrozenLake_{map_name}.npz")

    if not os.path.exists(dqn_path):
        raise FileNotFoundError(f"DQN stats file not found: {dqn_path}")
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"PPO stats file not found: {ppo_path}")

    dqn_data = np.load(dqn_path)
    ppo_data = np.load(ppo_path)

    dqn_x = dqn_data["timesteps"]
    dqn_rewards = smooth_curve(dqn_data["avg_rewards"], window=smooth_window)
    dqn_holes = smooth_curve(dqn_data["avg_holes"], window=smooth_window)

    ppo_x = ppo_data["timesteps"]
    ppo_rewards = smooth_curve(ppo_data["avg_rewards"], window=smooth_window)
    ppo_holes = smooth_curve(ppo_data["avg_holes"], window=smooth_window)

    os.makedirs(output_dir, exist_ok=True)

    reward_fig_path = os.path.join(output_dir, f"{map_name}_comparaison_recompense.png")
    holes_fig_path = os.path.join(output_dir, f"{map_name}_comparaison_trous.png")

    plt.figure(figsize=(8, 5))
    plt.plot(dqn_x, dqn_rewards, label="DQN")
    plt.plot(ppo_x, ppo_rewards, label="PPO")
    plt.xlabel("Pas de temps d'entraînement")
    plt.ylabel("Récompense totale moyenne par batch")
    plt.ylim(-20, 50)
    plt.title(f"Map {cas_etude[map_name].get('name', map_name)} - Récompense totale moyenne par batch à l'entraînement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_fig_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(dqn_x, dqn_holes, label="DQN")
    plt.plot(ppo_x, ppo_holes, label="PPO")
    plt.xlabel("Pas de temps d'entraînement")
    plt.ylabel("Nombre de fois moyen tombé dans un trou par batch")
    plt.ylim(-0, 1.1)
    plt.title(f"Map {cas_etude[map_name].get('name', map_name)} - Nombre de fois moyen tombé dans un trou par batch à l'entraînement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(holes_fig_path, dpi=150)
    plt.close()

    return reward_fig_path, holes_fig_path


def generate_all_comparison_figures(
    map_names=None,
    stats_dir="training_stats",
    output_dir="training_figures",
    smooth_window=8,
):
    if map_names is None:
        map_names = list(cas_etude.keys())

    outputs = {}
    for map_name in map_names:
        outputs[map_name] = generate_comparison_figures_for_map(
            map_name,
            stats_dir=stats_dir,
            output_dir=output_dir,
            smooth_window=smooth_window,
        )
    return outputs

generate_all_comparison_figures()
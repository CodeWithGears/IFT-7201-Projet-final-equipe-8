import os
import numpy as np
import matplotlib.pyplot as plt

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

def generate_comparison_figures_for_map(
    map_name,
    stats_dir="training_stats",
    output_dir="training_figures",
):
    dqn_path = os.path.join(stats_dir, f"dqn_FrozenLake_{map_name}.npz")
    ppo_path = os.path.join(stats_dir, f"ppo_FrozenLake_{map_name}.npz")

    if not os.path.exists(dqn_path):
        raise FileNotFoundError(f"DQN stats file not found: {dqn_path}")
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"PPO stats file not found: {ppo_path}")

    dqn_data = np.load(dqn_path)
    ppo_data = np.load(ppo_path)

    dqn_x = dqn_data["batch_index"]
    dqn_rewards = dqn_data["avg_rewards"]
    dqn_holes = dqn_data["avg_holes"]

    ppo_x = ppo_data["batch_index"]
    ppo_rewards = ppo_data["avg_rewards"]
    ppo_holes = ppo_data["avg_holes"]

    os.makedirs(output_dir, exist_ok=True)

    reward_fig_path = os.path.join(output_dir, f"{map_name}_reward_comparison.png")
    holes_fig_path = os.path.join(output_dir, f"{map_name}_holes_comparison.png")

    plt.figure(figsize=(8, 5))
    plt.plot(dqn_x, dqn_rewards, label="DQN")
    plt.plot(ppo_x, ppo_rewards, label="PPO")
    plt.xlabel("Batch index")
    plt.ylabel("Average total reward per batch")
    plt.title(f"{map_name.capitalize()} map - reward during training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_fig_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(dqn_x, dqn_holes, label="DQN")
    plt.plot(ppo_x, ppo_holes, label="PPO")
    plt.xlabel("Batch index")
    plt.ylabel("Average hole falls per batch")
    plt.title(f"{map_name.capitalize()} map - hole falls during training")
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
):
    if map_names is None:
        map_names = list(cas_etude.keys())

    outputs = {}
    for map_name in map_names:
        outputs[map_name] = generate_comparison_figures_for_map(
            map_name,
            stats_dir=stats_dir,
            output_dir=output_dir,
        )
    return outputs

generate_all_comparison_figures()
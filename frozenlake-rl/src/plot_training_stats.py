import os
from os import path
from turtle import color
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from main import cas_etude, make_env

DQN_COLOR = "#1f77b4"
PPO_COLOR = "#ff7f0e"
LAGRANGE_COLOR = "#2ca02c"

translate = {"easy": "Facile", "medium": "Médium", "hard": "Difficile"}

indexes = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]

def custom_plot_training_stats(axs, 
                               i,
                               cas,
                               dqn_path, 
                               ppo_path,
                               lagrange_path):  

    dqn = np.load(dqn_path)
    ppo = np.load(ppo_path)
    lagrange = np.load(lagrange_path)

    x_lagrange = lagrange["batch_index"]

    # For DQN
    if "timesteps" in dqn:
        x_dqn = dqn["timesteps"]
        xlabel = "Nombre de pas de temps total"
    elif "batch_index" in dqn:
        x_dqn = dqn["batch_index"]
        xlabel = "Batch index"
    else:
        raise KeyError(
            f"Neither 'timesteps' nor 'batch_index' found in {dqn_path}. "
            f"Available keys: {list(dqn.keys())}"
        )
    
    # For PPO
    if "timesteps" in ppo:
        x_ppo = ppo["timesteps"]
        xlabel = "Nombre de pas de temps total"
    elif "batch_index" in ppo:
        x_ppo = ppo["batch_index"]
        xlabel = "Batch index"
    else:
        raise KeyError(
            f"Neither 'timesteps' nor 'batch_index' found in {ppo_path}. "
            f"Available keys: {list(ppo.keys())}"
        )

    dqn_avg_rewards = dqn["avg_rewards"]
    dqn_avg_holes = dqn["avg_holes"]
    
    ppo_avg_rewards = ppo["avg_rewards"]
    ppo_avg_holes = ppo["avg_holes"]

    lagrange_avg_rewards = lagrange["avg_rewards"]
    lagrange_avg_holes = lagrange["avg_holes"]
    

    # Row 1 (DQN vs PPO)
    axs[0, i].plot(x_dqn, dqn_avg_rewards, label="DQN", color = DQN_COLOR)
    axs[0, i].plot(x_ppo, ppo_avg_rewards, label="PPO", color = PPO_COLOR)
    axs[0, i].plot(x_lagrange, lagrange_avg_rewards, label="Lagrangian QL", color = LAGRANGE_COLOR, alpha=0.7)
    axs[0, i].set_title(indexes[0][i] + f" Récompenses\n({translate[cas]})", fontweight='bold')

    # Row 2 (DQN vs PPO)
    axs[1, i].plot(x_dqn, dqn_avg_holes, label="DQN", color = DQN_COLOR)
    axs[1, i].plot(x_ppo, ppo_avg_holes, label="PPO", color = PPO_COLOR)
    axs[1, i].plot(x_lagrange, lagrange_avg_holes, label="Lagrangian QL", color = LAGRANGE_COLOR, alpha=0.7)
    axs[1, i].set_title(indexes[1][i] + f" Chutes dans trous\n({translate[cas]})", fontweight='bold')

    axs[1, i].set_ylim(0, 1) 


    # General formatting for all axs
    for row in range(2):
        for col in range(3):
            axs[row, col].grid(True, linestyle='--', alpha=0.5)
            if row == 1 and col == 1:
                axs[row, col].set_xlabel(xlabel, labelpad=15)

    for ax in axs.flat:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


if __name__ == "__main__":

    fig, axs = plt.subplots(2, 3, figsize=(6, 6), sharex=False, sharey=False)

    src_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(src_path)

    fig_path = os.path.join(project_path, "training_figures", f"training_comparison_basic_strategies.pdf")

    for i, cas in enumerate(list(cas_etude.keys())):

        dqn_stats = os.path.join(project_path, "training_stats", f"dqn_FrozenLake_{cas}.npz")
        ppo_stats = os.path.join(project_path, "training_stats", f"ppo_FrozenLake_{cas}.npz")
        lagrange_stats = os.path.join(project_path, "training_stats", f"lagrangian_ql_FrozenLake_{cas}.npz")
        
        if os.path.exists(dqn_stats) and os.path.exists(ppo_stats) and os.path.exists(lagrange_stats):
            with plt.rc_context({'font.size': 8, 'lines.linewidth': 0.5}): 
                custom_plot_training_stats(axs, i, cas, dqn_stats, ppo_stats, lagrange_stats)

    plt.tight_layout()
    
    # Add a big legend at the bottom of the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=10, frameon=True)
    
    fig.savefig(fig_path, bbox_inches='tight', dpi=500)
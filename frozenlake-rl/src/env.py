import gymnasium as gym
from main import cas_etude, make_env
import matplotlib.pyplot as plt


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 3, figsize=(6, 4))
    for idx, cas in enumerate(cas_etude.keys()):
        env_easy = make_env(cas_etude[cas], render_mode="rgb_array")
        env_easy.reset()
        frame = env_easy.render()
        axs[idx].imshow(frame)
        axs[idx].set_axis_off()
        axs[idx].set_title(f"{cas}", fontsize=12)

    axs[0].set_title("a) Facile", fontsize=8, fontweight='bold')
    axs[1].set_title("b) Médium", fontsize=8, fontweight='bold')
    axs[2].set_title("c) Difficile", fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig("frozen_lake_renders.pdf", bbox_inches='tight', dpi=400)
    plt.show()







# ==== OLD FUCNTION UNUSED ====

def create_frozenlake_env(
    map_desc=None,
    map_size=4,
    is_slippery=True,
    render_mode=None
):
    """
    Create a customizable FrozenLake environment.

    Parameters:
    - map_desc (list of str or None): Custom map as a list of strings, each string is a row.
        - 'S': Start, 'F': Frozen, 'H': Hole, 'G': Goal.
        - If None, uses a built-in map of size map_size.
    - map_size (int): Size of the built-in map to use if map_desc is None.
        - Supported: 4 or 8 (for "4x4" or "8x8" built-in maps).
    - is_slippery (bool): If True, the ice is slippery (stochastic transitions).
    - render_mode (str or None): Set to "human" to render the environment visually.

    Returns:
    - env: The Gymnasium FrozenLake environment instance.
    """
    if map_desc is not None:
        env = gym.make(
            "FrozenLake-v1",
            desc=map_desc,
            is_slippery=is_slippery,
            render_mode=render_mode
        )
    else:
        if map_size not in [4, 8]:
            raise ValueError("Only built-in map sizes 4 or 8 are supported if map_desc is None.")
        env = gym.make(
            "FrozenLake-v1",
            map_name=f"{map_size}x{map_size}",
            is_slippery=is_slippery,
            render_mode=render_mode
        )
    return env


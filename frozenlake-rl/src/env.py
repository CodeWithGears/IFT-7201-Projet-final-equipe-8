import gymnasium as gym

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


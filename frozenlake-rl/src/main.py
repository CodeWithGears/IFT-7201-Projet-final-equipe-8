import gymnasium as gym

cas_etude = {
    "easy": {
        "desc": [
            "SFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFG",
        ],
        "is_slippery": True,
        "success_rate": 0.9,
        "reward_schedule": (1, 0, 0),
    },
    "medium": {
        "desc": [
            "SFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFG",
        ],
        "is_slippery": True,
        "success_rate": 0.7,
        "reward_schedule": (1, 0, 0),
    },
    "hard": {
        "desc": [
            "SFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFG",
        ],
        "is_slippery": True,
        "success_rate": 0.5,
        "reward_schedule": (1, 0, 0),
    },
}

config = cas_etude["medium"]

env = gym.make(
    "FrozenLake-v1",
    desc=config ["desc"],
    is_slippery=config ["is_slippery"],
    success_rate=config ["success_rate"],
    reward_schedule=config ["reward_schedule"],
)

state, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print("State:", state, "Reward:", reward)
    done = terminated or truncated

env.close()
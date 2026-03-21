import os
import gymnasium as gym
from stable_baselines3 import DQN, PPO

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

def DQN_training (env, retrain=False, name="dqn_FrozenLake"):
    if retrain or not os.path.exists(f"{name}.zip"):
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=4)
        model.save(name)
    else:
        model = DQN.load(name, env=env)
    return model

def PPO_training (env, retrain=False, name="ppo_FrozenLake"):
    if retrain or not os.path.exists(f"{name}.zip"):
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=4)
        model.save(name)
    else:
        model = PPO.load(name, env=env)
    return model

def main():
    #Entrainer/charger les modèles DQN et PPO pour chaque cas d'étude
    DQN_models = {}
    PPO_models = {}
    train_models = False
    for cas in cas_etude.keys():
        config = cas_etude[cas]

        env = gym.make(
            "FrozenLake-v1",
            desc=config ["desc"],
            is_slippery=config ["is_slippery"],
            success_rate=config ["success_rate"],
            reward_schedule=config ["reward_schedule"],
            render_mode="human",
        )
        print(f"Training DQN for {cas} case...")
        DQN_models[cas] = DQN_training(env, retrain=train_models, name=f"dqn_FrozenLake_{cas}")

        print(f"Training PPO for {cas} case...")
        PPO_models[cas] = PPO_training(env, retrain=train_models, name=f"ppo_FrozenLake_{cas}")

    #Évaluation des modèles DQN et PPO pour chaque cas d'étude

    state, _ = env.reset()

    terminated, truncated = False, False

    while not (terminated or truncated):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        print("State:", state, "Reward:", reward)
        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()
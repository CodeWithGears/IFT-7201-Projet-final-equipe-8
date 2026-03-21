import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
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
        "reward_schedule": (20, -20, -0.1),
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
        "reward_schedule": (10, -20, -1),
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
        "reward_schedule": (10, -20, -1),
    },
}

def DQN_training (env, retrain=False, name="dqn_FrozenLake", total_timesteps=10000):
    if retrain or not os.path.exists(f"{name}.zip"):
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=5000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            verbose=1,
        )
        model.learn(total_timesteps=total_timesteps, log_interval=4)
        model.save(name)
    else:
        model = DQN.load(name, env=env)
    return model

def PPO_training (env, retrain=False, name="ppo_FrozenLake", total_timesteps=10000):
    if retrain or not os.path.exists(f"{name}.zip"):
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps, log_interval=4)
        model.save(name)
    else:
        model = PPO.load(name, env=env)
    return model

def evaluate_model(model, env, num_episodes=1):
    rewards = []
    for episode in range(num_episodes):
        terminated, truncated = False, False
        total_reward = 0

        state, _ = env.reset()

        while not (terminated or truncated):
            action, _ = model.predict(state, deterministic=True)
            action = int(action)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return sum(rewards) / num_episodes

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
    return env

def main():
    #Entrainer/charger les modèles DQN et PPO pour chaque cas d'étude
    DQN_models = {}
    PPO_models = {}
    train_models = False
    total_timesteps = 200000
    for cas in cas_etude.keys():
        config = cas_etude[cas]

        print(f"Training DQN for {cas} case...")
        env_dqn = make_env(config)
        DQN_models[cas] = DQN_training(env_dqn, retrain=train_models, name=f"dqn_FrozenLake_{cas}", total_timesteps=total_timesteps)

        print(f"Training PPO for {cas} case...")
        env_ppo = make_env(config)
        PPO_models[cas] = PPO_training(env_ppo, retrain=train_models, name=f"ppo_FrozenLake_{cas}", total_timesteps=total_timesteps)
    
    display_env = make_env(cas_etude["medium"], render_mode="human")
    display_model(DQN_models["easy"], display_env)

    #Évaluation des modèles DQN et PPO pour chaque cas d'étude
    for cas in cas_etude.keys():
        print(f"Evaluating DQN for {cas} case...")
        env_dqn = make_env(cas_etude[cas])
        dqn_reward = evaluate_model(DQN_models[cas], env_dqn, num_episodes=1)
        print(f"DQN average reward for {cas} case: {dqn_reward}")

        print(f"Evaluating PPO for {cas} case...")
        env_ppo = make_env(cas_etude[cas])
        ppo_reward = evaluate_model(PPO_models[cas], env_ppo, num_episodes=1)
        print(f"PPO average reward for {cas} case: {ppo_reward}")

if __name__ == "__main__":
    main()
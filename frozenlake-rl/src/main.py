import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO

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
        "reward_schedule": (100, -15, -0.1),
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
        "reward_schedule": (100, -15, -0.1),
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
        "reward_schedule": (100, -15, -0.1),
    },
}

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
            exploration_fraction=0.7,
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
    train_models = True
    total_timesteps_list = [250000, 250000, 1000000]
    for i, cas in enumerate(list(cas_etude.keys())[2:]):
        config = cas_etude[cas]
        total_timesteps = 1000000 #total_timesteps_list[i]

        print(f"Training DQN for {cas} case...")
        env_dqn = make_env(config)
        #DQN_models[cas] = DQN_training(env_dqn, retrain=train_models, name=f"dqn_FrozenLake_{cas}", total_timesteps=total_timesteps)

        print(f"Training PPO for {cas} case...")
        env_ppo = make_env(config)
        PPO_models[cas] = PPO_training(env_ppo, retrain=train_models, name=f"ppo_FrozenLake_{cas}", total_timesteps=total_timesteps)
    
    display_env = make_env(cas_etude["hard"], render_mode="human")
    display_model(DQN_models["hard"], display_env)

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
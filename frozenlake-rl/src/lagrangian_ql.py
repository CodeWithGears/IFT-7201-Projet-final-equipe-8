"""Lagrangian Q-Learning: Core algorithm only."""
import numpy as np


class LagrangianQLearning:
    """Lagrangian Primal-Dual Q-Learning."""
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, cost_limit=5.0, lambda_lr=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        
        unwrapped = env.unwrapped
        self.n_states = unwrapped.observation_space.n
        self.n_actions = unwrapped.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.lam = 0.0
    
    def _get_state(self, obs):
        """Extract state from observation (one-hot or int)."""
        return np.argmax(obs) if isinstance(obs, np.ndarray) and len(obs.shape) == 1 else int(obs)
    
    def select_action(self, state, train=True):
        """Epsilon-greedy on Lagrangian Q(s,a) - λ·C(s,a)."""
        if train and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        a = np.argmax([self.Q[state, i] for i in range(self.n_actions)])
        return a
    
    def train(self, num_timestamps, n_batches=20):
        """Train: Q-learning primal + λ dual update."""
        costs, rewards = [], []
        batch_timestamps, batch_costs, batch_rewards = [], [], []
        timestamp = 0
        episode = 0
        
        while timestamp < num_timestamps:
            obs, _ = self.env.reset()
            s = self._get_state(obs)
            cost, r_total = 0.0, 0.0
            done = False
            
            while not done:
                # Select action using select_action method
                a = self.select_action(s, train=True)
                
                obs_next, r, terminated, truncated, info = self.env.step(a)
                s_next = self._get_state(obs_next)
                
                # Q-learning update (primal)
                max_q = np.max(self.Q[s_next]) if not terminated else 0.0
                self.Q[s, a] += self.alpha * (r + self.gamma * max_q - self.Q[s, a])
                
                # Track metrics
                cost += float(info.get("fell_in_hole", False))
                r_total += r
                
                done = terminated or truncated
                s = s_next

                timestamp += 1
            
            episode += 1
            
            # Lagrange multiplier update (dual)
            self.lam = max(0.0, self.lam + self.lambda_lr * (cost - self.cost_limit))
            
            costs.append(cost)
            rewards.append(r_total)

            if episode % n_batches == 0:
                batch_costs.append(np.mean(costs[-n_batches:]))
                batch_rewards.append(np.mean(rewards[-n_batches:]))
                batch_timestamps.append(timestamp)

                costs, rewards = [], []
            
            if timestamp % 10000 == 0:
                print(f"  Episode {episode:4d}| Timestamp : {timestamp} | Holes: {np.mean(costs[-50:]):.2f} | Reward: {np.mean(rewards[-50:]):.2f} | λ: {self.lam:.4f}")
        
        return np.array(batch_timestamps), np.array(batch_costs), np.array(batch_rewards)

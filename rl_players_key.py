import numpy as np
import random

class SARSAAgent:
    """
    SARSA agent for the bank game.
    Uses constant learning rate (eta) and dictionary-based Q-table.
    """
    def __init__(self, env, eta=0.3, gamma=0.99, epsilon=0.7, epsilon_decay=0.9998, epsilon_min=0.1):
        self.env = env
        self.eta = eta  # Constant learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
        self.q_values = {}  # Dictionary: {(state, action): q_value}
        self.rewards_per_episode = []
    
    def _get_q_value(self, state, action):
        """Get Q-value for state-action pair, defaulting to 0 if not seen."""
        key = (state, action)
        if key not in self.q_values:
            self.q_values[key] = 0.0
        return self.q_values[key]
    
    def _set_q_value(self, state, action, value):
        """Set Q-value for state-action pair."""
        key = (state, action)
        self.q_values[key] = value
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state (score after 3 rolls)
        
        Returns:
            action: Chosen threshold action (0-500)
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.env.get_actions())
        else:
            # Exploit: choose best action
            actions = self.env.get_actions()
            best_action = None
            best_q = float('-inf')
            
            for action in actions:
                q_val = self._get_q_value(state, action)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
            
            # If no action has been seen, choose randomly
            if best_action is None:
                return np.random.choice(actions)
            
            # If multiple actions have same Q-value, choose randomly among them
            best_actions = [a for a in actions if self._get_q_value(state, a) == best_q]
            return np.random.choice(best_actions)
    
    def g(self, state, reward, done):
        """
        Reward function.
        For bank game, reward is given at the end of each round or game.
        """
        return reward
    
    def train(self, num_episodes):
        """
        Train the SARSA agent.
        
        Args:
            num_episodes: Number of episodes to train
        
        Returns:
            q_values: Dictionary of Q-values
            rewards_per_episode: List of total rewards per episode
        """
        rewards_per_episode = np.zeros(num_episodes)
        
        for episode in range(num_episodes):
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Reset environment and get initial state
            x_k = self.env.reset()
            total_reward = 0
            
            # Choose initial action
            u_k = self.choose_action(x_k)
            
            # Play until game is done
            done = False
            while not done:
                # Take step in environment
                xk_1, reward, done = self.env.step(u_k)
                total_reward += reward
                
                # Choose action for next state (if not done)
                if not done:
                    u_k_1 = self.choose_action(xk_1)
                else:
                    # Terminal state: next action doesn't matter, use 0
                    u_k_1 = 0
                
                # Calculate TD error
                current_q = self._get_q_value(x_k, u_k)
                if done:
                    # Terminal state: Q(s', a') = 0
                    next_q = 0.0
                else:
                    next_q = self._get_q_value(xk_1, u_k_1)
                
                # Reward for current state
                gk = self.g(x_k, reward, done)
                
                # TD error
                dk = gk + self.gamma * next_q - current_q
                
                # Update Q-value
                new_q = current_q + self.eta * dk
                self._set_q_value(x_k, u_k, new_q)
                
                # Move to next state and action
                x_k = xk_1
                u_k = u_k_1
            
            # Record total reward for episode
            rewards_per_episode[episode] = total_reward
        
        self.rewards_per_episode = rewards_per_episode
        return self.q_values, self.rewards_per_episode


class QLearningAgent:
    """
    Q-learning agent for the bank game.
    Uses constant learning rate (eta) and dictionary-based Q-table.
    """
    def __init__(self, env, eta=0.3, gamma=0.99, epsilon=0.7, epsilon_decay=0.9998, epsilon_min=0.1):
        self.env = env
        self.eta = eta  # Constant learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
        self.q_values = {}  # Dictionary: {(state, action): q_value}
        self.rewards_per_episode = []
    
    def _get_q_value(self, state, action):
        """Get Q-value for state-action pair, defaulting to 0 if not seen."""
        key = (state, action)
        if key not in self.q_values:
            self.q_values[key] = 0.0
        return self.q_values[key]
    
    def _set_q_value(self, state, action, value):
        """Set Q-value for state-action pair."""
        key = (state, action)
        self.q_values[key] = value
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state (score after 3 rolls)
        
        Returns:
            action: Chosen threshold action (0-500)
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.env.get_actions())
        else:
            # Exploit: choose best action
            actions = self.env.get_actions()
            best_action = None
            best_q = float('-inf')
            
            for action in actions:
                q_val = self._get_q_value(state, action)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
            
            # If no action has been seen, choose randomly
            if best_action is None:
                return np.random.choice(actions)
            
            # If multiple actions have same Q-value, choose randomly among them
            best_actions = [a for a in actions if self._get_q_value(state, a) == best_q]
            return np.random.choice(best_actions)
    
    def g(self, state, reward, done):
        """
        Reward function.
        For bank game, reward is given at the end of each round or game.
        """
        return reward
    
    def train(self, num_episodes):
        """
        Train the Q-learning agent.
        
        Args:
            num_episodes: Number of episodes to train
        
        Returns:
            q_values: Dictionary of Q-values
            rewards_per_episode: List of total rewards per episode
        """
        rewards_per_episode = np.zeros(num_episodes)
        
        for episode in range(num_episodes):
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Reset environment and get initial state
            x_k = self.env.reset()
            total_reward = 0
            
            # Play until game is done
            done = False
            while not done:
                # Choose action using epsilon-greedy
                u_k = self.choose_action(x_k)
                
                # Take step in environment
                xk_1, reward, done = self.env.step(u_k)
                total_reward += reward
                
                # Calculate TD error (Q-learning uses max over next actions)
                current_q = self._get_q_value(x_k, u_k)
                
                if done:
                    # Terminal state: max Q(s', a') = 0
                    max_next_q = 0.0
                else:
                    # Find max Q-value over all actions in next state
                    actions = self.env.get_actions()
                    max_next_q = float('-inf')
                    for action in actions:
                        q_val = self._get_q_value(xk_1, action)
                        if q_val > max_next_q:
                            max_next_q = q_val
                    if max_next_q == float('-inf'):
                        max_next_q = 0.0
                
                # Reward for current state
                gk = self.g(x_k, reward, done)
                
                # TD error
                dk = gk + self.gamma * max_next_q - current_q
                
                # Update Q-value
                new_q = current_q + self.eta * dk
                self._set_q_value(x_k, u_k, new_q)
                
                # Move to next state
                x_k = xk_1
            
            # Record total reward for episode
            rewards_per_episode[episode] = total_reward
        
        self.rewards_per_episode = rewards_per_episode
        return self.q_values, self.rewards_per_episode


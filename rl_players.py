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

    ###############################################
    # IMPLEMENT ALGORITHM IN THE TRAIN FUNCTION #
    ###############################################
    def train(self, num_episodes):
        """
        Train the SARSA agent.
        
        Args:
            num_episodes: Number of episodes to train
        
        Returns:
            q_values: Dictionary of Q-values
            rewards_per_episode: List of total rewards per episode
        """
        
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
    
    ###############################################
    # IMPLEMENT ALGORITHM IN THE TRAIN FUNCTION #
    ###############################################
    def train(self, num_episodes):
        """
        Train the Q-learning agent.
        
        Args:
            num_episodes: Number of episodes to train
        
        Returns:
            q_values: Dictionary of Q-values
            rewards_per_episode: List of total rewards per episode
        """
    
        return self.q_values, self.rewards_per_episode


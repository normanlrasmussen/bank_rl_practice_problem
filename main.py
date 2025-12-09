import numpy as np
import matplotlib.pyplot as plt
import argparse
from bank_environment import BankEnvironment
from rl_players import SARSAAgent, QLearningAgent
from players import ProbabilisticPlayer

def train_agent(agent, env, num_episodes):
    """
    Train an agent and track rewards.
    
    Returns:
        rewards_per_episode: Array of total rewards per episode
    """
    rewards_per_episode = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        # Decay epsilon for the agent
        if hasattr(agent, 'epsilon_decay'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Reset environment and get initial state
        x_k = env.reset()
        total_reward = 0
        done = False
        
        if isinstance(agent, SARSAAgent):
            # SARSA: choose initial action
            u_k = agent.choose_action(x_k)
            while not done:
                # Take step in environment
                xk_1, reward, done = env.step(u_k)
                total_reward += reward
                
                # Choose action for next state (if not done)
                if not done:
                    u_k_1 = agent.choose_action(xk_1)
                else:
                    u_k_1 = 0
                
                # Calculate TD error
                current_q = agent._get_q_value(x_k, u_k)
                if done:
                    next_q = 0.0
                else:
                    next_q = agent._get_q_value(xk_1, u_k_1)
                
                gk = agent.g(x_k, reward, done)
                dk = gk + agent.gamma * next_q - current_q
                
                # Update Q-value
                new_q = current_q + agent.eta * dk
                agent._set_q_value(x_k, u_k, new_q)
                
                x_k = xk_1
                u_k = u_k_1
        else:
            # Q-learning
            while not done:
                u_k = agent.choose_action(x_k)
                xk_1, reward, done = env.step(u_k)
                total_reward += reward
                
                current_q = agent._get_q_value(x_k, u_k)
                if done:
                    max_next_q = 0.0
                else:
                    actions = env.get_actions()
                    max_next_q = float('-inf')
                    for action in actions:
                        q_val = agent._get_q_value(xk_1, action)
                        if q_val > max_next_q:
                            max_next_q = q_val
                    if max_next_q == float('-inf'):
                        max_next_q = 0.0
                
                gk = agent.g(x_k, reward, done)
                dk = gk + agent.gamma * max_next_q - current_q
                
                new_q = current_q + agent.eta * dk
                agent._set_q_value(x_k, u_k, new_q)
                
                x_k = xk_1
        
        # Record reward
        rewards_per_episode[episode] = total_reward
    
    return rewards_per_episode

def evaluate_agent(agent, env, num_games=1000):
    """
    Evaluate an agent by playing games and tracking win rate.
    Uses greedy policy (no exploration).
    
    Returns:
        win_rate: Win percentage (0-1)
    """
    # Save original epsilon
    original_epsilon = agent.epsilon
    
    # Set epsilon to 0 for greedy policy
    agent.epsilon = 0.0
    
    wins = 0
    
    for game in range(num_games):
        # Reset environment and get initial state
        x_k = env.reset()
        done = False
        
        if isinstance(agent, SARSAAgent):
            # SARSA: choose initial action
            u_k = agent.choose_action(x_k)
            while not done:
                # Take step in environment
                xk_1, reward, done = env.step(u_k)
                
                # Choose action for next state (if not done)
                if not done:
                    u_k_1 = agent.choose_action(xk_1)
                else:
                    u_k_1 = 0
                
                x_k = xk_1
                u_k = u_k_1
        else:
            # Q-learning
            while not done:
                u_k = agent.choose_action(x_k)
                xk_1, reward, done = env.step(u_k)
                x_k = xk_1
        
        # Check if agent won
        agent_score, opponent_score = env.get_final_scores()
        if opponent_score is None:
            wins += 1
        else:
            if agent_score > opponent_score:
                wins += 1
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return wins / num_games

def main(training_steps=None):
    parser = argparse.ArgumentParser(description='Train and compare SARSA and Q-learning on bank game')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of rounds per game (default: 10)')
    default_episodes = training_steps if training_steps is not None else 500
    parser.add_argument('--episodes', type=int, default=default_episodes,
                        help=f'Number of training episodes (default: {default_episodes})')
    parser.add_argument('--eta', type=float, default=0.3,
                        help='Learning rate (default: 0.3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial epsilon for epsilon-greedy (default: 1.0)')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995,
                        dest='epsilon_decay',
                        help='Epsilon decay rate per episode (default: 0.9995)')
    parser.add_argument('--epsilon-min', type=float, default=0.1,
                        dest='epsilon_min',
                        help='Minimum epsilon value (default: 0.1)')
    
    args = parser.parse_args()
    
    # Always use probability 0.3 opponent
    opponent = ProbabilisticPlayer(probability=0.3)
    reward_types = ['sparse', 'relative', 'score', 'custom']
    
    print(f"Training against probability 0.3 opponent")
    print(f"Rounds per game: {args.rounds}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate (eta): {args.eta}")
    print(f"Discount factor (gamma): {args.gamma}")
    print(f"Epsilon: {args.epsilon} (decay: {args.epsilon_decay}, min: {args.epsilon_min})")
    print()
    
    # Store results for each reward type
    results = {}
    
    # Train for each reward type
    for reward_type in reward_types:
        print(f"Training with reward type: {reward_type}")
        
        # Create environments
        env_sarsa = BankEnvironment(
            rounds=args.rounds,
            opponent=opponent,
            reward_type=reward_type
        )
        env_qlearning = BankEnvironment(
            rounds=args.rounds,
            opponent=opponent,
            reward_type=reward_type
        )
        
        # Train SARSA
        print(f"  Training SARSA...")
        sarsa_agent = SARSAAgent(
            env=env_sarsa,
            eta=args.eta,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min
        )
        sarsa_rewards = train_agent(sarsa_agent, env_sarsa, args.episodes)
        
        # Train Q-learning
        print(f"  Training Q-learning...")
        qlearning_agent = QLearningAgent(
            env=env_qlearning,
            eta=args.eta,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min
        )
        qlearning_rewards = train_agent(qlearning_agent, env_qlearning, args.episodes)
        
        # Evaluate agents against opponent (1000 games each)
        print(f"  Evaluating SARSA against opponent (1000 games)...")
        eval_env_sarsa = BankEnvironment(
            rounds=args.rounds,
            opponent=opponent,
            reward_type=reward_type
        )
        sarsa_win_rate = evaluate_agent(sarsa_agent, eval_env_sarsa, num_games=1000)
        
        print(f"  Evaluating Q-learning against opponent (1000 games)...")
        eval_env_qlearning = BankEnvironment(
            rounds=args.rounds,
            opponent=opponent,
            reward_type=reward_type
        )
        qlearning_win_rate = evaluate_agent(qlearning_agent, eval_env_qlearning, num_games=1000)
        
        # Store results
        results[reward_type] = {
            'sarsa_rewards': sarsa_rewards,
            'sarsa_win_rate': sarsa_win_rate,
            'qlearning_rewards': qlearning_rewards,
            'qlearning_win_rate': qlearning_win_rate
        }
        
        print(f"  SARSA - Mean reward: {np.mean(sarsa_rewards):.3f}, Win rate: {sarsa_win_rate:.3f}")
        print(f"  Q-learning - Mean reward: {np.mean(qlearning_rewards):.3f}, Win rate: {qlearning_win_rate:.3f}")
        print()
    
    # Create 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.suptitle('SARSA vs Q-Learning: Training against Probability 0.3 Opponent', fontsize=16, y=0.98)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.06, wspace=0.3, hspace=0.4)
    
    # Column titles
    col_titles = ['SARSA Reward', 'Q-Learning Reward', 'Win Rate (1000 Games)']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')
    
    # Plot for each reward type
    for row, reward_type in enumerate(reward_types):
        # Row label (add text on the left side, positioned to avoid y-axis labels)
        # Calculate y position: bottom is 0.06, top is 0.94, 4 rows
        y_pos = 0.94 - (row + 0.5) * (0.94 - 0.06) / 4
        fig.text(0.005, y_pos, reward_type.capitalize(), 
                fontsize=11, fontweight='bold', rotation=90, 
                ha='center', va='center')
        
        data = results[reward_type]
        episodes = np.arange(1, args.episodes + 1)
        
        # Calculate shared y-limits for both plots
        all_rewards = np.concatenate([data['sarsa_rewards'], data['qlearning_rewards']])
        y_min = np.min(all_rewards)
        y_max = np.max(all_rewards)
        # Add a small padding (5% of range)
        y_range = y_max - y_min
        if y_range > 0:
            y_padding = y_range * 0.05
            y_min_lim = y_min - y_padding
            y_max_lim = y_max + y_padding
        else:
            y_min_lim = y_min - 1
            y_max_lim = y_max + 1
        
        # SARSA Reward
        axes[row, 0].plot(episodes, data['sarsa_rewards'], alpha=0.7, color='blue', linewidth=0.8, label='SARSA')
        axes[row, 0].set_ylim([y_min_lim, y_max_lim])
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_ylabel('Reward', fontsize=9)
        
        # Q-Learning Reward
        axes[row, 1].plot(episodes, data['qlearning_rewards'], alpha=0.7, color='red', linewidth=0.8, label='Q-Learning')
        axes[row, 1].set_ylim([y_min_lim, y_max_lim])
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_ylabel('Reward', fontsize=9)
        
        # Win Rate Bar Chart
        bars = axes[row, 2].bar(['SARSA', 'Q-Learning'], 
                               [data['sarsa_win_rate'], data['qlearning_win_rate']],
                               color=['blue', 'red'], alpha=0.7)
        axes[row, 2].set_ylim([0, 1])
        axes[row, 2].grid(True, alpha=0.3, axis='y')
        axes[row, 2].set_ylabel('Win Rate', fontsize=9)
        axes[row, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Add value labels on bars
        for bar, rate in zip(bars, [data['sarsa_win_rate'], data['qlearning_win_rate']]):
            height = bar.get_height()
            axes[row, 2].text(bar.get_x() + bar.get_width()/2., height,
                             f'{rate:.3f}',
                             ha='center', va='bottom', fontsize=8)
        
        # Set x-axis label only on bottom row
        if row == 3:
            axes[row, 0].set_xlabel('Episode')
            axes[row, 1].set_xlabel('Episode')
            # No x-axis label needed for bar chart
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    # Print summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    for reward_type in reward_types:
        data = results[reward_type]
        print(f"\n{reward_type.upper()}:")
        print(f"  SARSA - Mean reward: {np.mean(data['sarsa_rewards']):.3f} ± {np.std(data['sarsa_rewards']):.3f}")
        print(f"         Win rate (1000 games): {data['sarsa_win_rate']:.3f}")
        print(f"  Q-Learning - Mean reward: {np.mean(data['qlearning_rewards']):.3f} ± {np.std(data['qlearning_rewards']):.3f}")
        print(f"              Win rate (1000 games): {data['qlearning_win_rate']:.3f}")

if __name__ == "__main__":
    # Choose amount of training steps (episodes)
    TRAINING_STEPS = 10000
    
    main(training_steps=TRAINING_STEPS)

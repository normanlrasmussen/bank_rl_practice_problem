import copy
from bank import Bank
from players import Player

class ThresholdRLPlayer(Player):
    """
    Player that uses a threshold action for RL agents.
    Banks when current_score >= threshold.
    """
    def __init__(self, threshold: int = 0):
        super().__init__()
        self.threshold = threshold
    
    def decide_action(self, state):
        if state["current_score"] >= self.threshold:
            return "bank"
        else:
            return "roll"

class BankEnvironment:
    """
    Environment for reinforcement learning with the bank game.
    State: score after 3 rolls (integer)
    Action: threshold value (50-)
    """
    def __init__(self, rounds: int = 10, opponent: Player = None, reward_type: str = "score"):
        """
        Initialize the bank environment.
        
        Args:
            rounds: Number of rounds per game
            opponent: Opponent player (None for single player)
            reward_type: Type of reward function ("sparse", "relative", "score", "custom")
        """
        self.rounds = rounds
        self.opponent = opponent
        self.reward_type = reward_type
        self.bank = None
        self.current_round = 0
        self.agent_player = None
        self.opponent_player = None
        self.current_state = None  # Store the current state (score after 3 rolls)
        
    def reset(self):
        """
        Reset the environment and start a new game.
        Returns the initial state (score after 3 rolls).
        """
        # Create agent player (will be updated with threshold in step)
        self.agent_player = ThresholdRLPlayer(threshold=0)
        
        # Create players list
        if self.opponent is None:
            players = [self.agent_player]
        else:
            self.opponent_player = copy.deepcopy(self.opponent)
            players = [self.agent_player, self.opponent_player]
        
        # Initialize bank game
        self.bank = Bank(self.rounds, players, verbose=False)
        self.bank.current_round = 1
        self.bank.player_scores = [0 for _ in range(len(players))]
        self.bank.player_score_history = [[] for _ in range(len(players))]
        
        # Get initial state (score after 3 rolls)
        initial_state = self.bank.first_rolls()
        self.current_state = initial_state  # Store the current state
        self.current_round = 1
        
        return initial_state
    
    def step(self, action):
        """
        Take an action (threshold) and play one round.
        
        Args:
            action: Threshold value (50-230)
        
        Returns:
            next_state: Score after 3 rolls of next round (or 0 if game done)
            reward: Reward for this round (given at the end of the round or game)
            done: True if game is complete
        """
        # Update agent's threshold
        self.agent_player.threshold = action
        
        # Play the current round using the stored state (score after 3 rolls)
        # DO NOT call first_rolls() here - use the state from previous step/reset
        score_after_3_rolls = self.current_state
        players_in = [True for _ in range(len(self.bank.players))]
        k = 0
        current_score = score_after_3_rolls
        
        while True:
            state = {
                "current_score": current_score,
                "rounds_remaining": self.rounds - self.bank.current_round,
                "player_scores": self.bank.player_scores.copy(),
                "players_in": players_in.copy(),
            }
            
            # Collect all player decisions simultaneously
            decisions = {}
            for i, player in enumerate(self.bank.players):
                if players_in[i]:
                    decisions[i] = player.decide_action(state)
                else:
                    decisions[i] = None
            
            # Apply all decisions simultaneously
            for i, action_decision in decisions.items():
                if action_decision == "bank" and players_in[i]:
                    players_in[i] = False
                    self.bank.player_scores[i] += current_score
            
            if all(not player_in for player_in in players_in):
                break
            
            roll = self.bank.roll()
            if sum(roll) == 7:
                break
            elif roll[0] == roll[1]:
                current_score *= 2
            else:
                current_score += sum(roll)
            k += 1
        
        # Update player score history
        for i in range(len(self.bank.players)):
            self.bank.player_score_history[i].append(self.bank.player_scores[i])
        
        # Move to next round
        self.bank.current_round += 1
        self.current_round = self.bank.current_round
        
        # Check if game is done (we just finished the last round)
        done = self.bank.current_round > self.rounds
        
        # Calculate reward (pass done flag to know if game is complete)
        reward = self._calculate_reward(done)
        
        # Get next state (score after 3 rolls of next round, or 0 if done)
        if done:
            next_state = 0
        else:
            next_state = self.bank.first_rolls()
        
        # Store the next state for the next step() call
        self.current_state = next_state
        
        return next_state, reward, done
    
    def _calculate_reward(self, done):
        """
        Calculate reward based on reward_type.
        
        Args:
            done: Whether the game is complete
        """
        ##########################################
        # IMPLEMENT REWARD FUNCTION HERE #
        ##########################################

        if self.reward_type == "sparse":
            pass
        
        elif self.reward_type == "relative":
            pass

        elif self.reward_type == "score":
            pass
        
        elif self.reward_type == "custom":
            pass
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
    
    def get_actions(self):
        """
        Return list of all possible actions (thresholds 50-230).
        """
        return [20 * i + 50 for i in range(10)]
    
    def get_final_scores(self):
        """
        Get final scores after game completion.
        Returns (agent_score, opponent_score or None)
        """
        if self.bank is None:
            return None, None
        agent_score = self.bank.player_scores[0]
        if self.opponent is None:
            return agent_score, None
        else:
            return agent_score, self.bank.player_scores[1]


import numpy as np

class Player:
    """
    Base class for all players.
    All players must implement the decide_action method.
    """
    def __init__(self, name: str = None):
        self.name = name
    
    def set_player_id(self, player_id: int):
        self.player_id = player_id
    
    def decide_action(self, state):
        raise NotImplementedError("Subclasses must implement this method")

class ProbabilisticPlayer(Player):
    """
    Player that banks with a certain probability.
    """
    def __init__(self, name: str = None, probability: float = 0.5):
        super().__init__(name)
        self.probability = probability
    
    def decide_action(self, state):
        if np.random.random() < self.probability:
            return "bank"
        else:
            return "roll"

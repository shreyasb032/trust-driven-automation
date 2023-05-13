from typing import Dict
from classes.RewardFunctions import RewardsBase

class PerformanceMetricBase:
    def __init__(self, idx:int = None):
        self.idx = idx
        pass
    
    def get_performance(self):
        raise NotImplementedError

class ImmediateObservedReward(PerformanceMetricBase):
    
    def __init__(self, reward_weights: Dict, reward_fun: RewardsBase, idx: int = 1):

        super().__init__(idx)

        self.reward_weights = reward_weights
        self.reward_fun = reward_fun
        
    def get_performance(self, rec: int, health: int, time: int, threat: int):

        rewards = self.reward_fun.reward(health, time)

        # time_loss_reward and health_loss_reward return negative values
        rew2follow = rec * self.reward_weights["time"] * rewards[1] + \
                     (1-rec) * self.reward_weights["health"] * threat * rewards[0]
        rew2unfollow = (1-rec) * self.reward_weights["time"] * rewards[1] + \
                       rec * self.reward_weights["health"] * threat * rewards[0]

        if rew2follow >= rew2unfollow:
            return 1
        
        return 0

class ImmediateExpectedReward(PerformanceMetricBase):

    def __init__(self, reward_weights: Dict, reward_fun: RewardsBase, idx: int = 2):

        super().__init__(idx)

        self.reward_weights = reward_weights
        self.reward_fun = reward_fun
    
    def get_performance(self, rec: int, health: int, time: int, threat_level: float):
        
        rewards = self.reward_fun.reward(health, time)

        # time_loss_reward and health_loss_reward return negative values
        rew2follow = rec * self.reward_weights["time"] * rewards[1] + (1 - rec) * self.reward_weights["health"] * threat_level * rewards[0]
        rew2unfollow = (1 - rec) * self.reward_weights["time"] * rewards[1] + rec * self.reward_weights["health"] * threat_level * rewards[0]

        if rew2follow >= rew2unfollow:
            return 1

        return 0    

from typing import Dict
from classes.RewardFunctions import RewardsBase


class PerformanceMetricBase:
    def __init__(self, idx: int = None):
        """
        Base class. Other classes should inherit from this class and implement the methods below.
        :param idx: the index associated with the performance metric
                    1 - immediate observed rewards metric
                    2 - immediate expected rewards metric
        """
        self.idx = idx
        pass

    def get_performance(self, rec: int, health: float, time: float, threat: float):
        raise NotImplementedError


class ImmediateObservedReward(PerformanceMetricBase):

    def __init__(self, reward_weights: Dict, reward_fun: RewardsBase, idx: int = 1):
        """
        The metric that gives a performance value after observing rewards from the environment
        :param: reward_weights: dictionary with key "health_weight", "time_weight" specifying the reward weights of the
        human
        :param reward_fun: function that returns rewards based on the state and action chosen/recommended
        idx: the index associated with this performance metric (default: 1)
        """

        super().__init__(idx)

        self.reward_weights = reward_weights
        self.reward_fun = reward_fun

    def get_performance(self, rec: int, health: float, time: float, threat: float):
        """
        The method that returns the performance at the current site.
        :param rec: the recommendation given to the human
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param threat: an integer representing the presence of threat inside the current site
        """

        threat = int(threat)

        rewards = self.reward_fun.reward(health, time, house=None)

        # time_loss_reward and health_loss_reward return negative values
        rew2follow = rec * self.reward_weights["time"] * rewards[1] + \
                     (1 - rec) * self.reward_weights["health"] * threat * rewards[0]
        rew2unfollow = (1 - rec) * self.reward_weights["time"] * rewards[1] + \
                       rec * self.reward_weights["health"] * threat * rewards[0]

        if rew2follow >= rew2unfollow:
            return 1

        return 0


class ImmediateExpectedReward(PerformanceMetricBase):

    def __init__(self, reward_weights: Dict, reward_fun: RewardsBase, idx: int = 2):
        """
        The metric that gives a performance value after observing rewards from the environment
        :param: reward_weights: dictionary with key "health_weight", "time_weight" specifying the reward weights of the
        human
        :param reward_fun: function that returns rewards based on the state and action chosen/recommended
        idx: the index associated with this performance metric (default: 2)
        """
        super().__init__(idx)

        self.reward_weights = reward_weights
        self.reward_fun = reward_fun

    def get_performance(self, rec: int, health: int, time: int, threat: float):
        """
        The method that returns the performance at the current site.
        :param rec: the recommendation given to the human
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param threat: a float representing the level of threat inside the current site
        """
        rewards = self.reward_fun.reward(health, time, house=None)

        # time_loss_reward and health_loss_reward return negative values
        rew2follow = rec * self.reward_weights["time"] * rewards[1] + (1 - rec) * self.reward_weights[
            "health"] * threat * rewards[0]
        rew2unfollow = (1 - rec) * self.reward_weights["time"] * rewards[1] + rec * self.reward_weights[
            "health"] * threat * rewards[0]

        if rew2follow >= rew2unfollow:
            return 1

        return 0

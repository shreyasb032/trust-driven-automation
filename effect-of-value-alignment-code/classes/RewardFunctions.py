"""Classes for keeping rewards consistent through the main code"""


class RewardsBase:
    """
    A base class for all reward functions
    """
    def __init__(self):
        pass
    
    def reward(self, health, time, house):
        """
        Inheriting classes should implement this function to return a tuple -> (health-loss-reward, time-loss-reward)
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param house: the index of the house currently searching
        """
        raise NotImplementedError


class Constant:
    """
    A reward function that returns constant rewards irrespective of the current state and stage
    """
    
    def __init__(self, hl, tc):
        """
        :param hl: the constant cost for losing health
        :param tc: the constant cost for losing time
        """
        self.hl = hl
        self.tc = tc
    
    def reward(self, health=None, time=None, house=None):
        """
        Inheriting classes should implement this function to return a tuple -> (health-loss-reward, time-loss-reward)
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param house: the index of the house currently searching
        """

        return -self.hl, -self.tc


class Linear(RewardsBase):
    """
    Class to implement rewards linear to the current state
    """
    def __init__(self):
        super().__init__()
    
    def reward(self, health, time, house=None):
        """
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param house: the current house number
        """
        return health, -time


class Affine(RewardsBase):
    """Class to implement affine rewards. Health reward is affine, time reward is linear"""

    def __init__(self, max_health) -> None:
        """
        :param max_health: the constant in the affine reward function
        """
        super().__init__()
        self.max_health = max_health

    def reward(self, health, time, house=None):
        """
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param house: the current house number
        """

        return health - self.max_health, -time


class Inverse(RewardsBase):
    """Class to implement inverse rewards"""

    def __init__(self, min_health, max_health, max_time):
        """
        :param min_health: the minimum health below which rewards are constant
        :param max_health: the maximum health level possible for the soldier
        :param max_time: the maximum time after which rewards become constant
        """

        super().__init__()
        self.min_health = min_health
        self.max_health = max_health
        self.max_time = max_time
    
    def reward(self, health, time, house=None):
        """
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param house: the current house number
        """

        if health <= self.min_health:
            health = self.min_health + 1

        h = -self.max_health / (health - self.min_health)
        c = -min(time, self.max_time)

        return h, c

class RewardsBase:

    def __init__(self, num_sites: int):
        """
        A base class for all reward functions
        :param num_sites: number of sites in the mission
        :return: None
        """
        self.N = num_sites

    def reward(self, health: float, time_: float, site_num: int):
        """
        :param health: current health of the soldier
        :param time_: time spent in simulation (corresponds to number of times armored robot is used)
        :param site_num: current site index
        :return: tuple of health loss reward and time loss reward
        Base class does not implement this. All inheriting classes should provide implementation.
        """
        raise NotImplementedError


class EndReward(RewardsBase):

    def __init__(self, num_sites: int):
        """
        :param num_sites: number of sites in the mission
        :return: None
        """
        super().__init__(num_sites)

    def reward(self, health: float, time_: float, site_num: int):
        """
        Returns a reward only at the last search site (at end-of-mission health and time). Returns 0 otherwise.
        :param health: current health of the soldier
        :param time_: time spent in simulation (corresponds to number of times armored robot is used)
        :param site_num: current site index
        :return: tuple of health loss reward and time loss reward
        """
        if health <= 0:
            return -1000, -time_
        if site_num == self.N:
            return health, -time_

        return 0, 0


class ConstantRewards(RewardsBase):

    def __init__(self, num_sites: int, health_loss_cost: float = -10., time_loss_cost: float = -9.):
        """
        :param num_sites: number of sites in the mission
        :param health_loss_cost: the constant cost of losing health
        :param time_loss_cost: the constant cost of losing time
        :return: None
        """
        super().__init__(num_sites)
        self.hl = health_loss_cost
        self.tc = time_loss_cost

    def reward(self, health: float, time_: float, site_num: int):
        """
        :param health: current health of the soldier
        :param time_: time spent in simulation (corresponds to number of times armored robot is used)
        :param site_num: current site index
        :return: tuple of health loss reward and time loss reward
        """
        return self.hl, self.tc

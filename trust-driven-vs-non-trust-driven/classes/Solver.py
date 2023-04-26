import numpy as np
from classes.Rewards import RewardsBase


class NonTrustSolver:
    """
    Solver that chooses optimal actions based on given data. Does not
    consider trust in the reward maximization
    """

    def __init__(self, num_sites: int,
                 prior_levels: np.ndarray[float],
                 after_scan_levels: np.ndarray[float],
                 whr: float,
                 df: float,
                 reward_fun: RewardsBase):
        """
        :param num_sites: number of sites over which to solve the MDP
        :param prior_levels: the prior information about the threat levels in the mission area
        :param after_scan_levels: threat levels after scanning a site
        :param whr: the health weight of the robot
        :param df: the discount factor to be used in expected reward maximization
        :param reward_fun: the reward function. It should take the current health, time, and site number and return
                            a reward value
        :return: None
        """
        self.N = num_sites
        self.prior_levels = prior_levels
        self.after_scan_levels = after_scan_levels
        self.whr = whr
        self.wcr = 1. - whr
        self.df = df
        self.reward_fun = reward_fun
        self.current_house = 0
        self.health = 100
        self.time_ = 0

    def get_action(self):
        """
        Returns the optimal action at the current site
        :return: the optimal action at the current site
        """
        sites_remaining = self.N - self.current_house
        value_matrix = np.zeros((sites_remaining + 1,), dtype=float)
        action_matrix = np.zeros((sites_remaining,), dtype=int)

        for i in reversed(range(sites_remaining)):
            # Going backward
            hl, tc = self.reward_fun.reward()
            site_idx = self.current_house + i
            threat_level = self.prior_levels[site_idx]  # Use the prior level at future sites
            if i == 0:
                threat_level = self.after_scan_levels[site_idx]  # Use the after scan level at current site

            reward_0 = self.whr * hl * threat_level
            reward_1 = self.wcr * tc

            qvalue_0 = reward_0 + self.df * value_matrix[i + 1]  # Q-value for selecting action 0 here
            qvalue_1 = reward_1 + self.df * value_matrix[i + 1]  # Q-value for selecting action 1 here

            if qvalue_1 >= qvalue_0:
                value_matrix[i] = qvalue_1
                action_matrix[i] = 1
            else:
                value_matrix[i] = qvalue_0
                action_matrix[i] = 0

        return action_matrix[0]

    def forward(self, threat_obs: int, action: int):
        """
        Moves the solver one stage forward
        :param threat_obs: an integer representing the presence of threat inside the current site
        :param action: the action chosen at the current site
        :return:
        """
        self.current_house += 1

        if action == 1:
            self.time_ += 10
        else:
            if threat_obs:
                self.health -= 10

    def get_action_one_step(self):
        """
        Returns the optimal action considering only immediate expected rewards
        :return: the optimal action
        """
        hl, tc = self.reward_fun.reward()
        threat_level = self.after_scan_levels[self.current_house]
        reward_0 = self.whr * hl * threat_level
        reward_1 = self.wcr * tc

        if reward_1 >= reward_0:
            return 1

        return 0

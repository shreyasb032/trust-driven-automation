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
                 reward_fun: RewardsBase,
                 health_loss: float = 10.,
                 time_loss: float = 10.):
        """
        :param num_sites: number of sites over which to solve the MDP
        :param prior_levels: the prior information about the threat levels in the mission area
        :param after_scan_levels: threat levels after scanning a site
        :param whr: the health weight of the robot
        :param df: the discount factor to be used in expected reward maximization
        :param reward_fun: the reward function. It should take the current health, time, and site number and return
                            a reward value
        :param health_loss: The health lost after encountering threat without protection (default: 10.)
        :param time_loss: The time loss resulted by using the armored robot (default: 10)
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
        self.health_loss = health_loss
        self.time_loss = time_loss

    def get_action(self):
        """
        Returns the optimal action at the current site
        :return: the optimal action at the current site
        """
        sites_remaining = self.N - self.current_house
        # Indexed by number of stages, health_level, time_level
        value_matrix = np.zeros((sites_remaining + 1, sites_remaining, sites_remaining), dtype=float)
        action_matrix = np.zeros((sites_remaining, sites_remaining, sites_remaining), dtype=int)

        # Going backwards through stages
        for i in reversed(range(sites_remaining)):
            site_number = i + self.current_house
            threat_level = self.prior_levels[site_number]
            possible_health_levels = self.health - np.arange(i) * self.health_loss
            possible_time_levels = self.time_ + np.arange(i) * self.time_loss

            if i == 0:
                threat_level = self.after_scan_levels[site_number]
                possible_health_levels = [possible_health_levels]
                possible_time_levels = [possible_time_levels]

            for idx_h, h in enumerate(possible_health_levels):
                for idx_c, c in enumerate(possible_time_levels):
                    hl, tc = self.reward_fun.reward(h, c, site_number)
                    # hl and tc are the state dependent (action independent) rewards
                    # r0 is the immediate reward obtained by recommending action 0
                    # r1 is the immediate reward obtained by recommending action 1
                    val0 = self.whr * hl + self.wcr * tc + \
                           self.df * (threat_level * value_matrix[i + 1, idx_h + 1, idx_c]
                                      + (1. - threat_level) * value_matrix[i + 1, idx_h, idx_c])
                    val1 = self.whr * hl + self.wcr * tc + self.df * (value_matrix[i + 1, idx_h, idx_c + 1])

                    if val0 > val1:
                        value_matrix[i, idx_h, idx_c] = val0
                        action_matrix[i, idx_h, idx_c] = 0
                    else:
                        value_matrix[i, idx_h, idx_c] = val1
                        action_matrix[i, idx_h, idx_c] = 1

            return action_matrix[0, 0, 0]

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
        hl, tc = self.reward_fun.reward(0.0, 0.0, 0)
        threat_level = self.after_scan_levels[self.current_house]
        reward_0 = self.whr * hl * threat_level
        reward_1 = self.wcr * tc

        if reward_1 >= reward_0:
            return 1

        return 0

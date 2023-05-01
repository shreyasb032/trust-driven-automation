import numpy as np
from classes.Rewards import RewardsBase
from classes.Human import HumanBase


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
        value_matrix = np.zeros((sites_remaining + 1, sites_remaining + 1, sites_remaining + 1), dtype=float)
        action_matrix = np.zeros((sites_remaining, sites_remaining + 1, sites_remaining + 1), dtype=int)

        # Going backwards through stages
        for i in reversed(range(sites_remaining)):
            site_number = i + self.current_house
            threat_level = self.prior_levels[site_number]
            possible_health_levels = self.health - np.arange(i + 1) * self.health_loss
            possible_time_levels = self.time_ + np.arange(i + 1) * self.time_loss

            if i == 0:
                threat_level = self.after_scan_levels[site_number]

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


class SolverWithTrust:

    def __init__(self, num_sites: int,
                 prior_levels: np.ndarray[float],
                 after_scan_levels: np.ndarray[float],
                 whr: float,
                 df: float,
                 reward_fun: RewardsBase,
                 human_model: HumanBase):
        """
        :param num_sites: number of sites over which to solve the MDP
        :param prior_levels: the prior information about the threat levels in the mission area
        :param after_scan_levels: threat levels after scanning a site
        :param whr: the health weight of the robot
        :param df: the discount factor to be used in expected reward maximization
        :param reward_fun: the reward function for the robot. It should take the current health, time, and site number
                    and return a reward value
        :param human_model: the estimated model for the human (trust params, reward weights, performance history, etc.)
        :return: None
        """

        self.N = num_sites
        self.prior_levels = prior_levels
        self.after_scan_levels = after_scan_levels
        self.whr = whr
        self.wcr = 1. - whr
        self.df = df
        self.reward_fun = reward_fun
        self.human_model = human_model
        self.trust_feedback_history = np.zeros((self.N,), dtype=float)

    def forward(self, threat_obs: int, action: int, threat_level: float):
        """
        Moves the solver one stage forward
        :param threat_obs: an integer representing the presence of threat inside the current site
        :param action: the action chosen by the human
        :param threat_level: the threat level reported by the drone AFTER SCANNING
        :return:
        """
        self.human_model.forward(threat_obs, action)
        self.human_model.update_posterior(threat_level)

    def get_action(self):
        """
        Returns the optimal action at the current site
        :return: the optimal action at the current site
        """
        sites_remaining = self.N - self.human_model.current_site
        # Indexed by number of stages, trust, health_level, time_level
        value_matrix = np.zeros((sites_remaining + 1, sites_remaining + 1, sites_remaining + 1, sites_remaining + 1),
                                dtype=float)
        action_matrix = np.zeros((sites_remaining, sites_remaining + 1, sites_remaining + 1, sites_remaining + 1),
                                 dtype=int)

        alpha_current, beta_current = self.human_model.get_alphabeta()
        # Going backwards through stages
        for i in reversed(range(sites_remaining)):
            site_number = i + self.human_model.current_site
            threat_level = self.prior_levels[site_number]
            possible_successes = np.arange(i + 1)
            possible_failures = sites_remaining - possible_successes
            if i == 0:
                threat_level = self.after_scan_levels[site_number]

            for j, ns in enumerate(possible_successes):
                nf = possible_failures[j]
                _alpha = alpha_current + ns * self.human_model.trust_params[2]
                _beta = beta_current + nf * self.human_model.trust_params[3]
                trust = _alpha / (_alpha + _beta)

                possible_health_levels = self.human_model.current_health() - \
                                         np.arange(i + 1) * self.human_model.health_loss
                possible_time_levels = self.human_model.current_time() + np.arange(i + 1) * self.human_model.time_loss

                for idx_h, h in enumerate(possible_health_levels):
                    for idx_c, c in enumerate(possible_time_levels):
                        hl, tc = self.reward_fun.reward(h, c, site_number)

                        # For recommending to NOT USE the armored robot
                        # 1. Compute the probabilities of choosing either action, based on the recommendation and
                        #    the estimate of the health reward weight of the human
                        prob0, prob1 = self.human_model.get_probabilities(trust,
                                                                          recommendation=0,
                                                                          threat_level=threat_level)
                        # 2. Compute the one-step expected rewards for recommending each action based on these
                        #    probabilities
                        reward = prob0 * self.whr * hl + prob1 * self.wcr * tc
                        # 3. Compute the q-value at this stage, state, and action using the Bellman equation
                        # 4. Set the value of this stage and state to be the maximum of the two
                        # 5. Return the action that corresponds to the value at stage 0, state 0,0,0
                        pass

        return -1

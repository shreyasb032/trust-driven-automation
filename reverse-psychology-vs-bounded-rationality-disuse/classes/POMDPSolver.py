from typing import Dict, List
import numpy as np
from copy import copy
from classes.RewardFunctions import RewardsBase


class Solver:
    """
    Base class for solving the MDP
    """

    def __init__(self, num_sites: int, rob_weights: Dict, trust_params: List,
                 prior_levels: List, after_scan_levels: List,
                 threats: List, est_human_weights: Dict,
                 reward_fun: RewardsBase,
                 hum_mod='bounded_rational', df=0.7, kappa=0.05):
        """
        :param num_sites: number of sites in the mission
        :param rob_weights: the weights of the robot's reward function, a dict with keys 'health' and 'time'
        :param prior_levels: the prior threat levels in the mission
        :param after_scan_levels: the threat level obtained after scanning a site
        :param threats: binary list indicating the presence of threat
        :param reward_fun: the reward function
        :param hum_mod: the human model to use: choice between bounded_rational, rev_psych, and disuse
        :param df: the discount factor associated with the value iteration algorithm
        :param kappa: the rationality coefficient of the bounded rational human model
        """

        # Total number of houses
        self.N = num_sites

        # Robot's health and time weights
        self.wh = rob_weights['health']
        self.wc = rob_weights['time']

        # Trust reward weight
        self.wt = rob_weights['trust']

        # Discount factor
        self.df = df

        # Estimated human reward weights
        self.wh_hum = est_human_weights['health']
        self.wc_hum = est_human_weights['time']

        # Human model type
        self.hum_mod = hum_mod

        # Reward function type
        self.reward_fun = reward_fun

        # Rationality coefficient
        self.kappa = kappa

        # Storage
        self.performance_history = np.zeros((self.N,), dtype=int)
        self.threat_levels = copy(prior_levels)
        self.trust_params = copy(trust_params)
        self.after_scan_levels = copy(after_scan_levels)
        self.threats = copy(threats)
        self.max_health = 100
        self.health = 100
        self.time = 0

        # Initial guesses for the trust params
        self.gp_list = {0.0: [2., 98., 20., 30.],
                        0.1: [10., 90., 20., 30.],
                        0.2: [20., 80., 20., 30.],
                        0.3: [30., 70., 20., 30.],
                        0.4: [40., 60., 20., 30.],
                        0.5: [50., 50., 20., 30.],
                        0.6: [60., 40., 20., 30.],
                        0.7: [70., 30., 20., 30.],
                        0.8: [80., 20., 20., 30.],
                        0.9: [90., 10., 20., 30.],
                        1.0: [98., 2., 20., 30.]}

    def update_danger(self, threats, prior_levels, after_scan_levels, reset=True):
        """
        Updates all the threat levels
        :param threats: a binary list indicating the presence of threats
        :param prior_levels: the threat levels prior to scanning a site
        :param after_scan_levels: the threat levels after scanning a site
        :param reset: a boolean indicating whether to erase the performance history
        """

        self.threat_levels = copy(prior_levels)
        self.threats = copy(threats)
        self.after_scan_levels = copy(after_scan_levels)

        if reset:
            self.reset()

    def update_params(self, params: List):
        """
        Update the estimated trust parameters of the human model
        :param params: a list [alpha0, beta0, ws, wf]
        """
        self.trust_params = params.copy()

    def get_trust_params(self):
        """
        Helper function to return the currently estimated trust params of the human
        """
        return self.trust_params

    def reset(self, trust_fb=None):
        """
        Resets the performance history
        :param trust_fb: trust feedback, if given, resets the estimated trust parameters to an initial guess
        """
        self.performance_history = 0

        if trust_fb is not None:
            self.trust_params = self.gp_list[round(trust_fb, 1)]

    def set_est_human_weights(self, est_human_weights):
        """
        Helper function to set the estimated reward weights of the human model
        :param est_human_weights: a dictionary with keys 'health' and 'time' giving the reward weights
        """
        self.wh_hum = est_human_weights['health']
        self.wc_hum = est_human_weights['time']

    def set_reward_weights(self, rob_weights):
        """
        Helper function to set the robot's reward weights
        :param rob_weights: a dict with keys 'health' and 'time' giving the reward weights
        """

        self.wh = rob_weights['health']
        self.wc = rob_weights['time']

    def __get_immediate_reward(self, house, health, time, action, wh, wc):
        """
        Helper function to get the immediate observed rewards given the state, stage, and reward weights
        """

        hl, tc = self.reward_fun.reward(health, time, house)

        r1 = wc * tc
        r2 = wh * hl
        r3 = 0

        r_follow = action * r1 + (1 - action) * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)
        r_not_follow = (1 - action) * r1 + action * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)

        return r_follow, r_not_follow

    def get_immediate_reward_rob(self, current_house, current_health, current_time, action):
        """
        Get the immediate observed rewards for the robot
        """
        return self.__get_immediate_reward(current_house, current_health, current_time, action, self.wh, self.wc)

    def get_immediate_reward_hum(self, current_house, current_health, current_time, action):
        """
        Get the immediate observed rewards for the human model
        """
        return self.__get_immediate_reward(current_house, current_health, current_time, action, self.wh_hum,
                                           self.wc_hum)

    def get_recommendation(self, current_house, current_health, current_time):
        """
        The MDP solving algorithm (value iteration)
        :param current_house: the current site number to search
        :param current_health: the current level of health of the soldier
        :param current_time: the current amount of time spent in the mission
        """

        alpha_0 = self.trust_params[0]
        beta_0 = self.trust_params[1]
        ws = self.trust_params[2]
        wf = self.trust_params[3]

        ns = np.sum(self.performance_history)
        nf = len(self.performance_history) - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        self.health = current_health
        self.time = current_time

        num_houses_to_go = self.N - current_house

        #                         stages                  successes         healths           times
        value_matrix = np.zeros(
            (num_houses_to_go + 1, num_houses_to_go + 1, num_houses_to_go + 1, num_houses_to_go + 1),
            dtype=float)  # Extra stage of value zero
        action_matrix = np.zeros((num_houses_to_go, num_houses_to_go + 1, num_houses_to_go + 1, num_houses_to_go + 1),
                                 dtype=int)

        # Give more info at current house
        self.threat_levels[current_house] = self.after_scan_levels[current_house]
        # import pdb; pdb.set_trace()

        # Going backwards in stages
        for t in reversed(range(num_houses_to_go)):

            # Possible vals at stage t
            possible_alphas = alpha_previous + np.arange(t + 1) * ws
            possible_betas = beta_previous + (t - np.arange(t + 1)) * wf
            possible_healths = current_health - np.arange(t + 1) * 10
            possible_times = current_time + np.arange(t + 1) * 10
            # import pdb; pdb.set_trace()

            for i, alpha in enumerate(possible_alphas):
                beta = possible_betas[i]
                trust = alpha / (alpha + beta)

                for j, h in enumerate(possible_healths):
                    for k, c in enumerate(possible_times):

                        phl = 0.
                        pcl = 0.
                        ptl = 0.

                        self.wh_hum = self.wh         # Assume that the human has the same reward weight as the solver
                        self.wc_hum = 1 - self.wh_hum

                        # Estimated expected immediate rewards for human for choosing to NOT USE and
                        # USE RARV respectively
                        hl, tc = self.reward_fun.reward(h, c)
                        r0_hum = self.wh_hum * hl * self.threat_levels[t + current_house]
                        r1_hum = self.wc_hum * tc

                        # CASE 1: Expected reward-to-go to recommend to NOT USE RARV
                        if self.hum_mod == 'rev_psych':
                            # probability of health loss
                            # Probability of NOT USING RARV * probability of threat
                            phl = trust * self.threat_levels[t + current_house]

                            # probability of time loss
                            # Probability of USING RARV
                            pcl = 1 - trust

                        elif self.hum_mod == 'disuse':
                            # probability of health loss
                            # Probability of NOT USING RARV * Probability of Threat Presence
                            phl = (trust + (1 - trust) * int(r0_hum > r1_hum)) * self.threat_levels[t + current_house]

                            # probability of time loss
                            # Probability of using RARV
                            pcl = (1 - trust) * int(r1_hum > r0_hum)

                        elif self.hum_mod == 'bounded_rational':
                            # Probability of health loss
                            # Probability of NOT USING RARV (Proportional to)
                            p0 = np.exp(self.kappa * r0_hum)
                            # Probability of USING RARV (Proportional to)
                            p1 = np.exp(self.kappa * r1_hum)

                            # Normalizing
                            p0 /= (p0 + p1)
                            p1 = 1 - p0

                            # Probability of NOT USING RARV * Probability of Threat Presence
                            phl = (trust + (1 - trust) * p0) * self.threat_levels[t + current_house]

                            # Probability of time loss
                            # Probability of using RARV
                            pcl = (1 - trust) * p1

                        else:
                            raise "Human model incorrectly specified"

                        # Expected immediate reward to recommend to not use RARV
                        r0 = phl * self.wh * hl + pcl * self.wc * tc

                        # probability of trust loss
                        ptl = int(r0_hum < r1_hum)
                        pti = 1 - ptl
                        trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                        # Trust increase, health loss, no time loss + Trust increase,
                        # no health loss, time loss + Trust increase, no health loss, no time loss
                        next_stage_reward = pti * (
                                phl * (1 - pcl) * value_matrix[t + 1, i + 1, j + 1, k] + pcl * (1 - phl) *
                                value_matrix[t + 1, i + 1, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[
                                    t + 1, i + 1, j, k])

                        # Trust decrease, health loss, no time loss + Trust deccrease,
                        # no health loss, time loss + Trust decrease, no health loss, no time loss
                        next_stage_reward += ptl * (
                                phl * (1 - pcl) * value_matrix[t + 1, i, j + 1, k] + pcl * (1 - phl) * value_matrix[
                            t + 1, i, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[t + 1, i, j, k])

                        r0 += self.df * next_stage_reward + trust_gain_reward

                        # Expected reward to go to recommend to USE RARV
                        if self.hum_mod == "rev_psych":
                            # Probability of losing health
                            phl = (1 - trust) * self.threat_levels[t + current_house]
                            # Probability of losing time
                            pcl = trust

                        elif self.hum_mod == "disuse":
                            # Probability of losing health
                            # Probability of NOT USING RARV * probability of threat presence
                            phl = (1 - trust) * int(r0_hum > r1_hum) * self.threat_levels[t + current_house]

                            # Probability of losing time
                            # Probabilit of USING RARV
                            pcl = trust + (1 - trust) * int(r1_hum > r0_hum)

                        elif self.hum_mod == 'bounded_rational':
                            # Probability of health loss
                            # Probability of NOT USING RARV (Proportional to)
                            p0 = np.exp(self.kappa * r0_hum)
                            # Probability of USING RARV (Proportional to)
                            p1 = np.exp(self.kappa * r1_hum)

                            # Normalizing
                            p0 /= (p0 + p1)
                            p1 = 1 - p0

                            # Probability of NOT USING RARV * Probability of Threat Presence
                            phl = (1 - trust) * p0 * self.threat_levels[t + current_house]

                            # Probability of time loss
                            # Probability of using RARV
                            pcl = trust + (1 - trust) * p1

                        else:
                            raise "Human model incorrectly specified"

                        # Probability of trust loss
                        ptl = int(r0_hum > r1_hum)

                        # Probability of trust increase
                        pti = 1 - ptl

                        # Expected immediate reward to recommend to USE RARV
                        r1 = phl * self.wh * hl + pcl * self.wc * tc

                        trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                        # Trust increase, health loss, no time loss + Trust increase,
                        # no health loss, time loss + Trust increase, no health loss, no time loss
                        next_stage_reward = pti * (
                                phl * (1 - pcl) * value_matrix[t + 1, i + 1, j + 1, k] + pcl * (1 - phl) *
                                value_matrix[t + 1, i + 1, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[
                                    t + 1, i + 1, j, k])

                        # Trust decrease, health loss, no time loss + Trust decrease,
                        # no health loss, time loss + Trust decrease, no health loss, no time loss
                        next_stage_reward += ptl * (
                                phl * (1 - pcl) * value_matrix[t + 1, i, j + 1, k] + pcl * (1 - phl) * value_matrix[
                            t + 1, i, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[t + 1, i, j, k])

                        r1 += self.df * next_stage_reward + trust_gain_reward

                        action_matrix[t, i, j, k] = int(r1 > r0)
                        value_matrix[t, i, j, k] = max(r1, r0)

        return action_matrix[0, 0, 0, 0]

    def forward(self, current_house, rec, health, curr_time):
        """
        Moves the solver forward one step. Changes health, time, house number, and adds to the performance history
        :param current_house: the number of hte current search site
        :param rec: the recommendation given by the robot
        :param health: the level of health of the soldier before searching this site
        :param curr_time: the time spent in the mission before searching this site
        """

        self.wh_hum = self.wh                 # Assume that the human has the same reward weight as the solver
        self.wc_hum = 1 - self.wh_hum

        hl, tc = self.reward_fun.reward(health, curr_time)

        rew2use = self.wc_hum * tc
        rew2notuse = self.wh_hum * self.threats[current_house] * hl

        if rec:
            if rew2use >= rew2notuse:
                self.performance_history[current_house] = 1
            else:
                self.performance_history[current_house] = 0
        else:
            if rew2notuse >= rew2use:
                self.performance_history[current_house] = 1
            else:
                self.performance_history[current_house] = 0

    def get_last_performance(self, current_site):
        """
        Helper function to return the last performance of the robot
        :param current_site: the index of the current search site
        """

        return self.performance_history[current_site]

    def get_trust_estimate(self, current_site):
        """
        Helper function to return the estimated value of the human's current level of trust
        :param current_site: the index of the current search site
        """

        params = self.trust_params
        per = np.sum(self.performance_history)
        _alpha = params[0] + per * params[2]
        _beta = params[1] + (current_site - per) * params[3]

        return _alpha / (_alpha + _beta)


class SolverConstantRewards(Solver):

    def __init__(self, num_sites: int, rob_weights: Dict, trust_params: List, prior_levels: List,
                 after_scan_levels: List, threats: List, est_human_weights: Dict, reward_fun: RewardsBase,
                 hum_mod='bounded_rational',
                 df=0.9, kappa=0.05, hl=10.0, tc=10.0):
        """
        :param num_sites: number of sites in the mission
        :param rob_weights: the weights of the robot's reward function, a dict with keys 'health' and 'time'
        :param prior_levels: the prior threat levels in the mission
        :param after_scan_levels: the threat level obtained after scanning a site
        :param threats: binary list indicating the presence of threat
        :param reward_fun: the reward function
        :param hum_mod: the human model to use: choice between bounded_rational, rev_psych, and disuse
        :param df: the discount factor associated with the value iteration algorithm
        :param kappa: the rationality coefficient of the bounded rational human model
        :param hl: the cost for losing health (positive)
        :param tc: the cost for losing time (positive)
        """

        super().__init__(num_sites, rob_weights, trust_params, prior_levels, after_scan_levels, threats,
                         est_human_weights, reward_fun, hum_mod, df, kappa)
        self.hl = hl
        self.tc = tc

    def __get_immediate_reward(self, house, action, wh, wc):
        """
        :param house: the index of the site at which the reward is to be calculated
        :param action: the action for which reward is to be calculated
        :param wh: the health reward weight
        :param wc: the time reward weight
        """

        r1 = -wc * self.tc
        r2 = -wh * self.hl
        r3 = 0

        r_follow = action * r1 + (1 - action) * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)
        r_not_follow = (1 - action) * r1 + action * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)

        return r_follow, r_not_follow

    def get_immediate_reward_rob(self, current_house, action):
        """
        :param current_house: the index of the current search site
        :param action: the action for which reward is to be calculated
        """
        return self.__get_immediate_reward(current_house, action, self.wh, self.wc)

    def get_immediate_reward_hum(self, current_house, action):
        """
        :param current_house: the index of the current search site
        :param action: the action for which reward is to be calculated
        """
        return self.__get_immediate_reward(current_house, action, self.wh_hum, self.wc_hum)

    def get_recommendation(self, current_house):
        """
        :param current_house: the index of the current search site
        """

        alpha_0 = self.trust_params[0]
        beta_0 = self.trust_params[1]
        ws = self.trust_params[2]
        wf = self.trust_params[3]

        ns = np.sum(self.performance_history)
        nf = current_house - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        num_houses_to_go = self.N - current_house

        #                         stages               successes
        value_matrix = np.zeros((num_houses_to_go + 1, num_houses_to_go + 1), dtype=float)  # Extra stage of value zero
        action_matrix = np.zeros((num_houses_to_go, num_houses_to_go + 1), dtype=int)

        # Give more info at current house
        self.threat_levels[current_house] = self.after_scan_levels[current_house]

        # Going backwards in stages
        for t in reversed(range(num_houses_to_go)):

            # Possible values at stage t
            possible_alphas = alpha_previous + np.arange(t + 1) * ws
            possible_betas = beta_previous + (t - np.arange(t + 1)) * wf

            # The below are actual observable rewards based on threat presence
            self.wh_hum = self.wh                      # Assume that the human has the same reward weight as the solver
            self.wc_hum = 1. - self.wh_hum
            r0_no_threat = 0
            r0_threat = -self.wh_hum * self.hl

            # The below are expected rewards based on the threat level
            r0_hum = -self.wh_hum * self.hl * self.threat_levels[t + current_house]
            r1_hum = -self.wc_hum * self.tc

            if self.hum_mod == 'bounded_rational':
                # Probability of NOT USING RARV (Proportional to)
                p0 = 1. / (1. + np.exp(self.kappa * (r1_hum - r0_hum)))
                # Probability of USING RARV (Proportional to)
                p1 = 1. - p0

            for i, alpha in enumerate(possible_alphas):

                beta = possible_betas[i]
                trust = alpha / (alpha + beta)

                phl = 0.
                pcl = 0.
                ptl = 0.

                # CASE 1: Expected reward-to-go to recommend to NOT USE RARV
                if self.hum_mod == 'rev_psych':
                    # probability of health loss
                    # Probability of NOT USING RARV * probability of threat
                    phl = trust * self.threat_levels[t + current_house]

                    # probability of time loss
                    # Probability of USING RARV
                    pcl = 1. - trust

                elif self.hum_mod == 'disuse':
                    # probability of health loss
                    # Probability of NOT USING RARV * Probability of Threat Presence
                    phl = (trust + (1. - trust) * int(r0_hum > r1_hum)) * self.threat_levels[t + current_house]

                    # probability of time loss
                    # Probability of using RARV
                    pcl = (1. - trust) * int(r1_hum > r0_hum)

                elif self.hum_mod == 'bounded_rational':
                    # Probability of health loss
                    # Probability of NOT USING RARV * Probability of Threat Presence
                    phl = (trust + (1. - trust) * p0) * self.threat_levels[t + current_house]

                    # Probability of time loss
                    # Probability of using RARV
                    pcl = (1. - trust) * p1

                else:
                    raise "Human model incorrectly specified"

                # Expected immediate reward to recommend to not use RARV
                r0 = -phl * self.wh * self.hl - pcl * self.wc * self.tc

                # probability of trust gain
                pti = self.threat_levels[t + current_house] * int(r0_threat > r1_hum) + (
                        1.0 - self.threat_levels[t + current_house]) * int(r0_no_threat > r1_hum)

                # probability of trust loss
                ptl = 1. - pti

                # Trust gain reward
                trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                # Trust increase
                next_stage_reward = pti * value_matrix[t + 1, i + 1]

                # Trust decrease
                next_stage_reward += ptl * value_matrix[t + 1, i]
                r0 += self.df * next_stage_reward + trust_gain_reward

                # Expected reward to go to recommend to USE RARV
                if self.hum_mod == "rev_psych":
                    # Probability of losing health
                    phl = (1. - trust) * self.threat_levels[t + current_house]
                    # Probability of losing time
                    pcl = trust

                elif self.hum_mod == "disuse":
                    # Probability of losing health
                    # Probability of NOT USING RARV * probability of threat presence
                    phl = (1. - trust) * int(r0_hum > r1_hum) * self.threat_levels[t + current_house]

                    # Probability of losing time
                    # Probability of USING RARV
                    pcl = trust + (1. - trust) * int(r1_hum > r0_hum)

                elif self.hum_mod == 'bounded_rational':
                    # Probability of health loss
                    # Probability of NOT USING RARV * Probability of Threat Presence
                    phl = (1. - trust) * p0 * self.threat_levels[t + current_house]

                    # Probability of time loss
                    # Probability of using RARV
                    pcl = trust + (1. - trust) * p1

                else:
                    raise "Human model incorrectly specified"

                # Probability of trust gain
                pti = self.threat_levels[t + current_house] * int(r0_threat < r1_hum) + (
                        1.0 - self.threat_levels[t + current_house]) * int(r0_no_threat < r1_hum)

                # Probability of trust loss
                ptl = 1. - pti

                # Expected immediate reward to recommend to USE RARV
                r1 = -phl * self.wh * self.hl - pcl * self.wc * self.tc

                # Trust gain reward
                trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                # Trust increase
                next_stage_reward = pti * value_matrix[t + 1, i + 1]

                # Trust decrease
                next_stage_reward += ptl * value_matrix[t + 1, i]

                r1 += self.df * next_stage_reward + trust_gain_reward

                action_matrix[t, i] = int(r1 > r0)
                value_matrix[t, i] = max(r1, r0)

        return action_matrix[0, 0]

    def forward(self, current_house, rec):
        """
        :param current_house: the index of the current search site
        :param rec: the recommendation given by the robot
        """

        self.wh_hum = self.wh            # Assume that the human has the same reward weight as the solver
        self.wc_hum = 1. - self.wh_hum

        rew2use = -self.wc_hum * self.tc
        rew2notuse = -self.wh_hum * self.threats[current_house] * self.hl

        if rec:
            if rew2use >= rew2notuse:
                self.performance_history[current_house] = 1
            else:
                self.performance_history[current_house] = 0
        else:
            if rew2notuse >= rew2use:
                self.performance_history[current_house] = 1
            else:
                self.performance_history[current_house] = 0

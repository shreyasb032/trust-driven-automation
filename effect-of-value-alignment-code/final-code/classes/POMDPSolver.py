from typing import Dict, List
import numpy as np
from copy import copy
from classes.RewardFunctions import RewardsBase
from classes.IRLModel import Posterior


class Solver:

    def __init__(self, num_sites: int, rob_weights: Dict, trust_params: List,
                 prior_levels: List, after_scan_levels: List,
                 threats: List, est_human_weights: Dict,
                 reward_fun: RewardsBase,
                 hum_mod='bounded_rational', df=0.7, kappa=0.05):

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
        self.performance_history = []
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
        self.threat_levels = copy(prior_levels)
        self.threats = copy(threats)
        self.after_scan_levels = copy(after_scan_levels)

        if reset:
            self.reset()

    def update_params(self, params: List):
        self.trust_params = params.copy()

    def get_trust_params(self):
        return self.trust_params

    def reset(self, trust_fb=None):
        self.performance_history.clear()

        if trust_fb is not None:
            self.trust_params = self.gp_list[round(trust_fb, 1)]

    def set_est_human_weights(self, est_human_weights):
        self.wh_hum = est_human_weights['health']
        self.wc_hum = est_human_weights['time']

    def set_reward_weights(self, rob_weights):

        self.wh = rob_weights['health']
        self.wc = rob_weights['time']

    def __get_immediate_reward(self, house, health, time, action, wh, wc):

        hl, tc = self.reward_fun.reward(health, time, house)

        r1 = wc * tc
        r2 = wh * hl
        r3 = 0

        r_follow = action * r1 + (1 - action) * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)
        r_not_follow = (1 - action) * r1 + action * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)

        return r_follow, r_not_follow

    def get_immediate_reward_rob(self, current_house, current_health, current_time, action):
        return self.__get_immediate_reward(current_house, current_health, current_time, action, self.wh, self.wc)

    def get_immediate_reward_hum(self, current_house, current_health, current_time, action):
        return self.__get_immediate_reward(current_house, current_health, current_time, action, self.wh_hum,
                                           self.wc_hum)

    def get_recommendation(self, current_house, current_health, current_time, posterior: Posterior):

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

                        self.wh_hum = posterior.get_mean()
                        self.wc_hum = 1 - self.wh_hum

                        # Estimated expected immediate rewards for human for choosing to NOT USE and USE RARV respectively
                        hl, tc = self.reward_fun.reward(h, c)
                        # import pdb; pdb.set_trace()
                        r0_hum = self.wh_hum * hl * self.threat_levels[t + current_house]
                        r1_hum = self.wc_hum * tc

                        ######### CASE 1: Expected reward-to-go to recommend to NOT USE RARV ###########
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

                        # Trust increase, health loss, no time loss + Trust increase, no health loss, time loss + Trust increase, no health loss, no time loss
                        next_stage_reward = pti * (
                                phl * (1 - pcl) * value_matrix[t + 1, i + 1, j + 1, k] + pcl * (1 - phl) *
                                value_matrix[t + 1, i + 1, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[
                                    t + 1, i + 1, j, k])

                        # Trust decrease, health loss, no time loss + Trust deccrease, no health loss, time loss + Trust decrease, no health loss, no time loss
                        next_stage_reward += ptl * (
                                phl * (1 - pcl) * value_matrix[t + 1, i, j + 1, k] + pcl * (1 - phl) * value_matrix[
                            t + 1, i, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[t + 1, i, j, k])

                        r0 += self.df * next_stage_reward + trust_gain_reward

                        ############### Expected reward to go to recommend to USE RARV #############
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

                        # Trust increase, health loss, no time loss + Trust increase, no health loss, time loss + Trust increase, no health loss, no time loss
                        next_stage_reward = pti * (
                                phl * (1 - pcl) * value_matrix[t + 1, i + 1, j + 1, k] + pcl * (1 - phl) *
                                value_matrix[t + 1, i + 1, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[
                                    t + 1, i + 1, j, k])

                        # Trust decrease, health loss, no time loss + Trust deccrease, no health loss, time loss + Trust decrease, no health loss, no time loss
                        next_stage_reward += ptl * (
                                phl * (1 - pcl) * value_matrix[t + 1, i, j + 1, k] + pcl * (1 - phl) * value_matrix[
                            t + 1, i, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[t + 1, i, j, k])

                        r1 += self.df * next_stage_reward + trust_gain_reward

                        action_matrix[t, i, j, k] = int(r1 > r0)
                        value_matrix[t, i, j, k] = max(r1, r0)

                        # import pdb; pdb.set_trace()

        return action_matrix[0, 0, 0, 0]

    def forward(self, current_house, rec, health, curr_time, posterior: Posterior):

        self.wh_hum = posterior.get_mean()
        self.wc_hum = 1 - self.wh_hum

        hl, tc = self.reward_fun.reward(health, curr_time)

        rew2use = self.wc_hum * tc
        rew2notuse = self.wh_hum * self.threats[current_house] * hl

        if rec:
            if rew2use >= rew2notuse:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)
        else:
            if rew2notuse >= rew2use:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)

    def get_last_performance(self):

        return self.performance_history[-1]

    def get_trust_estimate(self):

        params = self.trust_params
        per = np.sum(self.performance_history)
        _alpha = params[0] + per * params[2]
        _beta = params[1] + (len(self.performance_history) - per) * params[3]

        return _alpha / (_alpha + _beta)


class SolverConstantRewards(Solver):

    def __init__(self, N: int, rob_weights: Dict, trust_params: List, prior_levels: List,
                 after_scan_levels: List, threats: List, est_human_weights: Dict, reward_fun: RewardsBase,
                 hum_mod='bounded_rational',
                 df=0.9, kappa=0.05, hl=10.0, tc=10.0):

        super().__init__(N, rob_weights, trust_params, prior_levels, after_scan_levels, threats, est_human_weights,
                         reward_fun, hum_mod, df, kappa)
        self.hl = hl
        self.tc = tc

    def __get_immediate_reward(self, house, action, wh, wc):

        r1 = -wc * self.tc
        r2 = -wh * self.hl
        r3 = 0

        if action:
            r_follow = r1
            if self.threats[house] == 1:
                r_not_follow = r2
            else:
                r_not_follow = r3
        else:
            r_not_follow = r1
            if self.threats[house] == 1:
                r_follow = r2
            else:
                r_follow = r3

        return r_follow, r_not_follow

    def get_immediate_reward_rob(self, current_house, action):
        return self.__get_immediate_reward(current_house, action, self.wh, self.wc)

    def get_immediate_reward_hum(self, current_house, action):
        return self.__get_immediate_reward(current_house, action, self.wh_hum, self.wc_hum)

    def get_recommendation(self, current_house, posterior: Posterior):

        alpha_0 = self.trust_params[0]
        beta_0 = self.trust_params[1]
        ws = self.trust_params[2]
        wf = self.trust_params[3]

        ns = np.sum(self.performance_history)
        nf = len(self.performance_history) - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        num_houses_to_go = self.N - current_house

        #                         stages                  successes
        value_matrix = np.zeros((num_houses_to_go + 1, num_houses_to_go + 1), dtype=float)  # Extra stage of value zero
        action_matrix = np.zeros((num_houses_to_go, num_houses_to_go + 1), dtype=int)

        # Give more info at current house
        self.threat_levels[current_house] = self.after_scan_levels[current_house]

        # Going backwards in stages
        for t in reversed(range(num_houses_to_go)):

            # Possible vals at stage t
            possible_alphas = alpha_previous + np.arange(t + 1) * ws
            possible_betas = beta_previous + (t - np.arange(t + 1)) * wf

            if self.hum_mod == 'disuse' or self.hum_mod == 'bounded_rational':

                # Estimated expected immediate rewards for human for choosing to NOT USE and USE RARV respectively
                self.wh_hum = posterior.get_mean()
                self.wc_hum = 1. - self.wh_hum
                r0_hum = -self.wh_hum * self.hl * self.threat_levels[t + current_house]
                r1_hum = -self.wc_hum * self.tc

                if self.hum_mod == 'bounded_rational':
                    # Probability of NOT USING RARV (Proportional to)
                    p0 = np.exp(self.kappa * r0_hum)
                    # Probability of USING RARV (Proportional to)
                    p1 = np.exp(self.kappa * r1_hum)

                    # Normalizing
                    p0 /= (p0 + p1)
                    p1 = 1. - p0

            for i, alpha in enumerate(possible_alphas):

                beta = possible_betas[i]
                trust = alpha / (alpha + beta)

                phl = 0.
                pcl = 0.
                ptl = 0.

                ######### CASE 1: Expected reward-to-go to recommend to NOT USE RARV ###########
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

                # probability of trust loss
                ptl = int(r0_hum < r1_hum)
                pti = 1. - ptl
                trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                # Trust increase
                next_stage_reward = pti * value_matrix[t + 1, i + 1]

                # Trust decrease
                next_stage_reward += ptl * value_matrix[t + 1, i]
                r0 += self.df * next_stage_reward + trust_gain_reward

                ############### Expected reward to go to recommend to USE RARV #############
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
                    # Probabilit of USING RARV
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

                # Probability of trust loss
                ptl = int(r0_hum > r1_hum)

                # Probability of trust increase
                pti = 1. - ptl

                # Expected immediate reward to recommend to USE RARV
                r1 = -phl * self.wh * self.hl - pcl * self.wc * self.tc

                trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                # Trust increase
                next_stage_reward = pti * value_matrix[t + 1, i + 1]

                # Trust decrease
                next_stage_reward += ptl * value_matrix[t + 1, i]

                r1 += self.df * next_stage_reward + trust_gain_reward

                action_matrix[t, i] = int(r1 > r0)
                value_matrix[t, i] = max(r1, r0)

                # import pdb; pdb.set_trace()

        return action_matrix[0, 0]

    def forward(self, current_house, rec, posterior: Posterior):

        self.wh_hum = posterior.get_mean()
        self.wc_hum = 1. - self.wh_hum

        rew2use = -self.wc_hum * self.tc
        rew2notuse = -self.wh_hum * self.threats[current_house] * self.hl

        if rec:
            if rew2use >= rew2notuse:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)
        else:
            if rew2notuse >= rew2use:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)


class SolverOnlyEndReward(Solver):

    def __init__(self, N, wh, wc, wt, params_list, prior_levels, after_scan_levels, threats, est_human_weights, hl=10,
                 tc=10, df=0.9, hum_mod='rev_psych', reward_fun='linear'):
        super().__init__(N, wh, wc, wt, params_list, prior_levels, after_scan_levels, threats, est_human_weights, hl,
                         tc, df, hum_mod, reward_fun)

    def get_action(self, current_house, current_health, current_time, params, posterior=None):

        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = np.sum(self.performance_history)
        nf = len(self.performance_history) - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        self.health = current_health
        self.time = current_time

        i = current_house
        n = self.N

        num_houses_to_go = n - i
        # stages                  successes         healths           times
        value_matrix = np.zeros(
            (num_houses_to_go + 1, num_houses_to_go + 1, num_houses_to_go + 1, num_houses_to_go + 1),
            dtype=float)  # Extra stage of value zero
        action_matrix = np.zeros((num_houses_to_go, num_houses_to_go + 1, num_houses_to_go + 1, num_houses_to_go + 1),
                                 dtype=int)

        # Give more info at current house
        self.threat_levels[i] = self.after_scan_levels[i]

        # Going backwards in time
        for t in reversed(range(num_houses_to_go)):

            # Possible vals at stage t
            possible_alphas = alpha_previous + np.arange(t + 1) * ws
            possible_betas = beta_previous + (t - np.arange(t + 1)) * wf
            possible_healths = current_health - np.arange(t + 1) * 10
            possible_times = current_time + np.arange(t + 1) * 10

            for i, alpha in enumerate(possible_alphas):
                beta = possible_betas[i]
                trust = alpha / (alpha + beta)

                for j, h in enumerate(possible_healths):
                    for k, c in enumerate(possible_times):

                        phl = 0.
                        pcl = 0.
                        ptl = 0.

                        if posterior is not None:
                            self.est_human_weights['health'] = posterior.get_mean()
                            self.est_human_weights['time'] = 1 - self.est_human_weights['health']

                        # Estimated expected immediate rewards for human for choosing to NOT USE and USE RARV respectively
                        r0_hum = self.est_human_weights['health'] * self.health_loss_reward(h) * self.threat_levels[t]
                        r1_hum = self.est_human_weights['time'] * self.time_loss_reward(c)

                        ######### CASE 1: Expected reward-to-go to recommend to NOT USE RARV ###########
                        if self.hum_mod == 'rev_psych':
                            # probability of health loss
                            # Probability of NOT USING RARV * probability of threat
                            phl = trust * self.threat_levels[t]

                            # probability of time loss
                            # Probability of USING RARV
                            pcl = 1 - trust

                        elif self.hum_mod == 'disuse':
                            # probability of health loss
                            # Probability of NOT USING RARV * Probability of Threat Presence
                            phl = (trust + (1 - trust) * int(r0_hum > r1_hum)) * self.threat_levels[t]

                            # probability of time loss
                            # Probability of using RARV
                            pcl = (1 - trust) * int(r1_hum > r0_hum)
                        elif self.hum_mod == 'bounded_rational':
                            # Probability of health loss
                            # Probability of NOT USING RARV (Proportional to)
                            p0 = np.exp(r0_hum)
                            # Probability of USING RARV (Proportional to)
                            p1 = np.exp(r1_hum)

                            # Normalizing
                            p0 /= (p0 + p1)
                            p1 = 1 - p0

                            # Probability of NOT USING RARV * Probability of Threat Presence
                            phl = (trust + (1 - trust) * p0) * self.threat_levels[t]

                            # Probability of time loss
                            # Probability of using RARV
                            pcl = (1 - trust) * p1

                        else:
                            raise "Human model incorrectly specified"

                        # Expected immediate reward to recommend to not use RARV
                        if t == num_houses_to_go - 1:
                            r0 = phl * self.wh * self.health_loss_reward(h) + pcl * self.wc * self.time_loss_reward(c)
                            trust_gain_reward = self.wt * (alpha / (alpha + beta))
                        else:
                            r0 = 0
                            trust_gain_reward = 0

                        # probability of trust loss
                        ptl = int(r0_hum < r1_hum)
                        pti = 1 - ptl
                        # trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                        # Trust increase, health loss, no time loss + Trust increase, no health loss, time loss + Trust increase, no health loss, no time loss
                        next_stage_reward = pti * (
                                phl * (1 - pcl) * value_matrix[t + 1, i + 1, j + 1, k] + pcl * (1 - phl) *
                                value_matrix[t + 1, i + 1, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[
                                    t + 1, i + 1, j, k])

                        # Trust decrease, health loss, no time loss + Trust decrease, no health loss, time loss + Trust decrease, no health loss, no time loss
                        next_stage_reward += ptl * (
                                phl * (1 - pcl) * value_matrix[t + 1, i, j + 1, k] + pcl * (1 - phl) * value_matrix[
                            t + 1, i, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[t + 1, i, j, k])

                        r0 += next_stage_reward + trust_gain_reward

                        ############### Expected reward to go to recommend to USE RARV #############
                        if self.hum_mod == "rev_psych":
                            # Probability of losing health
                            phl = (1 - trust) * self.threat_levels[t]
                            # Probability of losing time
                            pcl = trust

                        elif self.hum_mod == "disuse":
                            # Probability of losing health
                            # Probability of NOT USING RARV * probability of threat presence
                            phl = (1 - trust) * int(r0_hum > r1_hum) * self.threat_levels[t]

                            # Probability of losing time
                            # Probabilit of USING RARV
                            pcl = trust + (1 - trust) * int(r1_hum > r0_hum)

                        elif self.hum_mod == 'bounded_rational':
                            # Probability of health loss
                            # Probability of NOT USING RARV (Proportional to)
                            p0 = np.exp(r0_hum)
                            # Probability of USING RARV (Proportional to)
                            p1 = np.exp(r1_hum)

                            # Normalizing
                            p0 /= (p0 + p1)
                            p1 = 1 - p0

                            # Probability of NOT USING RARV * Probability of Threat Presence
                            phl = (1 - trust) * p0 * self.threat_levels[t]

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
                        if t == num_houses_to_go - 1:
                            r1 = phl * self.wh * self.health_loss_reward(h) + pcl * self.wc * self.time_loss_reward(c)
                            trust_gain_reward = self.wt * (alpha / (alpha + beta))
                        else:
                            r1 = 0
                            trust_gain_reward = 0

                        # trust_gain_reward = pti * self.wt * np.sqrt(num_houses_to_go - t)

                        # Trust increase, health loss, no time loss + Trust increase, no health loss, time loss + Trust increase, no health loss, no time loss
                        next_stage_reward = pti * (
                                phl * (1 - pcl) * value_matrix[t + 1, i + 1, j + 1, k] + pcl * (1 - phl) *
                                value_matrix[t + 1, i + 1, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[
                                    t + 1, i + 1, j, k])

                        # Trust decrease, health loss, no time loss + Trust deccrease, no health loss, time loss + Trust decrease, no health loss, no time loss
                        next_stage_reward += ptl * (
                                phl * (1 - pcl) * value_matrix[t + 1, i, j + 1, k] + pcl * (1 - phl) * value_matrix[
                            t + 1, i, j, k + 1] + (1 - phl) * (1 - pcl) * value_matrix[t + 1, i, j, k])

                        r1 += next_stage_reward + trust_gain_reward

                        action_matrix[t, i, j, k] = int(r1 > r0)
                        value_matrix[t, i, j, k] = max(r1, r0)

        # import pdb; pdb.set_trace()
        return action_matrix[0, 0, 0, 0]


class SolverConstantRewardsNew(Solver):

    def __init__(self, num_sites: int, rob_weights: Dict, trust_params: List, prior_levels: List,
                 after_scan_levels: List, threats: List, est_human_weights: Dict, reward_fun: RewardsBase,
                 hum_mod='bounded_rational',
                 df=0.9, kappa=0.05, hl=10.0, tc=10.0):

        super().__init__(num_sites, rob_weights, trust_params, prior_levels, after_scan_levels, threats,
                         est_human_weights, reward_fun, hum_mod, df, kappa)
        self.hl = hl
        self.tc = tc

    def __get_immediate_reward(self, house, action, wh, wc):

        r1 = -wc * self.tc
        r2 = -wh * self.hl
        r3 = 0

        r_follow = action * r1 + (1 - action) * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)
        r_not_follow = (1 - action) * r1 + action * (self.threats[house] * r2 + (1 - self.threats[house]) * r3)

        return r_follow, r_not_follow

    def get_immediate_reward_rob(self, current_house, action):
        return self.__get_immediate_reward(current_house, action, self.wh, self.wc)

    def get_immediate_reward_hum(self, current_house, action):
        return self.__get_immediate_reward(current_house, action, self.wh_hum, self.wc_hum)

    def get_recommendation(self, current_house, posterior: Posterior):

        alpha_0 = self.trust_params[0]
        beta_0 = self.trust_params[1]
        ws = self.trust_params[2]
        wf = self.trust_params[3]

        ns = np.sum(self.performance_history)
        nf = len(self.performance_history) - ns

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

            # Possible vals at stage t
            possible_alphas = alpha_previous + np.arange(t + 1) * ws
            possible_betas = beta_previous + (t - np.arange(t + 1)) * wf

            # Compute some extra values if the human model is disuse or bounded rational
            if self.hum_mod == 'disuse' or self.hum_mod == 'bounded_rational':

                # Estimated expected immediate rewards for human for choosing to NOT USE and USE RARV respectively
                self.wh_hum = posterior.get_mean()
                self.wc_hum = 1. - self.wh_hum

                # The below are expected rewards based on the threat level
                r0_hum = -self.wh_hum * self.hl * self.threat_levels[t + current_house]
                r1_hum = -self.wc_hum * self.tc

                # The below are actual observable rewards based on threat presence
                r0_no_threat = 0
                r0_threat = -self.wh_hum * self.hl

                if self.hum_mod == 'bounded_rational':
                    # Probability of NOT USING RARV (Proportional to)
                    p0 = np.exp(self.kappa * r0_hum)
                    # Probability of USING RARV (Proportional to)
                    p1 = np.exp(self.kappa * r1_hum)

                    # Normalizing
                    p0 /= (p0 + p1)
                    p1 = 1. - p0

            for i, alpha in enumerate(possible_alphas):

                beta = possible_betas[i]
                trust = alpha / (alpha + beta)

                phl = 0.
                pcl = 0.
                ptl = 0.

                ######### CASE 1: Expected reward-to-go to recommend to NOT USE RARV ###########
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

                ############### Expected reward to go to recommend to USE RARV #############
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
                    # Probabilit of USING RARV
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

    def forward(self, current_house, rec, posterior: Posterior):

        self.wh_hum = posterior.get_mean()
        self.wc_hum = 1. - self.wh_hum

        rew2use = -self.wc_hum * self.tc
        rew2notuse = -self.wh_hum * self.threats[current_house] * self.hl

        if rec:
            if rew2use >= rew2notuse:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)
        else:
            if rew2notuse >= rew2use:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)

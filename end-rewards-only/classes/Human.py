from classes.Rewards import RewardsBase
from typing import Dict
import numpy as np
from classes.IRLModel import Posterior
from classes.ParamsUpdater import Estimator


class HumanBase:

    def __init__(self, posterior: Posterior,
                 reward_fun: RewardsBase,
                 trust_params: Dict,
                 num_sites: int,
                 seed: int = 123,
                 health: float = 100.,
                 time_: float = 0.,
                 health_loss: float = 10.,
                 time_loss: float = 10.):
        """
        Initializes the base human class. The human class maintains a level of trust.
        It chooses action based on the recommendation, trust, the behavior model
        :param posterior: The posterior distribution on the health reward weight for the human
        :param reward_fun: The reward function
        :param trust_params: The true trust params of this human - a dict with keys alpha0, beta0, ws, wf
        :param num_sites: The number of sites in the mission
        :param seed: A seed for the random number generator used to sample trust and behavior
        :param health: the starting health of the human
        :param time_: the time spent in the simulation
        :param health_loss: the amount of health lost after encountering threat without protection
        :param time_loss: the amount of time lost in using the armored robot
        """
        self.posterior = posterior
        self.wh = posterior.mean()
        self.wc = 1. - self.wh
        self.reward_fun = reward_fun
        self.trust_params = trust_params
        self.recommendation = None
        self.action = None
        self.rng = np.random.default_rng(seed=seed)

        self.health = health
        self.time_ = time_

        self.health_loss = health_loss
        self.time_loss = time_loss

        # Storage
        self.current_site = 0
        self.N = num_sites

        self.recommendation_history = np.zeros((self.N,), dtype=int)
        self.performance_history = np.zeros((self.N,), dtype=int)
        self.action_history = np.zeros_like(self.performance_history)

        self.trust_sample_history = np.zeros((self.N+1,), dtype=float)
        self.trust_mean_history = np.zeros_like(self.trust_sample_history)

        self.health_history = np.zeros((self.N+1,), dtype=float)
        self.health_history[0] = self.health

        self.time_history = np.zeros((self.N+1,), dtype=float)
        self.time_history[0] = self.time_

        self.trust_parameter_history = {'alpha0': np.zeros((self.N+1,), dtype=float),
                                        'beta0': np.zeros((self.N+1,), dtype=float),
                                        'ws': np.zeros((self.N+1,), dtype=float),
                                        'wf': np.zeros((self.N+1,), dtype=float)}

    def choose_action(self, recommendation: int, threat_level: float):
        """
        A function to choose the action given the recommendation and the threat level, and the maintained human trust
        :param recommendation: The recommendation given by the robot
        :param threat_level: The threat level reported by the drone
        :return: action: the action chosen
        """
        trust = self.sample_trust()
        self.recommendation = recommendation
        self.action = None
        raise NotImplementedError

    def sample_trust(self):
        """
        Samples a value of trust from the beta distribution
        :return: trust_sample: a sampled value of trust
        """
        num_successes = self.performance_history.sum()
        num_failures = self.current_site + 1 - num_successes

        alpha = self.trust_params['alpha0'] + self.trust_params['ws'] * num_successes
        beta = self.trust_params['beta0'] + self.trust_params['wf'] * num_failures

        trust_sample = self.rng.beta(alpha, beta)

        return trust_sample

    def add_initial_trust(self):
        """
        Adds an initial value of trust to the trust history.
        This is assumed to be the trust before any interaction. Should not be used more than once
        """
        trust = self.sample_trust()
        self.trust_sample_history[0] = trust

        trust_mean = self.trust_params['alpha0'] / (self.trust_params['alpha0'] + self.trust_params['beta0'])
        self.trust_mean_history[0] = trust_mean

        return trust

    def mean_trust(self):
        """
        Gives the mean of the beta distribution modeling the trust
        :return: trust_mean
        """
        num_successes = self.performance_history.sum()
        num_failures = self.current_site + 1 - num_successes

        alpha = self.trust_params['alpha0'] + self.trust_params['ws'] * num_successes
        beta = self.trust_params['beta0'] + self.trust_params['wf'] * num_failures

        return alpha / (alpha + beta)

    def forward(self, threat_obs: int, action: int):
        """
        Updates the performance history based on immediate observed rewards
        Also updates the health and time based on the action chosen
        :param threat_obs: the observed value of threat presence
        :param action: the action chosen by the human
        :return: trust_sample: a sampled value of trust
        """

        # Update the performance history
        hl, tc = self.reward_fun.reward(0.0, 0.0, 0)
        reward_0 = self.wh * hl * threat_obs  # Negative if threat observed, zero if not observed
        reward_1 = self.wc * tc

        if self.recommendation == 1:
            if reward_1 >= reward_0:
                self.performance_history[self.current_site] = 1
            else:
                self.performance_history[self.current_site] = 0
        else:
            if reward_1 >= reward_0:
                self.performance_history[self.current_site] = 0
            else:
                self.performance_history[self.current_site] = 1

        # Update the action history
        self.action_history[self.current_site] = action

        # Update the site number
        self.current_site += 1

        # Update health and time
        if action == 1:
            self.time_ += self.time_loss
        else:
            if threat_obs:
                self.health -= self.health_loss

        # Add this to the history
        self.time_history[self.current_site] = self.time_
        self.health_history[self.current_site] = self.health

        # Sample trust and add it to the history
        trust = self.sample_trust()
        self.trust_sample_history[self.current_site] = trust

        trust_mean = self.mean_trust()
        self.trust_mean_history[self.current_site] = trust_mean

        return trust

    def current_health(self):
        return self.health

    def current_time(self):
        return self.time_

    def get_alphabeta(self):

        ns = self.performance_history.sum()
        _alpha = self.trust_params['alpha0'] + ns * self.trust_params['ws']
        _beta = self.trust_params['beta0'] + (self.current_site + 1 - ns) * self.trust_params['wf']

        return _alpha, _beta

    def update_posterior(self, threat_level: float, trust: float):
        """
        Updates the posterior distribution on the health reward weight for the human
        ONLY use it for the model maintained by the robot.
        DO NOT update the posterior when simulating the human
        :param threat_level: The threat level reported by the drone AFTER SCANNING
        :param trust: the level of trust reported by the human BEFORE CHOOSING AN ACTION
        """
        raise NotImplementedError

    def get_rewards(self, threat_level: float):
        """
        A function to return the rewards for all possible actions
        :param threat_level: The threat level reported by the drone
        :return: reward_0, reward_1: The rewards for not using and using the armored robot respectively
        """

        hl, tc = self.reward_fun.reward(0.0, 0.0, 0)
        reward_0 = threat_level * hl * self.wh
        reward_1 = tc * self.wc

        return reward_0, reward_1

    def get_action_probabilities(self, trust: float, recommendation: int, threat_level: float):
        """
        Gives the probabilities of choosing action 0 and action 1 respectively
        :param trust: the current level of trust (BEFORE CHOOSING AN ACTION)
        :param recommendation: the recommendation given by the robot
        :param threat_level: the threat level in the site (either reported by the drone after scanning or a prior level
                of threat inside the site, depending on the site number)
        :return: prob0, prob1: the probabilities of choosing action 0 and action 1 respectively
        """

        reward0, reward1 = self.get_rewards(threat_level)
        p0 = 1. / (1. + np.exp(reward1 - reward0))
        p1 = 1. - p0

        if recommendation == 1:
            prob1 = trust + (1. - trust) * p1
            prob0 = (1. - trust) * p0
        else:
            prob1 = (1. - trust) * p1
            prob0 = trust + (1. - trust) * p0

        return prob0, prob1

    def get_trust_probabilities(self, threat_level: float, recommendation: int, w_star: float):
        """
        Gives the probabilities of increasing trust, given the threat level and the recommendation
        :param threat_level: the threat level (either after or before scanning)
        :param recommendation: the recommendation given by the robot
        :param w_star: the critical health weight above which the human cares about losing health
        :return
        """
        raise NotImplementedError

    def update_params(self, trust_fb: float, first: bool = False):
        """
        Updates the estimated trust parameters of the human
        :param trust_fb: the trust feedback given at the current site
        :param first: if true, the feedback is given BEFORE ANY INTERACTION with the robot (inherent trust), else
                      it is the trust after searching a site (after observing performance)
        """
        raise NotImplementedError


class DisuseBoundedRationalSimulator(HumanBase):
    """
    This class should be used for simulating the human's choice
    """

    def __init__(self, posterior: Posterior,
                 kappa: float,
                 reward_fun: RewardsBase,
                 trust_params: Dict,
                 num_sites: int,
                 seed: int = 123,
                 health: float = 100.,
                 time_: float = 0.,
                 health_loss: float = 10.,
                 time_loss: float = 10.):
        """
        Initializes the base human class. The human class maintains a level of trust.
        It chooses action based on the recommendation, trust, the behavior model
        :param posterior: The distribution on the health reward weight for the human
        :param kappa: The rationality coefficient
        :param reward_fun: The reward function
        :param trust_params: The true trust params of this human - a dict with keys alpha0, beta0, ws, wf
        :param num_sites: The number of sites in the mission
        :param seed: A seed for the random number generator used to sample trust and behavior
        :param health: the starting health of the human
        :param time_: the time spent in the simulation
        :param health_loss: the amount of health lost after encountering threat without protection
        :param time_loss: the amount of time lost in using the armored robot
        """
        super().__init__(posterior, reward_fun, trust_params, num_sites, seed, health, time_, health_loss, time_loss)
        self.kappa = kappa

    def choose_action(self, recommendation: int, threat_level: float):
        """
        A function to choose the action given the recommendation and the threat level, and the maintained human trust
        :param recommendation: The recommendation given by the robot
        :param threat_level: The threat level reported by the drone
        :return: action: the action chosen
        """
        trust = self.sample_trust()

        # In case using the recommendation
        if self.rng.uniform() < trust:
            return recommendation

        # In case disusing the recommendation
        hl, tc = self.reward_fun.reward(0.0, 0.0, 0)
        reward_0 = threat_level * hl * self.wh
        reward_1 = tc * self.wc
        prob_0 = 1. / (1 + np.exp(self.kappa * (reward_1 - reward_0)))

        if self.rng.uniform() < prob_0:
            return 0

        return 1


class DisuseBoundedRationalModel(HumanBase):
    """
    This class should be used for modeling the human by the robot. It will maintain the posterior,
    the state of the human (health, time, trust, etc.)
    """

    def __init__(self, posterior: Posterior,
                 kappa: float,
                 reward_fun: RewardsBase,
                 trust_params: Dict,
                 num_sites: int,
                 params_updater: Estimator,
                 seed: int = 123,
                 health: float = 100.,
                 time_: float = 0.,
                 health_loss: float = 10.,
                 time_loss: float = 10.):
        """
        Initializes the base human class. The human class maintains a level of trust.
        It chooses action based on the recommendation, trust, the behavior model
        :param posterior: The distribution on the health reward weight of the human
        :param kappa: The rationality coefficient
        :param reward_fun: The reward function
        :param trust_params: The estimated trust params of this human - a dict with keys alpha0, beta0, ws, wf
        :param num_sites: The number of sites in the mission
        :param params_updater: The class that updates trust parameters after receiving feedback
        :param seed: A seed for the random number generator used to sample trust and behavior
        :param health: the starting health of the human
        :param time_: the time spent in the simulation
        :param health_loss: the amount of health lost after encountering threat without protection
        :param time_loss: the amount of time lost in using the armored robot
        """

        super().__init__(posterior, reward_fun, trust_params, num_sites, seed, health, time_, health_loss, time_loss)
        self.kappa = kappa
        self.params_updater = params_updater

    def choose_action(self, recommendation: int, threat_level: float):
        """This should NOT be used. This class only models the human for the robot.
        It does not simulate choosing an action"""
        return -1

    def get_rewards(self, threat_level: float):
        """
        A function to return the rewards for all possible actions
        :param threat_level: The threat level reported by the drone
        :return: reward_0, reward_1: The rewards for not using and using the armored robot respectively
        """

        hl, tc = self.reward_fun.reward(0.0, 0.0, 0)
        reward_0 = threat_level * hl * self.wh
        reward_1 = tc * self.wc

        return reward_0, reward_1

    def update_posterior(self, threat_level: float, trust:float):
        """
        Updates the posterior distribution on the health reward weight for the human
        ONLY use it for the model maintained by the robot.
        DO NOT update the posterior when simulating the human
        :params threat_level: the threat level reported by the drone AFTER SCANNING
        :params trust: the level of trust reported by the human BEFORE CHOOSING AN ACTION
        """
        rec = self.recommendation_history[self.current_site - 1]
        action = self.action_history[self.current_site - 1]

        # Here, -1 gives the previous trust feedback. This makes sure that the trust we are using is from before making
        # the choice and not the value after making the action choice
        # trust = self.trust_history[self.current_site - 1]
        health = self.health_history[self.current_site - 1]
        time_ = self.time_history[self.current_site - 1]

        self.posterior.update(rec, action, trust, health, time_, threat_level, self.current_site - 1)

    def get_trust_probabilities(self, threat_level: float, recommendation: int, w_star: float):
        """
        Gives the probability of increasing trust, given the threat level and the recommendation
        :param threat_level: the threat level (either after or before scanning)
        :param recommendation: the recommendation given by the robot
        :param w_star: the critical health weight above which the human cares about losing health
        :return prob: the probability of increasing trust
        """

        if recommendation == 0:
            #       No threat           human cares about losing health    human does not care about losing health
            prob = (1 - threat_level) * (1 - self.posterior.cdf(w_star)) + self.posterior.cdf(w_star)
            return prob

        # else (if recommendation = 1)
        #       Threat present   Human cares about losing health
        prob = threat_level * self.posterior.cdf(w_star)

        return prob

    def update_params(self, trust_fb: float, first: bool = False):
        """
        Updates the estimated trust parameters of the human
        :param trust_fb: the trust feedback given at the current site
        :param first: if true, the feedback is given BEFORE ANY INTERACTION with the robot (inherent trust), else
                      it is the trust after searching a site (after observing performance)
        """

        if first:
            self.trust_params = self.params_updater.get_initial_guess(trust_fb)
            for k, v in self.trust_params.items():
                self.trust_parameter_history[k][0] = v
        else:
            self.trust_params = self.params_updater.get_params(self.trust_params,
                                                               self.performance_history[self.current_site-1],
                                                               trust_fb,
                                                               self.current_site)
            for k, v in self.trust_params.items():
                self.trust_parameter_history[k][self.current_site] = v

from classes.Rewards import RewardsBase
from typing import Dict
import numpy as np
from classes.IRLModel import Posterior


class HumanBase:

    def __init__(self, whh: float,
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
        :param whh: The health reward weight for the human
        :param reward_fun: The reward function
        :param trust_params: The true trust params of this human - a dict with keys alpha0, beta0, ws, wf
        :param seed: A seed for the random number generator used to sample trust and behavior
        :param health: the starting health of the human
        :param time_: the time spent in the simulation
        :param health_loss: the amount of health lost after encountering threat without protection
        :param time_loss: the amount of time lost in using the armored robot
        """

        self.wh = whh
        self.wc = 1. - whh
        self.reward_fun = reward_fun
        self.trust_params = trust_params
        self.recommendation = None
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
        self.trust_history = np.zeros((self.N+1,), dtype=float)
        self.health_history = np.zeros((self.N+1,), dtype=float)
        self.time_history = np.zeros((self.N+1,), dtype=float)
        self.health_history[0] = self.health
        self.time_history[0] = self.time_

    def choose_action(self, recommendation: int, threat_level: float):
        """
        A function to choose the action given the recommendation and the threat level, and the maintained human trust
        :param recommendation: The recommendation given by the robot
        :param threat_level: The threat level reported by the drone
        :return: action: the action chosen
        """
        trust = self.sample_trust()
        self.recommendation = recommendation
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

    def forward(self, threat_obs: int):
        """
        Updates the performance history based on immediate observed rewards
        Also updates the health and time based on the action chosen
        :param threat_obs: the observed value of threat presence
        :return: trust_sample: a sampled value of trust
        """
        i = self.current_site
        hl, tc = self.reward_fun.reward(0.0, 0.0, 0)
        reward_0 = self.wh * hl * threat_obs  # Negative if threat observed, zero if not observed
        reward_1 = self.wc * tc

        if self.recommendation == 1:
            if reward_1 >= reward_0:
                self.performance_history[i] = 1
            else:
                self.performance_history[i] = 0
        else:
            if reward_1 >= reward_0:
                self.performance_history[i] = 1
            else:
                self.performance_history[i] = 0

        self.current_site += 1
        return self.sample_trust()


class DisuseBoundedRationalSimulator(HumanBase):
    """
    This class should be used for simulating the human's choice
    """

    def __init__(self, whh: float, kappa: float, reward_fun: RewardsBase, trust_params: Dict):
        """
        Initializes the base human class. The human class maintains a level of trust.
        It chooses action based on the recommendation, trust, the behavior model
        :param whh: The health reward weight for the human
        :param kappa: The rationality coefficient
        :param reward_fun: The reward function
        :param trust_params: The true trust params of this human - a dict with keys alpha0, beta0, ws, wf
        """
        super().__init__(whh, reward_fun, trust_params)
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

    def __init__(self, posterior: Posterior, kappa: float, reward_fun: RewardsBase, trust_params: Dict):
        """
        Initializes the base human class. The human class maintains a level of trust.
        It chooses action based on the recommendation, trust, the behavior model
        :param posterior: The distribution on the health reward weight of the human
        :param kappa: The rationality coefficient
        :param reward_fun: The reward function
        :param trust_params: The estimated trust params of this human - a dict with keys alpha0, beta0, ws, wf
        """
        super().__init__(posterior.mean(), reward_fun, trust_params)
        self.posterior = posterior
        self.kappa = kappa

    def get_rewards(self, recommendation: int, threat_level: float):
        """
        A function to return the rewards for all possible actions
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

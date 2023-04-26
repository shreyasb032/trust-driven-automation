from classes.Rewards import RewardsBase
from typing import Dict
import numpy as np


class HumanBase:

    def __init__(self, whh: float,
                 reward_fun: RewardsBase,
                 trust_params: Dict,
                 seed: int = 123):
        """
        Initializes the base human class. The human class maintains a level of trust.
        It chooses action based on the recommendation, trust, the behavior model
        :param whh: The health reward weight for the human
        :param reward_fun: The reward function
        :param trust_params: The true trust params of this human - a dict with keys alpha0, beta0, ws, wf
        :param seed: A seed for the random number generator used to sample trust and behavior
        """

        self.wh = whh
        self.wc = 1. - whh
        self.reward_fun = reward_fun
        self.trust_params = trust_params
        self.performance_history = []
        self.recommendation = None
        self.rng = np.random.default_rng(seed=seed)

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
        num_successes = np.sum(self.performance_history)
        num_failures = len(self.performance_history) - num_successes

        alpha = self.trust_params['alpha0'] + self.trust_params['ws'] * num_successes
        beta = self.trust_params['beta0'] + self.trust_params['wf'] * num_failures

        trust_sample = self.rng.beta(alpha, beta)

        return trust_sample

    def update_trust(self, threat_obs: int):
        """
        Updates the performance history based on immediate observed rewards
        :param threat_obs: the observed value of threat presence
        :return: trust_sample: a sampled value of trust
        """
        hl, tc = self.reward_fun.reward()
        reward_0 = self.wh * hl * threat_obs  # Negative if threat observed, zero if not observed
        reward_1 = self.wc * tc

        if self.recommendation == 1:
            if reward_1 >= reward_0:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)
        else:
            if reward_1 >= reward_0:
                self.performance_history.append(0)
            else:
                self.performance_history.append(1)

        return self.sample_trust()


class DisuseBoundedRational(HumanBase):

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
        pass

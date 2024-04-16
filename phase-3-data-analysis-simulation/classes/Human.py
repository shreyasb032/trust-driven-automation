"""
Given a query, this file generates an answer to that query
"""
import numpy as np
from numpy.random import default_rng


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def clamp(x, _min, _max):
    return min(_max, max(_min, x))


class HumanBase:
    """
    A Base class for all children classes that have a specific function giving their reward weights given the state
    """

    def __init__(self, seed: int = 123, kappa: float = 0.2, hc: float = 10., tc: float = 10.,
                 min_wh: float = 0.02, max_wh: float = 0.98):
        self.rng = default_rng(seed)
        self.kappa = kappa
        self.weights_func = None
        self.hc = 10.
        self.tc = 10.
        self.wh = None
        self.wc = None
        self.min_wh = min_wh
        self.max_wh = max_wh

    def choose(self, state: np.ndarray, threat_chance: int):
        """
        Chooses an action given the current state and chance of threat presence
        :param state: numpy row vector [health, time]
        :param threat_chance: the chance of threat presence in percent
        :return:
        """
        raise NotImplementedError

    def get_wh(self, state):
        raise NotImplementedError


class SigmoidHuman(HumanBase):
    """
    Uses the sigmoid function with states as the input to select reward weights
    """
    def __init__(self, seed: int = 123, kappa: float = 0.2, hc: float = 10.0, tc: float = 10.0,
                 min_wh: float = 0.02, max_wh: float = 0.98, noise: bool = False):
        super().__init__(seed, kappa, hc, tc, min_wh, max_wh)
        self.weights_func = sigmoid
        # Weights associated with the linear function generating the input to the sigmoid
        # The first is the weight associated with health and the second is that associated with time
        self.sigmoid_weights = np.array([[-0.01], [0.015]], dtype=float)
        self.add_noise = noise

    def choose(self, state: np.ndarray, threat_chance: int):
        """
        Chooses an action given the current state and chance of threat presence.
        This class uses the sigmoid function to generate weights.
        :param state: numpy row vector [health, time]
        :param threat_chance: the chance of threat presence in percent
        :return:
            action: the action chosen by the human at this state
        """

        x = state.dot(self.sigmoid_weights)
        self.wh = sigmoid(x[0, 0])
        # Add some noise each time a query is asked
        if self.add_noise:
            self.wh += self.rng.normal(loc=0., scale=0.1)

        # Ensure that the value is in [min_wh=0.02, max_wh=0.98]
        self.wh = clamp(self.wh, self.min_wh, self.max_wh)
        self.wc = 1. - self.wh

        threat_prob = threat_chance / 100.
        reward_0 = -self.wh * threat_prob * self.hc
        reward_1 = -self.wc * self.tc

        prob_0 = 1. / (1 + np.exp((reward_1 - reward_0) * self.kappa))
        prob_1 = 1. - prob_0

        return self.rng.choice([0, 1], p=[prob_0, prob_1])

    def get_wh(self, state):

        x = state.dot(self.sigmoid_weights)
        x = x[0, 0]
        return sigmoid(x)

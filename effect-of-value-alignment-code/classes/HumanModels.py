from typing import Dict, List
import numpy as np
from copy import copy
from numpy.random import beta
from classes.RewardFunctions import RewardsBase
from classes.PerformanceMetrics import PerformanceMetricBase


class HumanBase:

    """
    Base model for the simulated human. Other classes should inherit from this class and implement their own functions
    """
    
    def __init__(self, params: List, reward_weights: Dict, reward_fun: RewardsBase,
                 performance_metric: PerformanceMetricBase):
        """
        Initializes the human base class
        :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
        :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
        :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
        :param performance_metric: the performance metric that returns the performance given the recommendation and
                                    outcome
        """

        # Initializing the params and trust
        self.params = copy(params)
        self.trust = params[0] / (params[0] + params[1])
        self._alpha = params[0]
        self._beta = params[1]        
        
        # Saving a copy for resetting
        self.init_params = copy(params)
        
        # Reward weights (Dict with keys 'health' and 'time)        
        self.reward_weights = reward_weights
        
        # Reward function
        self.reward_fun = reward_fun
        
        # Performance metric
        self.performance_metric = performance_metric

        # Storage
        self.performance_history = []
    
    def set_params(self, params):
        """
        Updates the trust parameters of the human
        :param params: the trust parameters list [alpha0, beta0, ws, wf]
        """
        self.params = copy(params)
        
    def reset(self):
        """
        Resets the human model. NOT SURE WHY I AM NOT CLEARING THE PERFORMANCE HISTORY HERE.
        """
        self.params = copy(self.init_params)
        self.trust = self.params[0] / (self.params[0] + self.params[1])
        self._alpha = self.params[0]
        self._beta = self.params[1]
    
    def update_trust(self, rec, threat, threat_level, health=100, time=0):
        """Update trust based on immediate observed reward
        :param rec: recommendation given to the human
        :param threat: integer representing the presence of threat
        :param threat_level: a float representing the level of threat
        :param health: the current health level of the soldier
        :param time: the time spent in the mission"""

        perf = None

        if self.performance_metric.idx == 1:
            perf = self.performance_metric.get_performance(rec, health, time, threat)
        elif self.performance_metric.idx == 2:
            perf = self.performance_metric.get_performance(rec, health, time, threat_level)

        self.performance_history.append(perf)

        if perf == 1:
            self._alpha += self.params[2]
        else:
            self._beta += self.params[3]
        
        self.trust = self._alpha / (self._alpha + self._beta)

    def get_last_performance(self):
        """Returns the last value in the maintained performance history"""
        return self.performance_history[-1]

    def get_immediate_reward(self, health, time, action, threat):
        """
        Helper function to return the immediate observed reward for the given action
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        :param action: the action chosen/recommended
        :param threat: an integer representing the presence of threat
        """

        hl, tc = self.reward_fun.reward(health, time, house=None)

        r1 = self.reward_weights["time"] * tc
        r2 = self.reward_weights["health"] * hl
        r3 = 0
        
        r = 0

        if action:
            r = r1
        else:
            if threat == 1:
                r = r2
            else:
                r = r3

        return r
    
    def get_mean(self):
        """Returns the mean level of trust"""

        return self.trust
    
    def get_feedback(self):
        """Samples trust from the beta distribution"""

        return beta(self._alpha, self._beta)


class ReversePsychology(HumanBase):
    """The reverse psychology model of human behavior"""

    def __init__(self, params: List, reward_weights: Dict, reward_fun: RewardsBase,
                 performance_metric: PerformanceMetricBase):
        """
        Initializes this human model
        :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
        :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
        :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
        :param performance_metric: the performance metric that returns the performance given the recommendation and
                                    outcome
        """
        super().__init__(params, reward_weights, reward_fun, performance_metric)

    def choose_action(self, rec, threat_level=None, health=None, time=None):
        """
        Chooses an action based on the reverse psychology model
        :param rec: the recommendation given to the human
        :param threat_level: a float representing the level fo threat in the current site
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        """

        return np.random.choice([rec, 1-rec], p=[self.trust, 1-self.trust])


class Disuse(HumanBase):
    """Old Disuse Model: Accept recommendation with probability trust, choose the action which gives the best
        immediate expected reward otherwise"""

    def __init__(self, params: List, reward_weights: Dict, reward_fun: RewardsBase,
                 performance_metric: PerformanceMetricBase):
        """
        Initializes the human base class
        :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
        :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
        :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
        :param performance_metric: the performance metric that returns the performance given the recommendation and
                                    outcome
        """
        super().__init__(params, reward_weights, reward_fun, performance_metric)

    def choose_action(self, rec, threat_level, health, time):
        """
        Chooses an action based on the disuse model
        :param rec: the recommendation given to the human
        :param threat_level: a float representing the level fo threat in the current site
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        """

        p = np.random.uniform(0, 1)
        
        # With probability = trust, choose the recommendation
        if p < self.trust:
            return rec

        hl, tc = self.reward_fun.reward(health, time, house=None)
        
        # With probability = 1 - trust, choose the action that maximizes immediate expected reward
        r0 = self.reward_weights["health"] * threat_level * hl
        r1 = self.reward_weights["time"] * tc

        return int(r0 < r1)


class BoundedRational(HumanBase):
    """Bounded rationality with disuse model. Accepts recommendation with probability trust, 
    Chooses the action that gives the best immediate expected reward with probability proportional 
    to the exponent of that reward multiplied by the rationality coefficient"""

    def __init__(self, params, reward_weights, reward_fun: RewardsBase,
                 performance_metric: PerformanceMetricBase,
                 kappa=1.0):
        """
        Initializes the human base class
        :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
        :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
        :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
        :param performance_metric: the performance metric that returns the performance given the recommendation and
                                    outcome
        :param kappa: the rationality coefficient for the human
        """
        super().__init__(params, reward_weights, reward_fun, performance_metric)
        self.kappa = kappa

    def choose_action(self, rec, threat_level, health, time):
        """
        Chooses an action based on the bounded rationality disuse model
        :param rec: the recommendation given to the human
        :param threat_level: a float representing the level fo threat in the current site
        :param health: the current health level of the soldier
        :param time: the time spent in the mission
        """

        p = np.random.uniform(0, 1)
        
        # With probability = trust, choose the recommendation
        if p < self.trust:
            return rec

        # Else, compute the rewards for action 0 and action 1
        hl, tc = self.reward_fun.reward(health, time, house=None)
        r0 = self.reward_weights["health"] * threat_level * hl
        r1 = self.reward_weights["time"] * tc

        p0 = 1. / (1 + np.exp(r1 - r0))

        return np.random.choice([0, 1], p=[p0, 1-p0])


class AlwaysAccept(HumanBase):
    """Human model that always accepts the recommendation, regardless of the trust level"""
    
    def __init__(self, params):
        super().__init__(params)

    def choose_action(self, rec):
        return rec

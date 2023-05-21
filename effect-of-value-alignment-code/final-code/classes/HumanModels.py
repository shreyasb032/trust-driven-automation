from typing import Dict, List
import numpy as np
from copy import copy
from numpy.random import beta
from classes.RewardFunctions import RewardsBase
from classes.PerformanceMetrics import PerformanceMetricBase

class HumanBase:
    
    def __init__(self, params: List, reward_weights: Dict, reward_fun: RewardsBase, performance_metric: PerformanceMetricBase):

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
        self.params = copy(params)
        
    def reset(self):
        self.params = copy(self.init_params)
        self.trust = self.params[0] / (self.params[0] + self.params[1])
        self._alpha = self.params[0]
        self._beta = self.params[1]
    
    def update_trust(self, rec, threat, threat_level, health=100, time=0):
        """Update trust based on immediate actual reward"""

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
        return self.performance_history[-1]

    def get_immediate_reward(self, health, time, action, threat):

        hl, tc = self.reward_fun.reward(health, time)

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

        return self.trust
    
    def get_feedback(self):

        return beta(self._alpha, self._beta)

class ReversePsychology(HumanBase):
    
    def __init__(self, params, reward_weights):
        super().__init__(params)
        self.reward_weights = reward_weights

    def choose_action(self, rec, threat_level=None, health=None, time=None):

        return np.random.choice([rec, 1-rec], p=[self.trust, 1-self.trust])

class Disuse(HumanBase):
    """Old Disuse Model: Accept recommendation with probability trust, choose the action which gives best 
        immediate expected reward otherwise"""

    def __init__(self, params, reward_weights, reward_fun='linear'):
        super().__init__(params, reward_weights, reward_fun)
    
    def choose_action(self, rec, threat_level, health, time):
        
        p = np.random.uniform(0, 1)
        
        # With probability = trust, choose the recommendation
        if p < self.trust:
            return rec
        
        # With probability = 1 - trust, choose the action that maximizes immediate expected reward
        r0 = self.reward_weights["health"] * threat_level * self.health_loss_reward(health)
        r1 = self.reward_weights["time"] * self.time_loss_reward(time)

        return int(r0 < r1)

class BoundedRational(HumanBase):
    """Bounded rationality with disuse model. Accepts recommendation with probability trust, 
    Chooses the action that gives the best immediate expected reward with probability proportional 
    to the exponent of that reward multiplied by the rationality coefficient"""

    def __init__(self, params, reward_weights, reward_fun: RewardsBase, perf_metric: PerformanceMetricBase, kappa=1.0):
        super().__init__(params, reward_weights, reward_fun, perf_metric)
        self.kappa = kappa

    def choose_action(self, rec, threat_level, health, time):
        
        p = np.random.uniform(0, 1)
        
        # With probability = trust, choose the recommendation
        if p < self.trust:
            return rec

        # Else, compute the rewards for action 0 and action 1
        h, c = self.reward_fun.reward(health, time)
        r0 = self.reward_weights["health"] * threat_level * h
        r1 = self.reward_weights["time"] * c

        p0 = np.exp(r0)
        p1 = np.exp(r1)

        p0 /= (p0+p1)

        return np.random.choice([0, 1], p=[p0, 1-p0])

class AlwaysAccept(HumanBase):
    """Human model that always accepts the recommendation, regardless of the trust level"""
    
    def __init__(self, params):
        super().__init__(params)
    
    def choose_action(self, rec):
        return rec


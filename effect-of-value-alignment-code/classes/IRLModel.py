import numpy as np
from classes.RewardFunctions import RewardsBase


class Posterior:

    def __init__(self, kappa: float, stepsize: float, reward_fun: RewardsBase):
        """Initializer of the posterior class
        :param kappa: the rationality coefficient
        :param stepsize: the stepsize of the discrete distribution
        :param reward_fun: the function that gives the rewards
        """

        self.kappa = kappa
        self.stepsize = stepsize
        num_weights = int(1.0/stepsize) + 1

        # Initialize the distribution to a uniform distribution
        self.dist = np.ones((num_weights,), dtype=float) / num_weights
        self.weights = np.linspace(0.0, 1.0, num_weights)

        # Store the type of reward function to be used
        self.reward_fun = reward_fun
    
    def reset(self):
        """
        Resets the distribution to the uniform distribution
        """

        num_weights = int(1.0/self.stepsize) + 1
        self.dist = np.ones((num_weights,), dtype=float) / num_weights

    def normalize(self):
        """Normalizes the distribution by dividing by its sum"""
        self.dist = self.dist / np.sum(self.dist)
    
    def update(self, rec, act, trust, health, time, threat_level):
        """
        Updates the distribution based on observed values of the variables
        :param rec: the recommendation given to the human
        :param act: the action chosen by the human
        :param trust: the trust level BEFORE choosing the action
        :param health: the health level of the soldier BEFORE choosing the action
        :param time: the time spent in the mission BEFORE choosing the action
        :param threat_level: the threat level at the current site
        """

        temp_dist = None
        
        rewards = self.reward_fun.reward(health, time, house=None)

        rewards_0 = self.weights * threat_level * rewards[0]
        rewards_1 = (1.0 - self.weights) * rewards[1]

        # Given disusing, probability of choosing the two actions (bounded rationality model)
        prob_choose_0 = 1./ (1 + np.exp(self.kappa * (rewards_1 - rewards_0)))
        prob_choose_1 = 1. - prob_choose_0

        if act:
            prob_choose_act = prob_choose_1
        else:
            prob_choose_act = prob_choose_0

        if act == rec:
            # Update considering "using" + "disusing"
            temp_dist = (trust + (1-trust) * prob_choose_act) * self.dist
        else:
            # Update considering "disusing"
            temp_dist = (1-trust) * prob_choose_act * self.dist

        self.dist = temp_dist
        self.normalize()
    
    def get_map(self):
        """
        Returns the MAP estimate of the health reward weight and its probability
        """
        # TODO: What if there are two peaks? Very unlikely in simulation, but possible in experiments

        max_prob = np.max(self.dist)
        weight = self.weights[np.argmax(self.dist)]

        return max_prob, weight
    
    def get_mean(self):
        """
        Returns the mean of the maintained distribution
        """
        return np.sum(self.dist * self.weights)

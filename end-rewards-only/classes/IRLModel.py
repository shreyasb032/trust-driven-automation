import numpy as np
from classes.Rewards import RewardsBase


class Posterior:

    def __init__(self, kappa: float, stepsize: float, reward_fun: RewardsBase):
        """
        :param kappa: Rationality coefficient for the human
        :param stepsize: The stepsize of the stored discrete distribution
        :param reward_fun: The reward function for the human (NOTE: HERE WE ASSUME
                                                    THAT THE HUMAN OBSERVES IMMEDIATE REWARDS)
        :return None
        """

        # Rationality index
        self.kappa = kappa

        # Stepsize for the discrete distribution
        self.stepsize = stepsize
        num_weights = int(1.0 / stepsize) + 1

        # Initialize the distribution to a uniform distribution
        self.dist = np.ones((num_weights,), dtype=float) / num_weights
        self.weights = np.linspace(0.0, 1.0, num_weights)

        # Store the type of reward function to be used
        self.reward_fun = reward_fun

    def reset(self):
        """
        Resets the distribution to a uniform distribution
        """

        num_weights = int(1.0 / self.stepsize) + 1
        self.dist = np.ones((num_weights,), dtype=float) / num_weights

    def normalize(self):
        """
        Normalizes the distribution by dividing by its sum
        """
        self.dist = self.dist / np.sum(self.dist)

    def update(self, rec, act, trust, health, time, threat_level, site_num):
        """
        Updates the maintained distribution
        :param rec: recommendation provided by the robot
        :param act: the action chosen by the human
        :param trust: the trust at the previous stage (BEFORE CHOOSING CURRENT ACTION AND OBSERVING OUTCOME)
        :param health: the health before searching the current site
        :param time: the time spent in simulation until current site
        :param threat_level: the threat level at the current site (after or before scanning)
        :param site_num: the current site number
        """

        rewards = self.reward_fun.reward(health, time, site_num)

        rewards_0 = self.weights * threat_level * rewards[0]
        rewards_1 = (1.0 - self.weights) * rewards[1]

        # Given disusing, probability of choosing the two actions (bounded rationality model)
        prob_choose_0 = 1. / (1. + np.exp(self.kappa * (rewards_1 - rewards_0)))
        prob_choose_1 = 1. - prob_choose_0

        if act:
            prob_choose_act = prob_choose_1
        else:
            prob_choose_act = prob_choose_0

        if act == rec:
            # Update considering "using" + "disusing"
            temp_dist = (trust + (1 - trust) * prob_choose_act) * self.dist
        else:
            # Update considering "disusing"
            temp_dist = (1 - trust) * prob_choose_act * self.dist

        self.dist = temp_dist
        self.normalize()

    def map(self):
        """
        Returns the maximum of the posterior distribution
        """
        # TODO: What if there are two peaks? Very unlikely in simulation, but possible in experiments

        max_prob = np.max(self.dist)
        weight = self.weights[np.argmax(self.dist)]

        return max_prob, weight

    def mean(self):
        """
        Returns the mean of the distribution
        """
        return np.sum(self.dist * self.weights)

    def cdf(self, val):
        """
        Returns the cumulative distribution function of the currently maintained distribution
        :param val: the value at which the cdf is to be computed
        """
        idx = np.sum(self.weights[self.weights <= val])
        return np.sum(self.dist[:idx])

import numpy as np
from classes.RewardFunctions import RewardsBase

class Posterior:

    def __init__(self, kappa: float, stepsize: float, reward_fun: RewardsBase):

        # Rationality index
        self.kappa = kappa

        # Stepsize for the discretized distribution
        self.stepsize = stepsize
        num_weights = int(1.0/stepsize) + 1

        # Initialize the distribution to a uniform distribution
        self.dist = np.ones((num_weights,), dtype=float) / num_weights
        self.weights = np.linspace(0.0, 1.0, num_weights)

        # Store the type of reward function to be used
        self.reward_fun = reward_fun
    
    def reset(self):

        num_weights = int(1.0/self.stepsize) + 1
        self.dist = np.ones((num_weights,), dtype=float) / num_weights

    def normalize(self):
        self.dist = self.dist / np.sum(self.dist)
    
    def update(self, rec, act, trust, health, time, threat_level):

        temp_dist = None
        
        rewards = self.reward_fun.reward(health, time)

        rewards_0 = self.weights * threat_level * rewards[0]
        rewards_1 = (1.0 - self.weights) * rewards[1]

        # Given disusing, probability of choosing the two actions (bounded rationality model)
        prob_choose_0 = np.exp(self.kappa * rewards_0)
        prob_choose_1 = np.exp(self.kappa * rewards_1)
        temp_sum = prob_choose_0 + prob_choose_1
        prob_choose_0 /= temp_sum
        prob_choose_1 /= temp_sum

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

        # import pdb; pdb.set_trace()        
        self.dist = temp_dist
        self.normalize()
    
    def get_map(self):
        # TODO: What if there are two peaks? Very unlikely in simulation, but possible in experiments

        max_prob = np.max(self.dist)
        weight = self.weights[np.argmax(self.dist)]

        return max_prob, weight
    
    def get_mean(self):
        return np.sum(self.dist * self.weights)

from classes.Human import HumanBase, SigmoidHuman
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="talk", style='darkgrid')


class WeightsLearner:
    """
    Learns the health weight associated with a particular state using Logistic Regression on the answered queries
    """

    def __init__(self, human: HumanBase, num_workers: int = 5, threat_chances: List[int] | None = None):
        self.human = human
        self.num_workers = num_workers
        self.threat_chances = threat_chances

        if threat_chances is None:
            self.threat_chances = [10*i for i in range(11)]

        self.model = LogisticRegression(random_state=0)
        self.choices = None
        self.independent_var = None
        self.dependent_var = None
        self.learned_wh = None
        self.learned_dstar = None
        self.true_wh = None
        self.true_dstar = None

    def simulate_and_learn(self, health: int, time: int):
        state = np.array([[health, time]])
        self.choices = []

        for _ in range(self.num_workers):
            for threat_chance in self.threat_chances:
                choice = self.human.choose(state, threat_chance)
                self.choices.append(choice)

        self.independent_var = np.array(self.threat_chances * self.num_workers)
        self.independent_var = self.independent_var.reshape(-1, 1)
        self.dependent_var = np.array(self.choices)
        clf = self.model.fit(self.independent_var, self.dependent_var)

        state = np.array([[health, time]], dtype=float)
        self.true_wh = self.human.get_wh(state)
        self.true_dstar = (1. - self.true_wh) / self.true_wh

        self.learned_dstar = ((0.5 - self.model.intercept_) / self.model.coef_) / 100.
        self.learned_wh = 1. / (1 + self.learned_dstar)

        return self.learned_wh[0, 0]

    def plot(self):
        if self.learned_wh is None:
            raise Exception("Simulate and learn before plotting")

        fig, ax = plt.subplots(layout='tight')
        threat_array = np.array(self.threat_chances * self.num_workers, dtype=float)
        threat_array += np.random.random(size=threat_array.shape) * 8
        ax.scatter(threat_array, self.choices, label='Action choice', alpha=0.7)
        ax.set_title('Simulated action choices')
        ax.set_xlabel('Threat chance %')
        ax.set_ylabel('Action chosen')

        x_test = np.linspace(0., 100., 101)
        y_test = x_test * self.model.coef_ + self.model.intercept_
        _sigmoid = expit(y_test)

        # ravel to convert the 2-d array to a flat array
        ax.plot(x_test, _sigmoid.ravel(), c="green", label="logistic fit")
        plt.axhline(.5, color="red", label="cutoff")
        if self.true_dstar < 1.0:
            plt.axvline(self.true_dstar * 100., color='red', label='dstar')

        return fig, ax

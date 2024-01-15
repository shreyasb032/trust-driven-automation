import numpy as np

seed = 123
prob_bdm = 31./45.
prob_disbeliever = 5./45.
prob_oscillator = 1.0 - prob_bdm - prob_disbeliever


class TrustParamsGenerator:

    def __init__(self):
        self.rng = np.random.default_rng(seed=seed)
        self.params = None

    def generate(self):
        choice = self.rng.choice(3, p=[prob_bdm, prob_disbeliever, prob_oscillator])

        if choice == 0:
            # Bayesian decision maker
            self.params = None
        elif choice == 1:
            # Disbeliever
            self.params = None
        elif choice == 2:
            # Oscillator
            self.params = None

        return list(self.params)

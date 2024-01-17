import numpy as np

seed = 123
prob_bdm = 31./45.
prob_disbeliever = 5./45.
prob_oscillator = 1.0 - prob_bdm - prob_disbeliever

bdm_hist = {
    "Alpha": {
        'p': [0.16, 0.13, 0.16, 0.32, 0.23],
        'bins': [1.47, 25.92, 50.36, 74.81, 99.26, 123.71]
    },
    "Beta": {
        'p': [0.26, 0.32, 0.19, 0.13, 0.1],
        'bins': [1.29, 19.31, 37.32, 55.34, 73.35, 91.36]
    },
    "ws": {
        'p': [0.26, 0.19, 0.26, 0.23, 0.06],
        'bins': [7.75, 11.86, 15.98, 20.09, 24.21, 28.33]
    },
    "wf": {
        'p': [0.06, 0.16, 0.23, 0.29, 0.26],
        'bins': [4.14,  8.89, 13.64, 18.39, 23.15, 27.9]
    }
}

oscillator_hist = {
    "Alpha": {
        'p': [0.6, 0., 0., 0., 0.4],
        'bins': [0.44, 8.42, 16.39, 24.36, 32.34, 40.31]
    },
    "Beta": {
        'p': [0.2, 0., 0., 0., 0.8],
        'bins': [5.22, 19.15, 33.08, 47., 60.93, 74.86]
    },
    "ws": {
        'p': [0.6, 0.2, 0., 0., 0.2],
        'bins': [2.87, 4.63, 6.4, 8.16, 9.92, 11.68]
    },
    "wf": {
        'p': [0.2, 0.4, 0., 0.2, 0.2],
        'bins': [17.81, 20.39, 22.96, 25.54, 28.12, 30.7]
    }
}

disbeliever_hist = {
    "Alpha": {
        'p': [0.11, 0.22, 0.11, 0.22, 0.34],
        'bins': [8.21, 20.49, 32.78, 45.06, 57.34, 69.62]
    },
    "Beta": {
        'p': [0.33, 0., 0.11, 0.33, 0.23],
        'bins': [0., 9.82, 19.64, 29.45, 39.27, 49.09]
    },
    "ws": {
        'p': [0.22, 0.56, 0.11, 0., 0.11],
        'bins': [3.45, 5.49, 7.53, 9.57, 11.6, 13.64]
    },
    "wf": {
        'p': [0.22, 0.11, 0.22, 0., 0.45],
        'bins': [7.75, 9.9, 12.05, 14.19, 16.34, 18.49]
    }
}


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

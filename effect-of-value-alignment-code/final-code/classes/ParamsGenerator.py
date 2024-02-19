import numpy as np
from typing import Dict
import json
import os.path as path

prob_bdm = 31. / 45.
prob_disbeliever = 5. / 45.
prob_oscillator = 1.0 - prob_bdm - prob_disbeliever


class TrustParamsGenerator:

    def __init__(self, seed=123, add_noise=True):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.add_noise = add_noise
        self.params = {}
        self.group = None

        self.bdm_dict = None
        self.oscillator_dict = None
        self.disbeliever_dict = None

        parent_directory = path.join('..', 'human-subjects-data')

        bdm_file = path.join(parent_directory, 'bdm.json')
        oscillator_file = path.join(parent_directory, 'oscillator.json')
        disbeliever_file = path.join(parent_directory, 'disbeliever.json')

        with open(bdm_file, 'r') as f:
            self.bdm_dict = json.load(f)
            self.num_bdm = len(self.bdm_dict['Alpha'])

        with open(oscillator_file, 'r') as f:
            self.oscillator_dict = json.load(f)
            self.num_oscillators = len(self.oscillator_dict['Alpha'])

        with open(disbeliever_file, 'r') as f:
            self.disbeliever_dict = json.load(f)
            self.num_disbelievers = len(self.disbeliever_dict['Alpha'])

    def generate_noise(self):
        return self.rng.normal(loc=0., scale=5.0)

    def generate(self):
        # Choose whether to select a bdm, a disbeliever, or an oscillator
        choice = self.rng.choice(3, p=[prob_bdm, prob_disbeliever, prob_oscillator])

        # Clear the params
        self.params = {}
        # If BDM
        if choice == 0:
            index = self.rng.choice(self.num_bdm)
            for key in self.bdm_dict.keys():
                self.params[key] = self.bdm_dict[key][index]
                if self.add_noise:
                    self.params[key] += self.generate_noise()
                    self.params[key] = max(self.params[key], 0.)
        # If disbeliever
        elif choice == 1:
            index = self.rng.choice(self.num_disbelievers)
            for key in self.disbeliever_dict.keys():
                self.params[key] = self.disbeliever_dict[key][index]
                if self.add_noise:
                    self.params[key] += self.generate_noise()
                    self.params[key] = max(self.params[key], 0.)
        # If oscillator
        elif choice == 2:
            index = self.rng.choice(self.num_oscillators)
            for key in self.oscillator_dict.keys():
                self.params[key] = self.oscillator_dict[key][index]
                if self.add_noise:
                    self.params[key] += self.generate_noise()
                    self.params[key] = max(self.params[key], 0.)

        return self.params

# OLD METHOD WITH INDEPENDENT HISTOGRAMS ##############################################################################
# bdm_hist = {
#     "Alpha": {
#         'p': [0.16, 0.13, 0.16, 0.32, 0.23],
#         'bins': [1.47, 25.92, 50.36, 74.81, 99.26, 123.71]
#     },
#     "Beta": {
#         'p': [0.26, 0.32, 0.19, 0.13, 0.1],
#         'bins': [1.29, 19.31, 37.32, 55.34, 73.35, 91.36]
#     },
#     "ws": {
#         'p': [0.26, 0.19, 0.26, 0.23, 0.06],
#         'bins': [7.75, 11.86, 15.98, 20.09, 24.21, 28.33]
#     },
#     "wf": {
#         'p': [0.06, 0.16, 0.23, 0.29, 0.26],
#         'bins': [4.14,  8.89, 13.64, 18.39, 23.15, 27.9]
#     }
# }
#
# oscillator_hist = {
#     "Alpha": {
#         'p': [0.6, 0., 0., 0., 0.4],
#         'bins': [0.44, 8.42, 16.39, 24.36, 32.34, 40.31]
#     },
#     "Beta": {
#         'p': [0.2, 0., 0., 0., 0.8],
#         'bins': [5.22, 19.15, 33.08, 47., 60.93, 74.86]
#     },
#     "ws": {
#         'p': [0.6, 0.2, 0., 0., 0.2],
#         'bins': [2.87, 4.63, 6.4, 8.16, 9.92, 11.68]
#     },
#     "wf": {
#         'p': [0.2, 0.4, 0., 0.2, 0.2],
#         'bins': [17.81, 20.39, 22.96, 25.54, 28.12, 30.7]
#     }
# }
#
# disbeliever_hist = {
#     "Alpha": {
#         'p': [0.11, 0.22, 0.11, 0.22, 0.34],
#         'bins': [8.21, 20.49, 32.78, 45.06, 57.34, 69.62]
#     },
#     "Beta": {
#         'p': [0.33, 0., 0.11, 0.33, 0.23],
#         'bins': [0., 9.82, 19.64, 29.45, 39.27, 49.09]
#     },
#     "ws": {
#         'p': [0.22, 0.56, 0.11, 0., 0.11],
#         'bins': [3.45, 5.49, 7.53, 9.57, 11.6, 13.64]
#     },
#     "wf": {
#         'p': [0.22, 0.11, 0.22, 0., 0.45],
#         'bins': [7.75, 9.9, 12.05, 14.19, 16.34, 18.49]
#     }
# }


# class TrustParamsGenerator:
#
#     def __init__(self, seed=123):
#         self.rng = np.random.default_rng(seed=seed)
#         self.params = None
#         # self.n_bins = len(bdm_hist['Alpha']['p'])
#         self.group = None
#
#     def choose(self, data: Dict):
#         trust_params = {}
#         for key in data.keys():
#             hist_dict = data[key]
#             bin_choice = self.rng.choice(self.n_bins, p=hist_dict['p'])
#             bin_left, bin_right = hist_dict['bins'][bin_choice], hist_dict['bins'][bin_choice + 1]
#             trust_params[key] = self.rng.uniform(bin_left, bin_right)
#
#         return trust_params
#
#     def generate(self):
#         choice = self.rng.choice(3, p=[prob_bdm, prob_disbeliever, prob_oscillator])
#
#         if choice == 0:
#             # Bayesian decision maker
#             self.group = 'BDM'
#             trust_params = self.choose(bdm_hist)
#             self.params = [trust_params['Alpha'], trust_params['Beta'], trust_params['ws'], trust_params['wf']]
#         elif choice == 1:
#             # Disbeliever
#             self.group = 'Disbeliever'
#             trust_params = self.choose(disbeliever_hist)
#             self.params = [trust_params['Alpha'], trust_params['Beta'], trust_params['ws'], trust_params['wf']]
#         elif choice == 2:
#             # Oscillator
#             self.group = 'Oscillator'
#             trust_params = self.choose(oscillator_hist)
#             self.params = [trust_params['Alpha'], trust_params['Beta'], trust_params['ws'], trust_params['wf']]
#
#         return self.params

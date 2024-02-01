import argparse
import numpy as np


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--trust-weight', type=float,
                        help='trust weight for the robot (default: 0.0)',
                        default=0.0)

    parser.add_argument('--kappa', type=float, help='rationality coefficient (default: 0.2)', default=0.2)

    parser.add_argument('--trust-params',
                        type=int,
                        help='Trust parameters for the human (0-low, 1-mid, 2-high), default=1',
                        default=1)

    parser.add_argument('--num-sites', type=int, help='Number of sites in a mission (default: 20)', default=20)
    parser.add_argument('--num-missions', type=int, help='Number of missions (default: 1)', default=1)
    parser.add_argument('--print-flag', type=bool, help="Flag to print the data to output (default: False)",
                        default=False)

    parser.add_argument('--num-simulations', type=int, help='Number of simulations to run (default: 40)', default=40)

    parser.add_argument('--posterior-stepsize', type=float,
                        help='Stepsize in the posterior distribution (default(0.05)', default=0.05)

    parser.add_argument('--num-gradient-steps', type=int,
                        help='Number of iterations of gradient descent for trust parameter estimation (default: 200)',
                        default=200)

    parser.add_argument('--gradient-stepsize', type=float,
                        help='Stepsize for gradient descent (default: 0.0001)', default=0.001)

    parser.add_argument('--tolerance', type=float,
                        help='Error tolerance for the gradient descent step (default: 0.01)', default=0.01)

    parser.add_argument('--perf-metric',
                        type=int, help="Performance metric to use: 1 - Observed, 2 - Expected (default: 1)", default=1)

    parser.add_argument('--hl', type=float, help="Health loss cost (default: 10.0)", default=10.0)

    parser.add_argument('--tc', type=float, help="Time loss cost (default: 10.0)", default=10.0)

    parser.add_argument('--threat-level', type=float,
                        help='Common Threat Level to set across all houses (default:0.7)', default=0.7)

    parser.add_argument('--human-model-solver',
                        type=int,
                        help='Human model to be used by the solver - 0:bounded rational, 1:reverse psychology, '
                             '2:disuse',
                        default=0)

    parser.add_argument('--human-model-actual',
                        type=int,
                        help='Human model to be used to simulate the human - 0:bounded rational, 1:reverse psychology, '
                             '2:one step optimal',
                        default=2)
    return parser


def initialize_storage_dict(num_simulations, N, num_weights):
    data = {'trust feedback': np.zeros((num_simulations, N + 1), dtype=float),
            'trust estimate': np.zeros((num_simulations, N + 1), dtype=float),
            'health': np.zeros((num_simulations, N + 1), dtype=int),
            'time': np.zeros((num_simulations, N + 1), dtype=int),
            'recommendation': np.zeros((num_simulations, N), dtype=int),
            'actions': np.zeros((num_simulations, N), dtype=int),
            'weights': np.zeros((num_simulations, N, num_weights), dtype=float),
            'posterior': np.zeros((num_simulations, N + 1, num_weights), dtype=float),
            'prior threat level': np.zeros((num_simulations, N), dtype=float),
            'after scan level': np.zeros((num_simulations, N), dtype=float),
            'threat': np.zeros((num_simulations, N), dtype=int),
            'trust parameter estimates': np.zeros((num_simulations, N + 1, 4), dtype=float),
            'mean health weight': np.zeros((num_simulations, N + 1), dtype=float),
            'map health weight': np.zeros((num_simulations, N + 1), dtype=float),
            'map health weight probability': np.zeros((num_simulations, N + 1), dtype=float),
            'performance estimates': np.zeros((num_simulations, N), dtype=int),
            'performance actual': np.zeros((num_simulations, N), dtype=int)}

    return data


def col_print(table_data):
    string = ""
    for _ in range(len(table_data[0])):
        string += "{: ^12}"

    for row in table_data:
        print(string.format(*row))


class SimParams:

    def __init__(self):
        # From inputs
        self.wh_hum = None
        self.wh_rob = None
        self.wt_rob = None
        self.hl = None
        self.tc = None
        self.threat_level = None
        self.kappa = None
        self.df = None
        self.k = None

        # Constants
        self.N = None
        self.alpha_0_list = None
        self.beta_0_list = None
        self.ws_list = None
        self.wf_list = None

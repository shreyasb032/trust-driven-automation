import numpy as np
from numpy.lib.scimath import sqrt
from typing import Dict


def norm(grads: Dict):
    val = 0
    for v in grads.values():
        val += v ** 2

    return sqrt(val)


class Estimator:

    def __init__(self, num_iterations: int = 1000,
                 stepsize: float = 0.0001,
                 error_tol: float = 0.01,
                 num_sites: int = 20):
        """
        Estimates trust parameters based on the performance history and the trust feedback history
        :param num_iterations: the maximum number of gradient steps to take
        :param stepsize: the learning rate of the gradient descent algorithm
        :param error_tol: the tolerance below which gradient descent is stopped
        """

        self.MAX_ITER = num_iterations
        self.stepsize = stepsize
        self.error_tol = error_tol
        self.N = num_sites

        # Initial guesses for the trust params
        self.gp_list = {0.0: [2., 98., 20., 30.],
                        0.1: [10., 90., 20., 30.],
                        0.2: [20., 80., 20., 30.],
                        0.3: [30., 70., 20., 30.],
                        0.4: [40., 60., 20., 30.],
                        0.5: [50., 50., 20., 30.],
                        0.6: [60., 40., 20., 30.],
                        0.7: [70., 30., 20., 30.],
                        0.8: [80., 20., 20., 30.],
                        0.9: [90., 10., 20., 30.],
                        1.0: [98., 2., 20., 30.]}

        self.performance = np.zeros((self.N, ), dtype=int)
        self.feedback = np.zeros((self.N+1, ), dtype=float)

    def reset(self):
        """
        Resets the performance history and the feedback history
        """
        self.performance *= 0
        self.feedback *= 0

    def get_initial_guess(self, feedback: float,
                          site_num: int = -1):
        """
        Decides an initial guess based on the feedback given by the human
        :param feedback: the initial feedback given by the human before any interaction
        :param site_num: (Optional) the site number at which we need a new initial guess
        """

        # Add the feedback to the history
        self.feedback[site_num + 1] = feedback

        t = round(feedback * 10) / 10
        guess_params_list = self.gp_list[t].copy()
        guess_params = {"alpha0": guess_params_list[0],
                        "beta0": guess_params_list[1],
                        "ws": guess_params_list[2],
                        "wf": guess_params_list[3]}

        return guess_params

    def get_params(self, initial_guess: Dict,
                   performance: int,
                   trust_fb: float,
                   site_num: int):

        """
        :param initial_guess: the guess from which the gradient descent is begun at this stage performance = The
        :param performance: performance of the drone at the current site.
        :param trust_fb: The trust feedback of the participant at the current site.
        :param site_num: current site number
        :return guess_params: the updated trust parameters
        """

        self.performance[site_num - 1] = performance
        self.feedback[site_num] = trust_fb

        factor = self.stepsize
        curr_house = len(self.performance)
        lr = np.array([factor, factor, factor / curr_house, factor / curr_house])

        guess_params = initial_guess.copy()  # To keep using current parameters as the initial guess
        # guess_params = self.getInitialGuess(performance[0], feedback[0])   # To use a new initial guess every time

        gradients_except_prior = self.get_grads(guess_params, site_num)
        num_iters = 0

        while norm(gradients_except_prior) > self.error_tol and num_iters < self.MAX_ITER:
            num_iters += 1
            gradients_except_prior = self.get_grads(guess_params, site_num)
            for j, k in enumerate(guess_params.keys()):
                guess_params[k] += lr[j] * gradients_except_prior[k]
                guess_params[k] = max(guess_params[k], 0.1)  # To make sure the di-gamma function behaves well

        return guess_params

    def get_grads(self, params: Dict, site_num: int):

        """
        Gets the gradients of the log likelihood function in order to take a gradient step
        :param params: the trust parameters at which the gradient needs to be computed, dictionary with keys
                        alpha0, beta0, ws, wf
        :param site_num: the current site number
        """

        grads = {"alpha0": 0., "beta0": 0., "ws": 0., "wf": 0.}
        alpha0 = params['alpha0']
        beta0 = params['beta0']
        ws = params['ws']
        wf = params['wf']

        ns = 0
        nf = 0

        # Initial feedback before any interaction
        digamma_both = digamma(alpha0 + beta0)
        digamma_alpha = digamma(alpha0)
        digamma_beta = digamma(beta0)

        delta_alpha = digamma_both - digamma_alpha + np.log(max(self.feedback[0], 0.01))
        delta_beta = digamma_both - digamma_beta + np.log(max(1 - self.feedback[0], 0.01))

        grads['alpha0'] += delta_alpha
        grads['beta0'] += delta_beta

        # For all the feedback received after beginning interaction
        for i in range(site_num):

            ns += self.performance[i]
            nf += 1 - self.performance[i]

            alpha = alpha0 + ns * ws
            beta = beta0 + nf * wf

            digamma_both = digamma(alpha + beta)
            digamma_alpha = digamma(alpha)
            digamma_beta = digamma(beta)

            delta_alpha = digamma_both - digamma_alpha + np.log(max(self.feedback[i+1], 0.01))
            delta_beta = digamma_both - digamma_beta + np.log(max(1 - self.feedback[i+1], 0.01))

            grads['alpha0'] += delta_alpha
            grads['beta0'] += delta_beta
            grads['ws'] += ns * delta_alpha
            grads['wf'] += nf * delta_beta

        return grads


def digamma(x):
    a = 1 / sqrt(6)
    b = 6 - 2 * sqrt(6)

    if x < 6.0:
        return digamma(x + 1) - 1 / x

    return np.log(x + a) - 1 / (b * x)

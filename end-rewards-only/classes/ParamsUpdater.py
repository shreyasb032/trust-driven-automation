import numpy as np
from numpy.linalg import norm
from numpy.lib.scimath import sqrt


class Estimator:

    def __init__(self, num_iterations=1000, stepsize=0.0001, error_tol=0.01):
        """
        Estimates trust parameters based on the performance history and the trust feedback history
        :param num_iterations: the maximum number of gradient steps to take
        :param stepsize: the learning rate of the gradient descent algorithm
        :param error_tol: the tolerance below which gradient descent is stopped
        """

        self.MAX_ITER = num_iterations
        self.stepsize = stepsize
        self.error_tol = error_tol

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

        self.performance = []
        self.feedback = []

    def reset(self):
        """
        Resets the performance history and the feedback history
        """
        self.performance.clear()
        self.feedback.clear()

    def get_initial_guess(self, feedback):
        """
        Decides an initial guess based on the feedback given by the human
        """

        t = round(feedback * 10) / 10
        guess_params = self.gp_list[t].copy()

        return guess_params

    def get_params(self, initial_guess, p, t):

        """
        :param initial_guess: the guess from which the gradient descent is begun at this stage performance = The
        :param p: performance of the drone at the current site.
        :param t: The trust feedback of the participant at the current site.
        :return guess_params: the updated trust parameters
        """

        self.performance.append(p)
        self.feedback.append(t)

        factor = self.stepsize
        curr_house = len(self.performance)
        lr = np.array([factor, factor, factor / curr_house, factor / curr_house])

        guess_params = initial_guess.copy()  # To keep using current parameters as the initial guess
        # guess_params = self.getInitialGuess(performance[0], feedback[0])   # To use a new initial guess every time

        gradients_except_prior = self.get_grads(guess_params)
        num_iters = 0

        while norm(gradients_except_prior) > self.error_tol and num_iters < self.MAX_ITER:
            num_iters += 1
            gradients_except_prior = self.get_grads(guess_params)
            guess_params += lr * gradients_except_prior
            guess_params[guess_params <= 0.1] = 0.1  # To make sure the di-gamma function behaves well

        return guess_params

    def get_grads(self, params):

        """
        Gets the gradients of the log likelihood function in order to take a gradient step
        :param params: the trust parameters at which the gradient needs to be computed
        """

        grads = np.zeros((4,))
        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = 0
        nf = 0

        for i in range(len(self.feedback)):

            # We need to add the number of successes and failures regardless of whether feedback was queried or not
            ns += self.performance[i]
            nf += 1 - self.performance[i]

            alpha = alpha_0 + ns * ws
            beta = beta_0 + nf * wf

            digamma_both = digamma(alpha + beta)
            digamma_alpha = digamma(alpha)
            digamma_beta = digamma(beta)

            delta_alpha = digamma_both - digamma_alpha + np.log(max(self.feedback[i], 0.01))
            delta_beta = digamma_both - digamma_beta + np.log(max(1 - self.feedback[i], 0.01))

            grads[0] += delta_alpha
            grads[1] += delta_beta
            grads[2] += ns * delta_alpha
            grads[3] += nf * delta_beta

        return grads


def digamma(x):
    a = 1 / sqrt(6)
    b = 6 - 2 * sqrt(6)

    if x < 6.0:
        return digamma(x + 1) - 1 / x

    return np.log(x + a) - 1 / (b * x)

import numpy as np
from numpy.linalg import norm
from numpy.lib.scimath import sqrt


class Estimator:

    def __init__(self, num_iterations=1000, step_size=0.0001,
                 query_feedback_at=None, error_tol=0.01,
                 num_sites=40):
        """
        Initializer of the Estimator class
        :param num_iterations: maximum number of iterations of gradient descent to run before stopping
        :param step_size: the step_size for the gradient descent algorithm
        :param query_feedback_at: a list of site numbers at which feedback was queried (default: None, which indicates
                                    that feedback was asked after every site)
        :param error_tol: the error below which we stop the gradient descent algorithm
        :param num_sites: the number of sites in the mission
        """

        self.prior = None
        self.MAX_ITER = num_iterations
        self.step_size = step_size
        self.define_prior()
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

        self.performance = np.zeros((self.N,), dtype=int)
        self.feedback = np.zeros((self.N+1,), dtype=float)
        self.query_feedback_at = query_feedback_at

    def reset(self):
        """
        Resets the estimator
        """
        self.performance = np.zeros((self.N,), dtype=int)
        self.feedback = np.zeros((self.N+1,), dtype=float)

    def define_prior(self):
        """
        Helper function to define the prior over the trust parameters
        """

        prior = {"AlphaEdges": np.array([0, 28, 56, 84, 112, 140]),
                 "AlphaValues": np.array([0.2051, 0.1538, 0.07692, 0.2308, 0.3333]),
                 "BetaEdges": np.array([0, 29, 58, 87, 116, 145]),
                 "BetaValues": np.array([0.1269, 0.2335, 0.3063, 0.1808, 0.1525]),
                 "wsEdges": np.array([0, 14, 28, 42, 56, 70]),
                 "wsValues": np.array([0.5897, 0.1795, 0.1032, 0.07625, 0.05128]),
                 "wfEdges": np.array([0, 28, 56, 84, 112, 140]),
                 "wfValues": np.array([0.5641, 0.1026, 0.05128, 0.0641, 0.2179])}

        self.prior = prior

    def get_initial_guess(self, feedback):
        """
        Get a good initial guess to start the gradient descent algorithm.
        The guess is chosen from a list to best estimate the initial value of trust given by the human
        :param feedback: the initial trust feedback given by the human
        """

        t = round(feedback * 10) / 10
        guess_params = self.gp_list[t].copy()

        return guess_params

    def get_params(self, initial_guess, performance, trust, current_site):

        """
        Function to get the updated list of trust parameters
        :param initial_guess: the guess from which the gradient descent is begun at this stage
        :param performance: The performance of the drone at the current site.
        :param trust: The trust feedback of the participant at the current site.
        :param current_site: the current search site index
        """

        p = performance
        t = trust
        self.performance[current_site] = p
        self.feedback[current_site+1] = t

        factor = self.step_size
        lr = np.array([factor, factor, factor / (current_site+1), factor / (current_site+1)])

        guess_params = initial_guess.copy()  # To keep using current parameters as the initial guess
        # guess_params = self.get_initial_guess(performance[0], feedback[0])   # To use a new initial guess every time

        if self.query_feedback_at is not None:
            if current_site not in self.query_feedback_at:
                return initial_guess

        gradients_except_prior = self.get_grads(guess_params, current_site)
        num_iters = 0

        while norm(gradients_except_prior) > self.error_tol and num_iters < self.MAX_ITER:
            num_iters += 1
            gradients_except_prior = self.get_grads(guess_params, current_site)
            guess_params += lr * gradients_except_prior
            guess_params[guess_params <= 0.1] = 0.1  # To make sure the digamma function behaves well

        return guess_params

    def get_grads(self, params, current_site):
        """
        Returns the gradients of the log-likelihood function using a digamma approximation
        :param params: the trust parameters at which to evaluate the gradients
        :param current_site: the index of the current search site
        """

        grads = np.zeros((4,))
        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = 0
        nf = 0

        digamma_both = digamma(alpha_0 + beta_0)
        digamma_alpha = digamma(alpha_0)
        digamma_beta = digamma(beta_0)

        delta_alpha = digamma_both - digamma_alpha + np.log(max(self.feedback[0], 0.01))
        delta_beta = digamma_both - digamma_beta + np.log(max(1 - self.feedback[0], 0.01))

        grads[0] += delta_alpha
        grads[1] += delta_beta

        for i in range(current_site):

            # We need to add the number of successes and failures regardless of whether feedback was queried or not
            ns += self.performance[i]
            nf += 1 - self.performance[i]

            if self.query_feedback_at is not None:
                if i not in self.query_feedback_at:
                    continue

            # If feedback was queried here, compute the gradients
            alpha = alpha_0 + ns * ws
            beta = beta_0 + nf * wf

            digamma_both = digamma(alpha + beta)
            digamma_alpha = digamma(alpha)
            digamma_beta = digamma(beta)

            delta_alpha = digamma_both - digamma_alpha + np.log(max(self.feedback[i+1], 0.01))
            delta_beta = digamma_both - digamma_beta + np.log(max(1 - self.feedback[i+1], 0.01))

            grads[0] += delta_alpha
            grads[1] += delta_beta
            grads[2] += ns * delta_alpha
            grads[3] += nf * delta_beta

        return grads


def digamma(x):
    """
    An approximation to the digamma function
    """
    a = 1 / sqrt(6)
    b = 6 - 2 * sqrt(6)

    if x < 6.0:
        return digamma(x + 1) - 1 / x

    return np.log(x + a) - 1 / (b * x)

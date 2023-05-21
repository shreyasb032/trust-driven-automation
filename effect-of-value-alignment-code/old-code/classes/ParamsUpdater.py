import numpy as np
from numpy.linalg import norm
from numpy.lib.scimath import sqrt
from scipy.stats import beta as beta_dist
from numpy.random import beta as beta_sampler


class Estimator:

    def __init__(self, num_iterations=1000, stepsize=0.0001, use_prior=True,
                 query_feedback_at=None, error_tol=0.01):
        """
        Initializer of the Estimator class
        :param num_iterations: maximum number of iterations of gradient descent to run before stopping
        :param stepsize: the stepsize for the gradient descent algorithm
        :param use_prior: whether to use the prior for estimating the trust parameters
        :param query_feedback_at: a list of site numbers at which feedback was queried (default: None, which indicates
                                    that feedback was asked after every site)
        :param error_tol: the error below which we stop the gradient descent algorithm
        """

        self.prior = None
        self.MAX_ITER = num_iterations
        self.stepsize = stepsize
        self.define_prior()
        self.use_prior = use_prior
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
        self.query_feedback_at = query_feedback_at

    def reset(self):
        """
        Resets the estimator
        """
        self.performance.clear()
        self.feedback.clear()

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

    def get_params(self, initial_guess, performance, trust):

        """
        Function to get the updated list of trust parameters
        :param initial_guess: the guess from which the gradient descent is begun at this stage
        :param performance: The performance of the drone at the current site.
        :param trust: The trust feedback of the participant at the current site.
        """

        p = performance
        t = trust
        self.performance.append(p)
        self.feedback.append(t)

        factor = self.stepsize
        curr_house = len(self.performance)
        lr = np.array([factor, factor, factor / curr_house, factor / curr_house])

        guess_params = initial_guess.copy()  # To keep using current parameters as the initial guess
        # guess_params = self.get_initial_guess(performance[0], feedback[0])   # To use a new initial guess every time

        if self.query_feedback_at is not None:
            if len(self.performance) - 1 not in self.query_feedback_at:
                return initial_guess

        gradients_except_prior = self.get_grads(guess_params)
        num_iters = 0

        while norm(gradients_except_prior) > self.error_tol and num_iters < self.MAX_ITER:
            num_iters += 1
            gradients_except_prior = self.get_grads(guess_params)
            guess_params += lr * gradients_except_prior
            guess_params[guess_params <= 0.1] = 0.1  # To make sure the digamma function behaves well

        return guess_params

    def get_grads(self, params):
        """
        Returns the gradients of the log-likelihood function using a digamma approximation
        :param params: the trust parameters at which to evaluate the gradients
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

            if self.query_feedback_at is not None:
                if i not in self.query_feedback_at:
                    continue

            # If feedback was queried here, compute the gradients
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

        if not self.use_prior:
            return grads

        prior = self.prior

        alpha_idx = np.sum(params[0] > prior["AlphaEdges"]) - 1
        alpha_prior = prior["AlphaValues"][alpha_idx]
        alpha_slope = -(prior["AlphaValues"][alpha_idx] - prior["AlphaValues"][max(alpha_idx - 1, 0)]) / (
                    prior["AlphaEdges"][0] - prior["AlphaEdges"][1])

        beta_idx = np.sum(params[1] > prior["BetaEdges"]) - 1
        beta_prior = prior["BetaValues"][beta_idx]
        beta_slope = -(prior["BetaValues"][beta_idx] - prior["BetaValues"][max(beta_idx - 1, 0)]) / (
                    prior["BetaEdges"][0] - prior["BetaEdges"][1])

        ws_idx = np.sum(params[2] > prior["wsEdges"]) - 1
        ws_prior = prior["wsValues"][ws_idx]
        ws_slope = -(prior["wsValues"][ws_idx] - prior["wsValues"][max(ws_idx - 1, 0)]) / (
                    prior["wsEdges"][0] - prior["wsEdges"][1])

        wf_idx = np.sum(params[3] > prior["wfEdges"]) - 1
        wf_prior = prior["wfValues"][wf_idx]
        wf_slope = -(prior["wfValues"][wf_idx] - prior["wfValues"][max(wf_idx - 1, 0)]) / (
                    prior["wfEdges"][0] - prior["wfEdges"][1])

        # print("Alpha", grads[0], alpha_slope/alpha_prior)
        # print("Beta", grads[1], beta_slope/beta_prior)
        # print("ws", grads[2], ws_slope/ws_prior)
        # print("wf", grads[3], wf_slope/wf_prior)

        grads[0] += alpha_slope / alpha_prior
        grads[1] += beta_slope / beta_prior
        grads[2] += ws_slope / ws_prior
        grads[3] += wf_slope / wf_prior

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


class EstimateViaSearch:

    def __init__(self, method=1, max_iter=10000):
        self.definePrior()
        self.method = method
        self.max_iter = max_iter
        self.gp_list = {0.0: [10, 100, 20, 30],
                        0.1: [15, 135, 20, 30],
                        0.2: [30, 120, 20, 30],
                        0.3: [45, 105, 20, 30],
                        0.4: [60, 90, 20, 30],
                        0.5: [100, 100, 20, 30],
                        0.6: [120, 80, 20, 30],
                        0.7: [105, 45, 20, 30],
                        0.8: [120, 30, 20, 30],
                        0.9: [135, 15, 20, 30],
                        1.0: [100, 10, 20, 30]}

    def getInitialGuess(self, performance, feedback):

        t = round(feedback * 10) / 10
        guess_params = self.gp_list[t].copy()

        if performance == 1:
            guess_params[0] -= guess_params[2]
            guess_params[0] = max(0, guess_params[0])
        else:
            guess_params[1] -= guess_params[3]
            guess_params[1] = max(0, guess_params[1])

        return np.array(guess_params, dtype=float)

    def definePrior(self):

        prior = {"AlphaEdges": None, "AlphaValues": None, "BetaEdges": None, "BetaValues": None,
                 "wsEdges": None, "wsValues": None, "wfEdges": None, "wfValues": None}

        prior["AlphaEdges"] = np.array([0, 28, 56, 84, 112, 140])
        prior["AlphaValues"] = np.array([0.2051, 0.1538, 0.07692, 0.2308, 0.3333])

        prior["BetaEdges"] = np.array([0, 29, 58, 87, 116, 145])
        prior["BetaValues"] = np.array([0.1269, 0.2335, 0.3063, 0.1808, 0.1525])

        prior["wsEdges"] = np.array([0, 14, 28, 42, 56, 70])
        prior["wsValues"] = np.array([0.5897, 0.1795, 0.1032, 0.07625, 0.05128])

        prior["wfEdges"] = np.array([0, 28, 56, 84, 112, 140])
        prior["wfValues"] = np.array([0.5641, 0.1026, 0.05128, 0.0641, 0.2179])

        self.prior = prior

    def probTheta(self, theta):

        prior = self.prior

        idx = np.sum(prior["AlphaEdges"] < theta[0]) - 1
        idx = min(4, idx)
        p1 = prior["AlphaValues"][idx]

        idx = np.sum(prior["BetaEdges"] < theta[1]) - 1
        idx = min(4, idx)
        p2 = prior["BetaValues"][idx]

        idx = np.sum(prior["wsEdges"] < theta[2]) - 1
        idx = min(4, idx)
        p3 = prior["wsValues"][idx]

        idx = np.sum(prior["wfEdges"] < theta[3]) - 1
        idx = min(4, idx)
        p4 = prior["wfValues"][idx]

        return p1 * p2 * p3 * p4

    def LogL(self, theta, trustHistory, performance, q_list):

        prior = self.prior

        alpha_0 = theta[0]
        beta_0 = theta[1]
        ws = theta[2]
        wf = theta[3]

        k = trustHistory.shape[0]
        alpha = np.ones((k + 1,)) * alpha_0
        beta = np.ones((k + 1,)) * beta_0

        for i in range(1, k + 1):
            alpha[i] = alpha[i - 1] + ws * performance[i - 1]
            beta[i] = beta[i - 1] + wf * performance[i - 1]

        reports = q_list[q_list < k]
        idx = np.arange(k + 1)
        idx = idx[reports]

        logLBeta = np.sum(beta_dist.logpdf(trustHistory[idx], alpha[idx], beta[idx]))

        LogLTheta = np.log(self.probTheta(theta) + 0.1)

        logL = LogLTheta + logLBeta

        return logL

    def curveError(self, theta, trustHistory, performance, q_list):

        alpha = theta[0]
        beta = theta[1]
        ws = theta[2]
        wf = theta[3]

        err = (trustHistory[0] - alpha / (alpha + beta)) ** 2

        for i, p in enumerate(performance):
            alpha = alpha + ws * p
            beta = beta + wf * (1 - p)

            if i in q_list:
                err += (trustHistory[i] - alpha / (alpha + beta)) ** 2

        return err

    def getError(self, alpha, beta_, ws, wf, feedback, performance):

        error = 0
        est = []
        dist_est = np.zeros((50, 20))

        for i in range(feedback.shape[0]):
            alpha += ws * performance[i]
            beta_ += wf * (1 - performance[i])
            trust_est = alpha / (alpha + beta_)

            trust_dist = beta_sampler(alpha, max(beta_, 0.01), size=(20,))

            est.append(trust_est)
            dist_est[i, :] = trust_dist
            error += (feedback[i] - trust_est) ** 2

        error /= feedback.shape[0]
        error = np.sqrt(error)

        return error, est, dist_est

    def estimate(self, trust, performance, q_list=None, theta=np.array([100., 100., 10., 20.])):

        err_old = self.curveError(theta, trust, performance, q_list)
        logL_old = self.LogL(theta, trust, performance, q_list)

        for iter in range(self.max_iter):

            # if iter % 500 == 0:
            #     print("Iteration {k}/{K}".format(k=iter, K=self.max_iter))

            step = -0.05 + 0.1 * np.random.rand(4)
            theta_cpy = theta.copy()
            theta_cpy += step
            theta_cpy[theta_cpy < 0] = 0

            if self.method == 1:
                logL_new = self.LogL(theta_cpy, trust, performance, q_list)
                if logL_new > logL_old:
                    theta = theta_cpy
                    logL_old = logL_new
            else:
                err_new = self.curveError(theta_cpy, trust, performance, q_list)
                if err_new < err_old:
                    theta = theta_cpy
                    err_old = err_new

        return theta

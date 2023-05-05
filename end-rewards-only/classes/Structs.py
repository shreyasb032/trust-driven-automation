import argparse


class MissionSettings:

    def __init__(self, args: argparse.Namespace):
        self.num_sites = args.num_sites
        self.health_loss = args.health_loss
        self.time_loss = args.time_loss
        self.health = args.starting_health
        self.time_ = args.starting_time
        self.health_loss_cost = args.health_loss_cost
        self.time_loss_cost = args.time_loss_cost
        self.threat_level = args.threat_level
        self.threat_seed = args.threat_seed


class SimulatedHumanSettings:

    def __init__(self, args: argparse.Namespace):
        self.kappa = args.simulated_kappa
        self.posterior_stepsize = args.simulated_posterior_stepsize
        self.wh = args.simulated_wh
        self.trust_params = {"alpha0": args.simulated_alpha0,
                             "beta0": args.simulated_beta0,
                             "ws": args.simulated_ws,
                             "wf": args.simulated_wf}
        self.seed = args.simulated_seed


class HumanModelSettings:

    def __init__(self, args: argparse.Namespace):
        self.kappa = args.model_kappa
        self.posterior_stepsize = args.model_posterior_stepsize
        self.trust_params = {"alpha0": args.model_alpha0,
                             "beta0": args.model_beta0,
                             "ws": args.model_ws,
                             "wf": args.model_wf}
        self.seed = args.model_seed


class SolverSettings:

    def __init__(self, args: argparse.Namespace):
        self.wh = args.solver_wh
        self.df = args.discount_factor


class ParamsEstimatorSettings:

    def __init__(self, args: argparse.Namespace):
        self.num_iters = args.estimator_num_iterations
        self.stepsize = args.estimator_stepsize
        self.error_tolerance = args.estimator_error_tolerance

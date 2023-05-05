import argparse


def add_common_args():
    """
    Adds common arguments for the simulation script
    """
    parser = argparse.ArgumentParser()
    parser = add_mission_args(parser)
    parser = add_model_human_args(parser)
    parser = add_simulated_human_args(parser)
    parser = add_solver_args(parser)
    parser = add_estimator_args(parser)

    return parser


def add_mission_args(parser: argparse.ArgumentParser):
    """
    Adds arguments related to mission settings:
    num_sites: number of sites in the mission (default: 5)
    health_loss: health lost after encountering threat without protection (default: 5)
    time_loss: time lost after using armored robot (default: 10)
    starting_health: starting health of the human (default: 100.)
    starting_time: the time at which the mission starts (default: 0.)
    health_loss_cost: the reward associated with losing health (default: -10.)
    time_loss_cost: the reward associated with losing time (default: -9.)
    threat_level: the general level of threat in the mission (default: 0.5)
    threat_seed: the seed given to the threat setter (default: 123)
    """
    parser.add_argument('--num-sites', type=int, default=5,
                        help="number of sites in the mission (default: 5)")

    parser.add_argument('--health-loss', type=float, default=5.,
                        help="health lost after encountering threat without protection (default: 5.)")

    parser.add_argument('--time-loss', type=float, default=10.,
                        help="time lost after using armored robot (default: 10.)")

    parser.add_argument('--starting-health', type=float, default=100.,
                        help="starting health of the human (default: 100.)")

    parser.add_argument('--starting-time', type=float, default=0.,
                        help='the time at which the mission starts (default: 0.)')

    parser.add_argument('--health-loss-cost', type=float, default=-10.,
                        help='the reward associated with losing health (default: -10.)')

    parser.add_argument('--time-loss-cost', type=float, default=-9.,
                        help='the reward associated with losing time (default: -9.)')

    parser.add_argument('--threat-level', type=float, default=0.5,
                        help="the general level of threat in the mission (default: 0.5)")

    parser.add_argument('--threat-seed', type=int, default=123,
                        help='the seed given to the threat setter (default: 123)')

    return parser


def add_simulated_human_args(parser: argparse.ArgumentParser):
    """
    Adds arguments related to simulated human settings
    simulated_kappa: the rationality coefficient of the simulated human (default: 0.8)
    simulated_posterior_stepsize: the stepsize of the simulated human's posterior distribution (default: 0.02)
    simulated_wh: the simulated human's health reward weight (default: 0.8)
    simulated_alpha0: alpha0 parameter of the simulated human's trust dynamics (default: 20.)
    simulated_beta0: beta0 parameter of the simulated human's trust dynamics (default: 10.)
    simulated_ws: ws parameter of the simulated human's trust dynamics (default: 5.)
    simulated_wf: wf parameter of the simulated human's trust dynamics (default: 10.)
    simulated_seed: the seed of the random number generator used for selecting actions and reporting trust (default:123)
    """
    parser.add_argument('--simulated-kappa', type=float, default=0.8,
                        help='the rationality coefficient of the simulated human (default: 0.8)')

    parser.add_argument('--simulated-posterior-stepsize', type=float, default=0.02,
                        help="the stepsize of the simulated human's posterior distribution (default: 0.02)")

    parser.add_argument('--simulated-wh', type=float, default=0.8,
                        help="the simulated human's health reward weight (default: 0.8)")

    parser.add_argument('--simulated-alpha0', type=float, default=20.,
                        help="alpha0 parameter of the simulated human's trust dynamics (default: 20.)")

    parser.add_argument('--simulated-beta0', type=float, default=10.,
                        help="beta0 parameter of the simulated human's trust dynamics (default: 10.)")

    parser.add_argument('--simulated-ws', type=float, default=5.,
                        help="ws parameter of the simulated human's trust dynamics (default: 5.)")

    parser.add_argument('--simulated-wf', type=float, default=10.,
                        help="wf parameter of the simulated human's trust dynamics (default: 10.)")

    parser.add_argument('--simulated-seed', type=int, default=123,
                        help='The seed for the random number generator for the simulated human')

    return parser


def add_model_human_args(parser: argparse.ArgumentParser):
    """
    Adds arguments related to human model settings
    model_kappa: the rationality coefficient of the human model (default: 0.8)
    model_posterior_stepsize: the stepsize of the human model's posterior distribution (default: 0.02)
    model_wh: the human model's health reward weight (default: 0.8)
    model_alpha0: alpha0 parameter of the human model's trust dynamics (default: 20.)
    model_beta0: beta0 parameter of the human model's trust dynamics (default: 10.)
    model_ws: ws parameter of the human model's trust dynamics (default: 5.)
    model_wf: wf parameter of the human model's trust dynamics (default: 10.)
    model_seed: the seed of the random number generator used for reporting trust (default:123)
    """
    parser.add_argument('--model-kappa', type=float, default=0.8,
                        help='the rationality coefficient of the human model (default: 0.8)')

    parser.add_argument('--model-posterior-stepsize', type=float, default=0.02,
                        help="the stepsize of the human model's posterior distribution (default: 0.02)")

    parser.add_argument('--model-alpha0', type=float, default=20.,
                        help="alpha0 parameter of the human model's trust dynamics (default: 20.)")

    parser.add_argument('--model-beta0', type=float, default=10.,
                        help="beta0 parameter of the human model's trust dynamics (default: 10.)")

    parser.add_argument('--model-ws', type=float, default=5.,
                        help="ws parameter of the human model's trust dynamics (default: 5.)")

    parser.add_argument('--model-wf', type=float, default=10.,
                        help="wf parameter of the human model's trust dynamics (default: 10.)")

    parser.add_argument('--model-seed', type=int, default=123,
                        help='The seed for the random number generator for the human model')

    return parser


def add_solver_args(parser: argparse.ArgumentParser):
    """
    Adds arguments related to the solver (robot's recommendation system)
    solver_wh: the health reward weight for the solver (default=0.8)
    discount_factor: the discount factor in the value iteration algorithm (default=0.9)
    """
    parser.add_argument('--solver-wh', type=float, default=0.8,
                        help='the health reward weight for the solver (default=0.8)')

    parser.add_argument('--discount-factor', type=float, default=0.9,
                        help='the discount factor in the value iteration algorithm (default=0.9)')

    return parser


def add_estimator_args(parser: argparse.ArgumentParser):
    """
    Adds arguments related to the trust parameter estimator
    estimator_num_iterations: maximum number of gradient steps to take (default: 200)
    estimator_stepsize: learning rate of the gradient descent algorithm (default: 0.0005)
    estimator_error_tolerance: error tolerance below which the algorithm stops prematurely (default: 0.01)
    """

    parser.add_argument('--estimator-num-iterations', type=int, default=200,
                        help="maximum number of gradient steps to take (default: 200)")

    parser.add_argument('--estimator-stepsize', type=float, default=0.0005,
                        help='learning rate of the gradient descent algorithm (default: 0.0005)')

    parser.add_argument('--estimator-error-tolerance', type=float, default=0.01,
                        help='error tolerance below which the algorithm stops prematurely (default: 0.01)')

    return parser

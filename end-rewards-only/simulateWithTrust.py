import numpy as np
from classes.Human import DisuseBoundedRationalModel, DisuseBoundedRationalSimulator
from classes.ThreatSetter import ThreatSetter
from classes.Solver import SolverWithTrust
from classes.IRLModel import Posterior
from classes.Rewards import ConstantRewards, EndReward
from classes.ParamsUpdater import Estimator


def main():
    # Mission settings
    num_sites = 20
    threat_level = 0.5
    health_loss = 5
    time_loss = 10

    # Human model settings
    kappa_model = 0.2
    posterior_stepsize_model = 0.02
    health_loss_cost_model = -10
    time_loss_cost_model = -9
    reward_fun_model = ConstantRewards(num_sites, health_loss_cost=health_loss_cost_model,
                                       time_loss_cost=time_loss_cost_model)

    trust_params_model = {"alpha0": None, "beta0": None, "ws": None, "wf": None}

    # Trust Params Updater settings
    num_iterations = 200
    estimator_stepsize = 0.0005
    error_tolerance = 0.01
    params_estimator = Estimator(num_iterations, estimator_stepsize, error_tolerance)

    # Robot settings
    whr = 0.8
    df = 0.9
    reward_fun_solver = EndReward(num_sites)

    # Simulated human settings
    kappa_simulated = 0.8
    posterior_stepsize_simulated = 0.02
    health_loss_cost_simulated = -10.
    time_loss_cost_simulated = -9.
    whh = 0.9
    trust_params_simulated = {"alpha0": 20, "beta0": 10, "ws": 5, "wf": 10}

    # Simulated human
    reward_fun_simulated = ConstantRewards(num_sites, health_loss_cost=health_loss_cost_simulated,
                                           time_loss_cost=time_loss_cost_simulated)
    posterior_simulated = Posterior(kappa_simulated, posterior_stepsize_simulated, reward_fun=reward_fun_simulated)
    idx = (np.abs(posterior_simulated.weights - whh).argmin())
    posterior_simulated.dist[idx] += 100                       # Increase the distribution pdf at the specified weight
    posterior_simulated.normalize()
    simulated_human = DisuseBoundedRationalSimulator(posterior_simulated, kappa_simulated,
                                                     reward_fun_simulated, trust_params_simulated,
                                                     num_sites)

    # Threat setter
    threat_setter = ThreatSetter(num_sites, threat_level, seed=123)
    threat_setter.set_threats()
    prior_levels = threat_setter.prior_levels
    after_scan_levels = threat_setter.after_scan_levels
    threats = threat_setter.threats

    # Human model for the solver
    model_posterior = Posterior(kappa=kappa_model, stepsize=posterior_stepsize_model,
                                reward_fun=reward_fun_model)
    human_model = DisuseBoundedRationalModel(model_posterior, kappa_model, reward_fun_model, trust_params_model,
                                             num_sites, params_estimator)

    # Solver with trust
    solver = SolverWithTrust(num_sites, prior_levels, after_scan_levels, whr, df, reward_fun=reward_fun_solver,
                             human_model=human_model)


if __name__ == "__main__":
    main()

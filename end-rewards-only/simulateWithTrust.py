import numpy as np
from classes.Human import DisuseBoundedRationalModel, DisuseBoundedRationalSimulator
from classes.ThreatSetter import ThreatSetter
from classes.Solver import SolverWithTrust
from classes.IRLModel import Posterior
from classes.Rewards import ConstantRewards, EndReward
from classes.ParamsUpdater import Estimator
import pandas as pd
import json


def main():
    # Mission settings
    num_sites = 5
    threat_level = 0.5
    health_loss = 5
    time_loss = 10
    health_loss_cost = -10.
    time_loss_cost = -9.

    # Human model settings
    kappa_model = 0.2
    posterior_stepsize_model = 0.02
    reward_fun_model = ConstantRewards(num_sites, health_loss_cost=health_loss_cost,
                                       time_loss_cost=time_loss_cost)
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
    whh = 0.9
    trust_params_simulated = {"alpha0": 20, "beta0": 10, "ws": 5, "wf": 10}

    # Simulated human
    reward_fun_simulated = ConstantRewards(num_sites, health_loss_cost=health_loss_cost,
                                           time_loss_cost=time_loss_cost)
    posterior_simulated = Posterior(kappa_simulated, posterior_stepsize_simulated, reward_fun=reward_fun_simulated)
    idx = (np.abs(posterior_simulated.weights - whh).argmin())
    posterior_simulated.dist[idx] += 100                       # Increase the distribution pdf at the specified weight
    posterior_simulated.normalize()
    simulated_human = DisuseBoundedRationalSimulator(posterior_simulated, kappa_simulated,
                                                     reward_fun_simulated, trust_params_simulated, num_sites,
                                                     seed=123, health=100., time_=0.,
                                                     health_loss=health_loss, time_loss=time_loss)

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
                                             num_sites, params_estimator,
                                             seed=123, health=100., time_=0.,
                                             health_loss=health_loss, time_loss=time_loss)

    # Solver with trust
    solver = SolverWithTrust(num_sites, prior_levels, after_scan_levels, whr, df, reward_fun=reward_fun_solver,
                             human_model=human_model)

    trust_fb = simulated_human.add_initial_trust()
    # Add the initial trust feedback before observing any performance of the recommendations
    solver.add_trust(trust_fb, -1)
    # Get an initial guess for the trust parameters of the human model for the solver
    solver.get_initial_guess(trust_fb)

    for i in range(num_sites):
        # Get the recommendation
        rec = solver.get_recommendation()

        # Choose the action
        action = simulated_human.choose_action(rec, threat_level=after_scan_levels[i])

        # Move the human forward, and get trust feedback
        trust_fb = simulated_human.forward(threat_obs=threats[i], action=action)

        # Move the solver forward. This updates the posterior too
        solver.add_trust(trust_fb, i)
        solver.forward(threat_obs=threats[i], action=action, threat_level=after_scan_levels[i], trust_fb=trust_fb)

    # Save stuff to a file
    # Site number, threat level prior, threat level after scan, recommendation, action, trust_fb, trust_est mean,
    # trust_est sample


if __name__ == "__main__":
    main()

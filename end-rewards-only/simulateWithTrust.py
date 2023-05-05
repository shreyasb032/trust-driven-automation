import json

import numpy as np
from datetime import datetime
import os

import pandas as pd

from classes.Human import DisuseBoundedRationalModel, DisuseBoundedRationalSimulator
from classes.IRLModel import Posterior
from classes.ParamsUpdater import Estimator
from classes.Rewards import ConstantRewards, EndReward
from classes.Solver import SolverWithTrust
from classes.ThreatSetter import ThreatSetter
from classes.Structs import *
from classes.Utils import add_common_args


def main():
    parser = add_common_args()
    args = parser.parse_args()

    # Mission settings
    mission_settings = MissionSettings(args)

    # Human model settings
    human_model_settings = HumanModelSettings(args)
    reward_fun_model = ConstantRewards(mission_settings.num_sites,
                                       health_loss_cost=mission_settings.health_loss_cost,
                                       time_loss_cost=mission_settings.time_loss_cost)
    trust_params_model = human_model_settings.trust_params

    # Trust Params Updater settings
    trust_parameter_updater_settings = ParamsEstimatorSettings(args)
    params_estimator = Estimator(trust_parameter_updater_settings.num_iters,
                                 trust_parameter_updater_settings.stepsize,
                                 trust_parameter_updater_settings.error_tolerance)

    # Solver settings
    solver_settings = SolverSettings(args)
    reward_fun_solver = EndReward(mission_settings.num_sites)

    # Simulated human settings
    simulated_human_settings = SimulatedHumanSettings(args)

    # Simulated human
    reward_fun_simulated = ConstantRewards(mission_settings.num_sites,
                                           health_loss_cost=mission_settings.health_loss_cost,
                                           time_loss_cost=mission_settings.time_loss_cost)
    posterior_simulated = Posterior(simulated_human_settings.kappa,
                                    simulated_human_settings.posterior_stepsize,
                                    reward_fun=reward_fun_simulated)
    idx = (np.abs(posterior_simulated.weights - simulated_human_settings.wh).argmin())
    posterior_simulated.dist[idx] += 100  # Increase the distribution pdf at the specified weight
    posterior_simulated.normalize()
    simulated_human = DisuseBoundedRationalSimulator(posterior_simulated,
                                                     simulated_human_settings.kappa,
                                                     reward_fun_simulated,
                                                     simulated_human_settings.trust_params,
                                                     mission_settings.num_sites,
                                                     seed=simulated_human_settings.seed,
                                                     health=mission_settings.health,
                                                     time_=mission_settings.time_,
                                                     health_loss=mission_settings.health_loss,
                                                     time_loss=mission_settings.time_loss)

    # Threat setter
    threat_setter = ThreatSetter(mission_settings.num_sites, mission_settings.threat_level,
                                 seed=mission_settings.threat_seed)
    threat_setter.set_threats()
    prior_levels = threat_setter.prior_levels
    after_scan_levels = threat_setter.after_scan_levels
    threats = threat_setter.threats

    # Human model for the solver
    model_posterior = Posterior(kappa=human_model_settings.kappa,
                                stepsize=human_model_settings.posterior_stepsize,
                                reward_fun=reward_fun_model)
    human_model = DisuseBoundedRationalModel(model_posterior,
                                             human_model_settings.kappa,
                                             reward_fun_model,
                                             trust_params_model,
                                             mission_settings.num_sites,
                                             params_estimator,
                                             seed=human_model_settings.seed,
                                             health=mission_settings.health,
                                             time_=mission_settings.time_,
                                             health_loss=mission_settings.health_loss,
                                             time_loss=mission_settings.time_loss)

    # Solver with trust
    solver = SolverWithTrust(mission_settings.num_sites,
                             prior_levels,
                             after_scan_levels,
                             solver_settings.wh,
                             solver_settings.df,
                             reward_fun=reward_fun_solver,
                             human_model=human_model)

    trust_fb = simulated_human.add_initial_trust()
    # Add the initial trust feedback before observing any performance of the recommendations
    solver.add_trust(trust_fb, -1)
    # Get an initial guess for the trust parameters of the human model for the solver
    solver.get_initial_guess(trust_fb)

    for i in range(mission_settings.num_sites):
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
    # trust_est sample, true-performance, estimated-performance

    data = {'Site-number': np.arange(mission_settings.num_sites + 1)[1:], 'Prior': prior_levels,
            'After-scan': after_scan_levels, 'threat': threats,
            'recommendations': solver.human_model.recommendation_history,
            'actions': solver.human_model.action_history, 'trust-feedback': solver.trust_feedback_history[1:],
            'trust-feedback-mean': simulated_human.trust_mean_history[1:],
            'trust-estimated-mean': solver.human_model.trust_mean_history[1:],
            'trust-estimated-sample': solver.human_model.trust_sample_history[1:],
            'true-performance': simulated_human.performance_history,
            'estimated-performance': solver.human_model.performance_history,
            'wh-estimated-mean': solver.human_model.wh_mean_history[1:],
            'alpha0': solver.human_model.trust_parameter_history['alpha0'][1:],
            'beta0': solver.human_model.trust_parameter_history['beta0'][1:],
            'ws': solver.human_model.trust_parameter_history['ws'][1:],
            'wf': solver.human_model.trust_parameter_history['wf'][1:]}

    settings = {
        "Mission-settings": mission_settings.__dict__,
        "Simulated-human-settings": simulated_human_settings.__dict__,
        "Human-model-settings": human_model_settings.__dict__,
        "Solver-settings": solver_settings.__dict__
    }

    df = pd.DataFrame(data=data)

    df_first_row = pd.DataFrame(
        [[0, None, None, None, None, None, solver.trust_feedback_history[0],
          simulated_human.trust_mean_history[0],
          solver.human_model.trust_mean_history[0],
          solver.human_model.trust_sample_history[0],
          None, None,
          solver.human_model.wh_mean_history[0],
          solver.human_model.trust_parameter_history['alpha0'][0],
          solver.human_model.trust_parameter_history['beta0'][0],
          solver.human_model.trust_parameter_history['ws'][0],
          solver.human_model.trust_parameter_history['wf'][0]]], columns=df.columns
    )

    directory = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = os.path.join('.', 'data', directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    data_file = os.path.join(directory, 'data.csv')
    settings_file = os.path.join(directory, 'settings.json')

    df = pd.concat([df_first_row, df])
    df.to_csv(data_file, index=False)

    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)


if __name__ == "__main__":
    main()

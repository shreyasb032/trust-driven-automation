"""
This file runs multiple simulations with the use of SolverWithTrust and only saves the ending health and time for
plotting
"""

import json
import pickle
import numpy as np
import os
from tqdm import tqdm
import time

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
    parser.add_argument("--wh-start", type=float,
                        help='The starting point of the health reward weight for the solver (default: 0.0)',
                        default=0.0)
    parser.add_argument("--wh-end", type=float,
                        help='The end point of the health reward weight for the solver (default: 1.0)',
                        default=1.0)
    parser.add_argument("--weight-stepsize", type=float,
                        help="The stepsize to take steps between start-wh and end-wh (default: 0.05)",
                        default=0.05)

    args = parser.parse_args()

    num_weights = int((args.wh_end - args.wh_start) / args.weight_stepsize) + 1
    wh_list = [args.wh_start + args.weight_stepsize * i for i in range(num_weights)]
    num_weights_all = int(1 / args.weight_stepsize) + 1
    wh_list_all = [args.weight_stepsize * i for i in range(num_weights_all)]
    parent_directory = "./data/WithTrust/{:1.1f}/wh_start_{:1.2f}/".format(args.threat_level, args.wh_start)

    # DO WHAT?
    # FOR EACH WH in the above lists, run multiple simulations and save the ending health and time to a file. Other
    # data is not that important right now
    # What to save? wh_robot, wh_human, ending health and ending time for each simulation

    # Mission settings
    mission_settings = MissionSettings(args)
    num_simulations = mission_settings.num_simulations

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

    # Simulated human constants
    reward_fun_simulated = ConstantRewards(mission_settings.num_sites,
                                           health_loss_cost=mission_settings.health_loss_cost,
                                           time_loss_cost=mission_settings.time_loss_cost)
    data = {"Health": np.zeros((num_simulations, ), dtype=float),
            "Time": np.zeros((num_simulations, ), dtype=float)}

    for wh_rob in wh_list:
        for wh_hum in wh_list_all:
            for sim in tqdm(range(num_simulations)):

                # Simulated human posterior
                posterior_simulated = Posterior(simulated_human_settings.kappa,
                                                simulated_human_settings.posterior_stepsize,
                                                reward_fun=reward_fun_simulated)
                idx = (np.abs(posterior_simulated.weights - wh_hum).argmin())
                posterior_simulated.dist[idx] += 100  # Increase the distribution pdf at the specified weight
                posterior_simulated.normalize()

                simulated_human = DisuseBoundedRationalSimulator(posterior_simulated,
                                                                 simulated_human_settings.kappa,
                                                                 reward_fun_simulated,
                                                                 simulated_human_settings.trust_params,
                                                                 mission_settings.num_sites,
                                                                 seed=simulated_human_settings.seed * sim,
                                                                 health=mission_settings.health,
                                                                 time_=mission_settings.time_,
                                                                 health_loss=mission_settings.health_loss,
                                                                 time_loss=mission_settings.time_loss)

                # Threat setter
                threat_setter = ThreatSetter(mission_settings.num_sites, mission_settings.threat_level,
                                             seed=mission_settings.threat_seed * sim)
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
                                                         seed=human_model_settings.seed * sim,
                                                         health=mission_settings.health,
                                                         time_=mission_settings.time_,
                                                         health_loss=mission_settings.health_loss,
                                                         time_loss=mission_settings.time_loss)

                # Solver with trust
                solver = SolverWithTrust(mission_settings.num_sites,
                                         prior_levels,
                                         after_scan_levels,
                                         wh_rob,
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
                    trust_fb = simulated_human.forward(threat_obs=threats[i], action=action, recommendation=rec)

                    # Move the solver forward. This updates the posterior too
                    solver.add_trust(trust_fb, i)
                    solver.forward(threat_obs=threats[i], action=action, threat_level=after_scan_levels[i],
                                   trust_fb=trust_fb, recommendation=rec)

                end_health = simulated_human.health
                end_time = simulated_human.time_
                data["Health"][sim] = end_health
                data["Time"][sim] = end_time

            # Save data to .pkl file
            # Save parameters to a .json file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data_directory = os.path.join(parent_directory, timestamp)

            if not os.path.exists(data_directory):
                os.makedirs(data_directory)

            data_file = os.path.join(data_directory, 'data.pkl')
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)

            json_file = os.path.join(data_directory, 'args.json')
            json_data = {"Health Weight Robot": wh_rob,
                         "Health Weight Human": wh_hum}

            with open(json_file, 'wt') as f:
                json.dump(json_data, f, indent=4)

    settings = {
        "Mission-settings": mission_settings.__dict__,
        "Simulated-human-settings": simulated_human_settings.__dict__,
        "Human-model-settings": human_model_settings.__dict__,
        "Solver-settings": solver_settings.__dict__
    }

    settings_file = os.path.join(parent_directory, 'settings.json')

    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)


if __name__ == "__main__":
    main()

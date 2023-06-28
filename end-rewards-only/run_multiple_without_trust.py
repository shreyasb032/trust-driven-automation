"""
This file runs multiple simulations with the use of NonTrustSolver and only saves the ending health and time for
plotting
"""

import json
import pickle
import numpy as np
import os
from tqdm import tqdm
import time

from classes.Rewards import EndReward
from classes.Solver import NonTrustSolver
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
                        help="The stepsize to take steps between start-wh and end-wh (default: 0.02)",
                        default=0.02)

    args = parser.parse_args()

    num_weights = int((args.wh_end - args.wh_start) / args.weight_stepsize) + 1
    wh_list = [args.wh_start + args.weight_stepsize * i for i in range(num_weights)]
    parent_directory = "./data/WithoutTrust/{:1.1f}/wh_start_{:1.2f}/".format(args.threat_level, args.wh_start)

    # DO WHAT?
    # FOR EACH WH in the above lists, run multiple simulations and save the ending health and time to a file. Other
    # data is not that important right now
    # What to save? wh_robot, wh_human, ending health and ending time for each simulation

    # Mission settings
    mission_settings = MissionSettings(args)
    num_simulations = mission_settings.num_simulations

    # Solver settings
    solver_settings = SolverSettings(args)
    reward_fun_solver = EndReward(mission_settings.num_sites)

    data = {"Health": np.zeros((num_simulations, ), dtype=float),
            "Time": np.zeros((num_simulations, ), dtype=float)}

    for wh_rob in wh_list:
        for sim in tqdm(range(num_simulations)):

            # Threat setter
            threat_setter = ThreatSetter(mission_settings.num_sites, mission_settings.threat_level,
                                         seed=mission_settings.threat_seed * sim)
            threat_setter.set_threats()
            prior_levels = threat_setter.prior_levels
            after_scan_levels = threat_setter.after_scan_levels
            threats = threat_setter.threats

            # Solver without trust
            solver = NonTrustSolver(mission_settings.num_sites,
                                    prior_levels,
                                    after_scan_levels,
                                    wh_rob,
                                    solver_settings.df,
                                    reward_fun_solver,
                                    health_loss=mission_settings.health_loss,
                                    time_loss=mission_settings.time_loss)

            for i in range(mission_settings.num_sites):
                # Get the recommendation
                action = solver.get_action()
                solver.forward(threat_obs=threats[i], action=action)

            end_health = solver.health
            end_time = solver.time_
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
            json_data = {"Health Weight Robot": wh_rob}

            with open(json_file, 'wt') as f:
                json.dump(json_data, f, indent=4)

    settings = {
        "Mission-settings": mission_settings.__dict__,
        "Solver-settings": solver_settings.__dict__
    }

    settings_file = os.path.join(parent_directory, 'settings.json')

    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)


if __name__ == "__main__":
    main()

"""
This file uses the Non trust solver. It is not human-robot collaboration, rather only robot choosing the optimal actions
The robot solves the MDP for maximizing the end-rewards according to the weights assigned.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from classes.ThreatSetter import ThreatSetter
from classes.Solver import NonTrustSolver
from classes.Rewards import EndReward
from classes.Structs import MissionSettings, SolverSettings
from classes.Utils import add_common_args


def main():

    parser = add_common_args()
    args = parser.parse_args()

    mission_settings = MissionSettings(args)
    solver_settings = SolverSettings(args)

    # Threat Setter
    threat_setter = ThreatSetter(mission_settings.num_sites, threat_level=mission_settings.threat_level,
                                 seed=mission_settings.threat_seed)
    threat_setter.set_threats()
    prior_levels = threat_setter.prior_levels
    after_scan_levels = threat_setter.after_scan_levels
    threats = threat_setter.threats

    # Solver
    reward_fun_rob = EndReward(mission_settings.num_sites)
    solver = NonTrustSolver(mission_settings.num_sites, prior_levels, after_scan_levels,
                            solver_settings.wh, solver_settings.df, reward_fun_rob,
                            health_loss=mission_settings.health_loss,
                            time_loss=mission_settings.time_loss)

    for i in range(mission_settings.num_sites):
        # choose action
        action = solver.get_action()
        # move forward
        solver.forward(threats[i], action)

    data = {'Site-number': np.arange(mission_settings.num_sites + 1)[1:], 'Prior': prior_levels,
            'After-scan': after_scan_levels, 'threat': threats,
            'recommendations': solver.action_history,
            'health-after': solver.health_history[1:],
            'time-after': solver.time_history[1:]}

    settings = {
        "Mission-settings": mission_settings.__dict__,
        "Solver-settings": solver_settings.__dict__
    }

    df = pd.DataFrame(data=data)

    df_first_row = pd.DataFrame([[0, None, None, None, None, solver.health_history[0],
                                  solver.time_history[0]]], columns=df.columns)

    directory = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = os.path.join('.', 'data', 'WithoutTrust', directory)

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

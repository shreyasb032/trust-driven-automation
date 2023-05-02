"""
This file uses the Non trust solver. It is not human-robot collaboration, rather only robot choosing the optimal actions
The robot solves the MDP for maximizing the end-rewards according to the weights assigned.
"""

import numpy as np
from classes.ThreatSetter import ThreatSetter
from classes.Solver import NonTrustSolver
from classes.Rewards import ConstantRewards, EndReward


def main():
    num_sites = 30
    threat_level = 1.0
    seed = 123
    df = 0.9
    whr = 0.6
    health_loss_reward = -10
    time_loss_reward = -9

    threat_setter = ThreatSetter(num_sites, threat_level=threat_level, seed=None)
    threat_setter.set_threats()
    prior_levels = threat_setter.prior_levels
    after_scan_levels = threat_setter.after_scan_levels
    threats = threat_setter.threats

    reward_fun_rob = EndReward(num_sites)

    solver = NonTrustSolver(num_sites, prior_levels, after_scan_levels, whr, df, reward_fun_rob)

    action_history = np.zeros((num_sites,), dtype=int)
    action_history_immediate = np.zeros((num_sites,), dtype=int)
    health_history = np.zeros((num_sites + 1,), dtype=float)
    time_history = np.zeros((num_sites + 1,), dtype=float)

    health_history[0] = solver.health
    time_history[0] = solver.time_

    for i in range(num_sites):
        action = solver.get_action()
        action_imm = solver.get_action_one_step()
        action_history[i] = action
        action_history_immediate[i] = action_imm
        solver.forward(threats[i], action)
        health_history[i + 1] = solver.health
        time_history[i + 1] = solver.time_

    row = ["Prior", "After scan", "Action", "Action immediate", "Health", "Time"]
    tmp_str = "{:^17}"
    print_str = ""
    for i in range(len(row)):
        print_str += tmp_str

    print(print_str.format(row[0], row[1], row[2], row[3], row[4], row[5]))
    print(print_str.format("", "", "", "", str(round(health_history[0])), str(round(time_history[0]))))
    for i in range(num_sites):
        print(print_str.format(str(round(prior_levels[i], 2)),
                               str(round(after_scan_levels[i], 2)),
                               str(action_history[i]),
                               str(action_history_immediate[i]),
                               str(round(health_history[i + 1])),
              str(round(time_history[i + 1]))))


if __name__ == "__main__":
    main()

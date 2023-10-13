"""
Run a number of simulations
For each simulation,
Generate trust parameters randomly,
Generate threats randomly,
Fix reward weights for the robot and the human
At each search site, get two recommendations: one based on one-step rewards and the other based on full solve.
Update trust, health, and time
Check for differences in recommendations
"""


from classes.HumanModels import BoundedRational
from classes.RewardFunctions import Constant
from classes.PerformanceMetrics import ImmediateObservedReward
from classes.IRLModel import Posterior
from classes.ParamsUpdater import Estimator
from classes.POMDPSolver import SolverConstantRewards
from classes.ThreatSetter import ThreatSetter
import numpy as np


def main():
    num_simulations = int(1e5)
    num_search_sites = 30
    trust_params = [10., 5., 20., 30.]
    reward_weights_hum = {"health": 0.7, "time": 0.3}
    reward_weights_rob = {"health": 0.8, "time": 0.2}
    est_reward_weights = {"health": 0.5, "time": 0.5}

    reward_function = Constant(-10., -10.)
    perf_metric = ImmediateObservedReward(reward_weights_hum, reward_function)
    rng = np.random.default_rng()
    differences = []
    for sim in range(num_simulations):
        # Initialize the threats
        threat_setter = ThreatSetter(prior=rng.random())
        threat_setter.set_threats()

        # Initialize the human
        human = BoundedRational(trust_params, reward_weights_hum, reward_function, perf_metric)

        # Initialize the solver
        solver = SolverConstantRewards(num_search_sites, reward_weights_rob, trust_params,
                                       threat_setter.prior, threat_setter.after_scan, threat_setter.threats,
                                       est_reward_weights, reward_function)

        # Initialize tje posterior
        posterior = Posterior(kappe=0.2, stepsize=0.02, reward_fun=reward_function)
        health = 100
        time = 0

        for site in range(num_search_sites):
            rec1 = solver.get_recommendation(site)
            rec2 = get_one_step_recommendation(solver, )

            # Forwarding stuff
            act = human.choose_action(rec1, threat_setter.after_scan[site], health, time)


if __name__ == "__main__":
    main()

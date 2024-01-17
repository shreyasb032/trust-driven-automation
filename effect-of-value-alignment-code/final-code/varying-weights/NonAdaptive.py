"""Here, the robot tries to estimate the human's weights but only uses them for performance
computation for the trust update.
The weights of the objective function of the robot are fixed. 
Further, the true trust parameters of the human are not known a priori. They are updated using gradient descent after
receiving trust feedback"""

import _context
from Utils import *
from classes.POMDPSolver import SolverConstantRewards
from classes.HumanModels import BoundedRational, ReversePsychology
from classes.IRLModel import Posterior
from classes.ThreatSetter import ThreatSetter
from classes.RewardFunctions import Constant
from classes.ParamsUpdater import Estimator
from classes.PerformanceMetrics import ImmediateObservedReward, ImmediateExpectedReward
from classes.ParamsGenerator import TrustParamsGenerator
import os
import argparse
import pickle
from tqdm import tqdm
import json


class NonAdaptiveRobot:
    
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.trust_params_generator = TrustParamsGenerator(seed=123)

    def run_one_simulation(self, seed: int):

        args = self.args
        # Output data: Trust feedback, trust estimation,
        # posterior distribution, weights, healths, times, recommendations, actions
        data = {}

        # PARAMETERS THAT CAN BE MODIFIED #############################################################################
        wh_rob = args.health_weight_robot             # Fixed health weight of the robot
        wt_rob = args.trust_weight                    # Trust increase reward weight
        kappa = args.kappa                            # Assumed rationality coefficient in the bounded rationality model
        stepsize = args.posterior_stepsize            # stepsize in the posterior
        wh_hum = args.health_weight_human             # True health weight of the human. time weight = 1 - health weight
        num_sites = args.num_sites                    # Number of sites in a mission (Horizon for planning)
        human_model_solver = args.human_model_solver  # The human model to be used by the solver
        human_model_actual = args.human_model_actual  # The human model to simulate the human

        human_model_dict = {0: 'bounded_rational',
                            1: 'rev_psych',
                            2: 'disuse'}

        # For the threat setter
        threat_level = args.threat_level

        # Reward function
        health_loss_cost = args.hl
        time_loss_cost = args.tc
        reward_fun = Constant(health_loss_cost, time_loss_cost)

        # Trust parameter estimator
        num_iterations = args.num_gradient_steps
        gradient_stepsize = args.gradient_stepsize
        err_tol = args.tolerance
        estimator = Estimator(num_iterations, gradient_stepsize, error_tol=err_tol, num_sites=num_sites)

        PRINT_FLAG = args.print_flag     # Flag to decide whether to print the data to the console output
        ##############################################################################################################

        trust_params = self.trust_params_generator.generate()
        wc_rob = 1. - wh_rob
        est_human_weights = {'health': None, 'time': None}
        rob_weights = {'health': wh_rob, 'time': wc_rob, 'trust': wt_rob}
        solver = SolverConstantRewards(num_sites,
                                       rob_weights,
                                       trust_params.copy(), None, None, None,
                                       est_human_weights,
                                       hum_mod=human_model_dict[human_model_solver],
                                       reward_fun=reward_fun, hl=args.hl, tc=args.tc, kappa=kappa)

        # Initialize posterior
        posterior = Posterior(kappa=kappa, stepsize=stepsize, reward_fun=reward_fun)

        # Initialize human model
        wc_hum = 1. - wh_hum
        human_weights = {"health": wh_hum, "time": wc_hum}

        # Initialize the performance metric
        perf_idx = args.perf_metric
        if perf_idx == 1:
            perf_metric = ImmediateObservedReward(human_weights, reward_fun)
        else:
            perf_metric = ImmediateExpectedReward(human_weights, reward_fun)

        human = None

        if human_model_actual == 0:
            human = BoundedRational(trust_params,
                                    human_weights,
                                    reward_fun=reward_fun,
                                    kappa=1.0,
                                    performance_metric=perf_metric)
        else:
            human = ReversePsychology(trust_params,
                                      human_weights,
                                      reward_fun=reward_fun,
                                      performance_metric=perf_metric)

        # THINGS TO LOOK FOR AND STORE AND PLOT/PRINT
        # Trust, posterior after every interaction, health, time, recommendation, action
        # # Initialize storage
        # N stuff
        recs = np.zeros((num_sites,), dtype=int)
        acts = np.zeros((num_sites,), dtype=int)
        weights = posterior.weights.copy()
        perf_actual = np.zeros((num_sites,), dtype=int)
        perf_est = np.zeros((num_sites,), dtype=int)
        
        # N+1 stuff
        trust_feedback = np.zeros((num_sites+1,), dtype=float)
        trust_estimate = np.zeros((num_sites+1,), dtype=float)
        times = np.zeros((num_sites+1,), dtype=int)
        healths = np.zeros((num_sites+1,), dtype=int)
        parameter_estimates = np.zeros((num_sites+1, 4), dtype=float)
        wh_means = np.zeros((num_sites+1,), dtype=float)
        wh_map = np.zeros((num_sites+1,), dtype=float)
        wh_map_prob = np.zeros((num_sites+1,), dtype=float)
        posterior_dists = np.zeros((num_sites+1, len(posterior.dist)), dtype=float)

        # Initialize health and time
        health = 100
        current_time = 0

        if PRINT_FLAG:
            # For printing purposes
            table_data = [['prior', 'after_scan', 'rec', 'action', 'health', 'time', 'trust-fb',
                           'trust-est', 'perf-hum', 'perf-rob', 'wh-mean', 'wh-map']]

        # Initialize threats
        threat_setter = ThreatSetter(num_sites, prior=threat_level, seed=seed)
        threat_setter.set_threats()
        prior = threat_setter.prior
        after_scan = threat_setter.after_scan
        threats = threat_setter.threats

        solver.update_danger(threats, prior, after_scan, reset=False)

        # Initialize the trust feedbacks and estimates.
        # This serves as a general starting state of trust for a participant (more like propensity)
        # This is before any interaction. Serves as a starting point for trust parameters
        trust_feedback[0] = human.get_feedback()

        # Get an initial guess on the parameters based on this feedback
        initial_guess = estimator.get_initial_guess(trust_feedback[0])

        # Set the solver's trust params to this initial guess
        solver.update_params(initial_guess)

        trust_estimate[0] = solver.get_trust_estimate(0)

        # For each site, get recommendation, choose action, update health, time, trust, posterior
        for i in range(num_sites):

            # Get the recommendation
            rec = solver.get_recommendation(i, posterior)

            # Choose action
            action = human.choose_action(rec, after_scan[i], health, current_time)

            # Update health, time
            time_old = current_time
            health_old = health

            if action:
                current_time += 10.
            else:
                if threats[i]:
                    health -= 10.

            # Storage
            recs[i] = rec
            acts[i] = action
            times[i] = time_old
            healths[i] = health_old

            # Update posterior (UPDATE THIS BEFORE UPDATING TRUST)
            wh_means[i] = posterior.get_mean()
            prob, weight = posterior.get_map()
            wh_map[i] = weight
            wh_map_prob[i] = prob
            posterior_dists[i, :] = posterior.dist
            posterior.update(rec, action, human.get_mean(), health_old, time_old, after_scan[i])

            # Use the old values of health and time to compute the performance
            solver.forward(i, rec, posterior)
            trust_est_after = solver.get_trust_estimate(i+1)
            trust_estimate[i+1] = trust_est_after

            # Update trust (based on old values of health and time)
            human.update_trust(rec, threats[i], health_old, time_old)
            trust_fb_after = human.get_feedback()
            trust_feedback[i+1] = trust_fb_after

            # Update trust parameters
            opt_params = estimator.get_params(solver.trust_params, solver.get_last_performance(i), trust_fb_after, i)
            solver.update_params(opt_params)
            parameter_estimates[i+1, :] = np.array(opt_params)

            # Storage
            perf_est[i] = solver.get_last_performance(i)
            perf_actual[i] = human.get_last_performance()

            if PRINT_FLAG:
                # Store stuff
                row = ["{:.2f}".format(threat_setter.prior[i]), "{:.2f}".format(threat_setter.after_scan[i]), str(rec),
                       str(action), str(health_old), str(time_old), "{:.2f}".format(trust_fb_after),
                       "{:.2f}".format(trust_est_after), str(human.get_last_performance()),
                       str(solver.get_last_performance(i)), "{:.2f}".format(posterior.get_mean()),
                       "{:.2f}".format(posterior.get_map()[1])]
                table_data.append(row)

        if PRINT_FLAG:
            # Get the values after the last site
            row = ['', '', '', '', str(health), str(current_time),
                   "{:.2f}".format(human.get_mean()),
                   "{:.2f}".format(solver.get_trust_estimate(args.num_sites)),
                   str(human.get_last_performance()), str(solver.get_last_performance(args.num_sites)),
                   "{:.2f}".format(posterior.get_mean()), "{:.2f}".format(posterior.get_map()[1])]
            table_data.append(row)
            # Print
            col_print(table_data)
        
        # Store the final values after the last house
        healths[-1] = health
        times[-1] = current_time
        wh_means[-1] = posterior.get_mean()
        prob, weight = posterior.get_map()
        wh_map[-1] = weight
        wh_map_prob[-1] = prob
        posterior_dists[-1, :] = posterior.dist

        data['trust feedback'] = trust_feedback
        data['trust estimate'] = trust_estimate
        data['health'] = healths
        data['time'] = times
        data['recommendation'] = recs
        data['actions'] = acts
        data['weights'] = weights
        data['posterior'] = posterior_dists
        data['prior threat level'] = prior
        data['after scan level'] = after_scan
        data['threat'] = threats
        data['trust parameter estimates'] = parameter_estimates
        data['mean health weight'] = wh_means
        data['map health weight'] = wh_map
        data['map health weight probability'] = wh_map_prob
        data['performance estimates'] = perf_est
        data['performance actual'] = perf_actual

        return data

    def run(self, parent_directory, timestamp):

        args = self.args

        # PARAMETERS THAT CAN BE MODIFIED ############################################################################
        num_simulations = args.num_simulations      # Number of simulations to run
        num_sites = args.num_sites                  # Number of sites in a mission (Horizon for planning)
        stepsize = args.posterior_stepsize          # Stepsize in the posterior distrbution over the weights
        num_weights = int(1.0/stepsize) + 1         # Number of weight samples in the posterior distribution
        ##############################################################################################################

        data_all = initialize_storage_dict(num_simulations, num_sites, num_weights)

        for i in tqdm(range(num_simulations)):
            data_one_simulation = self.run_one_simulation(i)
            for k, v in data_one_simulation.items():
                # print(k)
                data_all[k][i] = v

        # STORING THE DATA #####################################################################################
        data_directory = os.path.join(parent_directory, timestamp)

        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        simulation_parameter_file = os.path.join(parent_directory, "sim_params.json")
        if not os.path.exists(simulation_parameter_file):
            sim_params = vars(args).copy()
            del sim_params['health_weight_human']
            del sim_params['health_weight_robot']
            with open(simulation_parameter_file, 'wt') as f:
                json.dump(sim_params, f, indent=4)

        data_file = os.path.join(data_directory, 'data.pkl')
        with open(data_file, 'wb') as f:
            pickle.dump(data_all, f)

        json_file = os.path.join(data_directory, 'args.json')
        json_data = {"health_weight_robot": args.health_weight_robot,
                     "health_weight_human": args.health_weight_human}

        with open(json_file, 'wt') as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Non-adaptive solver that only '
                                                 'uses learnt human weights to estimate trust')

    # Add the common arguments
    parser = add_common_args(parser)

    # Add specific arguments for this script
    parser.add_argument('--health-weight-robot',
                        type=float,
                        help='Fixed health weight of the robot (default: 0.7)',
                        default=0.7)

    parser.add_argument('--health-weight-human',
                        type=float,
                        help='True health weight of the human (default: 0.9)',
                        default=0.9)

    nar = NonAdaptiveRobot(parser.parse_args())
    nar.run('.', 'test_timestamp')

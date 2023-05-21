import numpy as np
from classes.POMDPSolver import Solver
from classes.HumanModels import BoundedRational
from classes.IRLModel import Posterior
from classes.ThreatSetter import ThreatSetter
from classes.RewardFunctions import Affine
import matplotlib.pyplot as plt
import os

def main():

    # PSEUDO CODE
    # Initialize solver, posterior, threats, human model
    # Do this for different values of kappa and see its effect

    # Number of sites
    N = 10
    # A region is a group of houses with a specific value of prior threat probability
    region_size = 1
    # Prior threat probability
    # priors = [0.9, 0.3, 0.7, 0.5, 0.8, 0.8, 0.9, 0.2, 0.1, 0.8]

    # # Initialize threats
    # threat_setter = ThreatSetter(N, region_size, priors=priors, seed=123)
    # threat_setter.setThreats()
    # priors = threat_setter.priors
    # after_scan = threat_setter.after_scan
    # prior_levels = np.zeros_like(after_scan)
    # prior_levels[::region_size] = priors
    # threats = threat_setter.threats

    # Initialize the reward function
    max_health = 110 # The constant in the affine reward function
    reward_fun = Affine(max_health=max_health)

    # Intitialize solver
    wh_rob = 0.5
    wc_rob = 1-wh_rob
    wt_rob = 10.0
    trust_params = [90., 30., 20., 30.]
    est_human_weights = {'health': None, 'time': None}
    rob_weights = {'health': wh_rob, 'time': wc_rob, 'trust': wt_rob}
    solver = Solver(N, rob_weights, trust_params.copy(), None, None, None, est_human_weights, hum_mod='bounded_rational', reward_fun=reward_fun)

    # Intialize posterior
    kappa = 0.05
    posterior = Posterior(kappa=kappa, stepsize=0.05, reward_fun=reward_fun)

    # Initialize human model
    wh_hum = 0.5
    wc_hum = 1-wh_hum
    human_weights = {"health":wh_hum, "time": wc_hum}
    human = BoundedRational(trust_params, human_weights, reward_fun=reward_fun, kappa=1.0)

    # Storage for the plots
    directory = "./figures/Bounded Rationality/same weights hum and rob/kappa" + str(kappa) + "/" + str(wh_hum)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Number of repetitions
    reps = 10

    # THINGS TO LOOK FOR AND STORE AND PLOT/PRINT
    # Trust, posterior after every interaction, health, time, recommendation, action
    # # Initialize storage
    recs = np.zeros((N,), dtype=int)
    acts = np.zeros((N,), dtype=int)
    trust = np.zeros((reps, N), dtype=float)
    posterior_dists = np.zeros((N, len(posterior.dist)), dtype=float)
    times = np.zeros((N,), dtype=int)
    healths = np.zeros((N,), dtype=int)

    health = 100
    time = 0

    for j in range(reps):

        # For printing purposes
        table_data = [['prior', 'after_scan', 'rec', 'action', 'health', 'time', 'trust-fb', 'trust-est', 'perf-hum', 'perf-rob', 'wh-mean', 'wh-map']]

        # Initialize threats
        rng = np.random.default_rng(seed=j)
        priors = rng.random(N // region_size)
        # priors = [0.9] * int(N / region_size)
        threat_setter = ThreatSetter(N, region_size, priors=priors, seed=j)
        threat_setter.setThreats()
        priors = threat_setter.priors
        after_scan = threat_setter.after_scan
        prior_levels = np.zeros_like(after_scan)

        for i in range(threat_setter.num_regions):
            prior_levels[i*threat_setter.region_size:(i+1)*threat_setter.region_size] = priors[i]

        threats = threat_setter.threats

        # I am not resetting the solver or the human for now. Multiple interactions with the same solver
        # Kind of like limited horizon planning for the solver
        solver.update_danger(threats, prior_levels, after_scan, reset=False)

        # Reset the solver to remove old performance history. But, we would need new parameters
        # solver.reset(human.get_mean())
        # human.reset()

        # Intialize health and time
        # health = 100
        # time = 0

        # For each site, get recommendation, choose action, update health, time, trust, posterior
        for i in range(N):

            # Get the recommendation
            rec = solver.get_recommendation(i, health, time, posterior)
            recs[i] = rec

            # Choose action
            action = human.choose_action(rec, after_scan[i], health, time)
            acts[i] = action

            # Update health, time
            time_old = time
            health_old = health
            if action:
                time += 10.
            else:
                if threats[i]:
                    health -= 10.

            times[i] = time
            healths[i] = health

            # Update posterior (UPDATE THIS BEFORE UPDATING TRUST)
            posterior.update(rec, action, human.get_mean(), health_old, time_old, after_scan[i])

            # Use the old values of health and time to compute the performance
            trust_est_old = solver.get_trust_estimate()
            solver.forward(i, rec, health_old, time_old, posterior)
            
            # Update trust (based on old values of health and time)
            trust[j, i] = human.get_mean()
            human.update_trust(rec, threats[i], health_old, time_old)

            # Store stuff
            row = []
            row.append("{:.2f}".format(threat_setter.prior_levels[i]))
            row.append("{:.2f}".format(threat_setter.after_scan[i]))
            row.append(str(rec))
            row.append(str(action))
            row.append(str(health_old))
            row.append(str(time_old))
            row.append("{:.2f}".format(trust[j, i]))
            row.append("{:.2f}".format(trust_est_old))
            row.append(str(human.get_last_performance()))
            row.append(str(solver.get_last_performance()))
            row.append("{:.2f}".format(posterior.get_mean()))
            row.append("{:.2f}".format(posterior.get_map()[1]))
            table_data.append(row)        

        # if j == 0:
        #     print("prior, after-scan, health, time, rec, action, trust-after, mean-weight, map-weight")
        # for k in range(reps):
        #     print(" ", round(prior_levels[k], 2), "  ", round(after_scan[k], 2), "     ", healths[k], "   ", times[k], "  ", recs[k], "  ", acts[k], "      ", round(trust[j, k], 2), "      ", round(posterior.get_mean(), 2), "       ", round(posterior.get_map()[1], 2))
        
        # Get the values after the last site
        row = ['', '', '', '', str(health), str(time), "{:.2f}".format(human.get_mean()), "{:.2f}".format(solver.get_trust_estimate()), str(human.get_last_performance()), str(solver.get_last_performance()), "{:.2f}".format(posterior.get_mean()), "{:.2f}".format(posterior.get_map()[1])]
        table_data.append(row)

        # Print
        col_print(table_data)

        posterior_dists[j, :] = posterior.dist
        # print("Mean weight: ", posterior.get_mean())

    # THINGS TO LOOK FOR AND STORE AND PLOT/PRINT
    # Trust, posterior after every interaction, health, time, recommendation, action
    # Performance history (robot's estimate and reports by human)
    # Plot it

    # 1. Trust
    fig, ax = plt.subplots()
    ax.plot(trust[:, -1], linewidth=2, c='tab:blue')
    ax.set_title("Ending Trust", fontsize=16)
    ax.set_xlabel("Mission number", fontsize=14)
    ax.set_ylabel("Trust")
    ax.set_ylim([-0.05, 1.05])
    fig.savefig(directory + "/trust.png")

    # 2. Posterior
    fig, ax = plt.subplots()
    for i in range(reps):
        ax.plot(posterior.weights, posterior_dists[i, :], linewidth=2, label=i)
    
    ax.legend()
    ax.set_title("Posterior Distribution", fontsize=16)
    ax.set_xlabel(r'$w$', fontsize=14)
    ax.set_ylabel(r'$P(w)$', fontsize=14)
    fig.savefig(directory + "/posterior.png")

    # 3. Health
    # fig, ax = plt.subplots()
    # ax.plot(healths, linewidth=2)
    # ax.set_title("Health", fontsize=16)
    # ax.set_ylabel(r'h', fontsize=14)
    # ax.set_xlabel("Site number", fontsize=14)

    # 4. Time
    # fig, ax = plt.subplots()
    # ax.plot(times, linewidth=2)
    # ax.set_title("Time", fontsize=16)
    # ax.set_ylabel(r'c', fontsize=14)
    # ax.set_xlabel("Site number", fontsize=14)

    # 5. Recommendation and action
    # print("prior, after-scan, health, time, rec, action, trust-after, mean-weight, map-weight")
    # for i in range(reps):
    #     print(" ", prior_levels[i], "   ", round(after_scan[i], 2), "     ", healths[i], "   ", times[i], "  ", recs[i], "  ", acts[i], "      ", round(trust[i], 2), "      ", round(np.sum(posterior_dists[i, :] * posterior.weights), 2), "       ", posterior.weights[np.argmax(posterior_dists[i, :])])

    plt.show()

def col_print(table_data):
    
    string = ""
    for _ in range(len(table_data[0])):
        string += "{: ^12}"

    for row in table_data:
        print(string.format(*row))

if __name__ == "__main__":
    main()


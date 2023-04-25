import numpy as np
from classes.HumanModels import AlwaysAccept, BoundedRational
from classes.ThreatSetter import ThreatSetter
from classes.IRLModel import Posterior
from classes.POMDPSolver import Solver, SolverOnlyEndReward
from classes.RewardFunctions import Affine

def updateSolver(solver : Solver, setter: ThreatSetter):
    setter.setThreats()
    priors = setter.priors
    
    ############### DEBUG ###################
    # setter.threats = [0, 1]
    # setter.after_scan = [0., 1.]

    after_scan = setter.after_scan
    prior_levels = np.zeros_like(after_scan)
    for i in range(setter.num_regions):
        prior_levels[i*setter.region_size:(i+1)*setter.region_size] = priors[i]

    threats = setter.threats
    solver.update_danger(threats, prior_levels, after_scan, reset=True)

    return solver

def case1(solver: Solver, human: BoundedRational, posterior: Posterior):
    """Runs the solver with no threats"""

    # Initialize the threat setter
    N = solver.N
    region_size = N
    prior = [0.0]
    scanner_accuracy = 1.0
    setter = ThreatSetter(N, region_size, prior, scanner_accuracy=scanner_accuracy)

    # Update the danger levels of the solver
    solver = updateSolver(solver, setter)

    health = 100
    time = 0

    # Initialize storage
    table_data = [['prior', 'after_scan', 'rec', 'action', 'health', 'time', 'trust-fb', 'trust-est']]

    # Run the solver
    for i in range(N):
        # Get the recommendation
        rec = solver.get_recommendation(i, health, time, posterior)

        # Choose the action
        act = human.choose_action(rec, setter.after_scan[i], health, time)

        health_old = health
        time_old = time

        # Update health, time
        if act:
            time += 5
        elif setter.threats[i]:
            health -= 5
        
        # Move the solver foward
        # import pdb; pdb.set_trace()
        trust_est_old = solver.get_trust_estimate()
        solver.forward(i, rec, health_old, time_old, posterior)

        # Update trust
        trust_old = human.get_mean()
        human.update_trust(rec, setter.threats[i], health_old, time_old)

        # Store stuff
        row = []
        row.append("{:.2f}".format(setter.prior_levels[i]))
        row.append("{:.2f}".format(setter.after_scan[i]))
        row.append(str(rec))
        row.append(str(act))
        row.append(str(health_old))
        row.append(str(time_old))
        row.append("{:.2f}".format(trust_old))
        row.append("{:.2f}".format(trust_est_old))
        table_data.append(row)

    row = ['', '', '', '', str(health), str(time), "{:.2f}".format(human.get_mean()), "{:.2f}".format(solver.get_trust_estimate())]
    table_data.append(row)

    # Print stuff
    col_print(table_data)

def case2(solver: Solver, human: BoundedRational, posterior: Posterior):
    """Runs the solver with threats at each site"""

    # Initialize the threat setter
    N = solver.N
    region_size = N
    prior = [1.0]
    scanner_accuracy = 1.0
    setter = ThreatSetter(N, region_size, prior, scanner_accuracy=scanner_accuracy)

    # Update the danger levels of the solver
    solver = updateSolver(solver, setter)

    health = 100
    time = 0

    # Initialize storage
    table_data = [['prior', 'after_scan', 'rec', 'action', 'health', 'time', 'trust-fb', 'trust-est']]

    # Run the solver
    for i in range(N):
        # Get the recommendation
        rec = solver.get_recommendation(i, health, time, posterior)

        # Choose the action
        act = human.choose_action(rec, setter.after_scan[i], health, time)

        health_old = health
        time_old = time

        # Update health, time
        if act:
            time += 5
        elif setter.threats[i]:
            health -= 5
        
        # Move the solver foward
        trust_est_old = solver.get_trust_estimate()
        solver.forward(i, rec, health_old, time_old, posterior)

        # Update trust
        trust_old = human.get_mean()
        human.update_trust(rec, setter.threats[i], health_old, time_old)

        # Store stuff
        row = []
        row.append("{:.2f}".format(setter.prior_levels[i]))
        row.append("{:.2f}".format(setter.after_scan[i]))
        row.append(str(rec))
        row.append(str(act))
        row.append(str(health_old))
        row.append(str(time_old))
        row.append("{:.2f}".format(trust_old))
        row.append("{:.2f}".format(trust_est_old))
        table_data.append(row)

    row = ['', '', '', '', str(health), str(time), "{:.2f}".format(human.get_mean()), "{:.2f}".format(solver.get_trust_estimate())]
    table_data.append(row)

    # Print stuff
    col_print(table_data)

def case3(solver: Solver, human: BoundedRational, posterior: Posterior, prior: float):
    """Runs the solver with a specific threat probability at each site. Scanner is still fully accurate"""

    # Initialize the threat setter
    N = solver.N
    region_size = N
    prior = [prior]
    scanner_accuracy = 1.0
    setter = ThreatSetter(N, region_size, prior, scanner_accuracy=scanner_accuracy)

    # Update the danger levels of the solver
    solver = updateSolver(solver, setter)

    health = 100
    time = 0

    # Initialize storage
    table_data = [['prior', 'after_scan', 'rec', 'action', 'health', 'time', 'trust-fb', 'trust-est']]

    # Run the solver
    for i in range(N):
        # Get the recommendation
        # import pdb; pdb.set_trace()
        rec = solver.get_recommendation(i, health, time, posterior)

        # Choose the action
        act = human.choose_action(rec, setter.after_scan[i], health, time)

        health_old = health
        time_old = time

        # Update health, time
        if act:
            time += 10
        elif setter.threats[i]:
            health -= 10
        
        # Move the solver foward
        trust_est_old = solver.get_trust_estimate()
        solver.forward(i, rec, health_old, time_old, posterior)

        # Update trust
        trust_old = human.get_mean()
        human.update_trust(rec, setter.threats[i], health_old, time_old)

        # Store stuff
        row = []
        row.append("{:.2f}".format(setter.prior_levels[i]))
        row.append("{:.2f}".format(setter.after_scan[i]))
        row.append(str(rec))
        row.append(str(act))
        row.append(str(health_old))
        row.append(str(time_old))
        row.append("{:.2f}".format(trust_old))
        row.append("{:.2f}".format(trust_est_old))
        table_data.append(row)
    
    row = ['', '', '', '', str(health), str(time), "{:.2f}".format(human.get_mean()), "{:.2f}".format(solver.get_trust_estimate())]
    table_data.append(row)

    # Print stuff
    col_print(table_data)

def col_print(table_data):
    
    string = ""
    for _ in range(len(table_data[0])):
        string += "{: ^12}"

    for row in table_data:
        print(string.format(*row))

def main():
    # Initialize the solver
    N = 10
    region_size = N
    wh_rob = 0.9
    wc_rob = 1 - wh_rob
    wt_rob = 10.0
    trust_params = [90., 30., 20., 30.]
    est_human_weights = {'health': None, 'time': None}
    hum_mod = 'bounded_rational'
    reward_fun = Affine(max_health=110)
    rob_weights = {'health': wh_rob, 'time': wc_rob, 'trust': wt_rob}
    solver = Solver(N, rob_weights, trust_params, None, None, None, est_human_weights, hum_mod=hum_mod, reward_fun=reward_fun)

    # Initialize the human
    wh_hum = 0.9
    wc_hum = 1 - wh_hum
    human_weights = {"health": wh_hum, "time": wc_hum}
    human = BoundedRational(trust_params.copy(), human_weights, reward_fun=reward_fun, kappa=1)

    # Intialize the posterior
    posterior = Posterior(kappa=0.05, stepsize=0.05, reward_fun=reward_fun)

    # Give a high probability to the true weight. This is only for testing. Here we do not update the posterior
    idx = np.argwhere(posterior.weights == wh_hum)[0, 0]
    posterior.dist[idx] = 10000.0
    posterior.normalize()
    print(posterior.get_mean())

    # Case 1: If there are no threats, what does the solver do?
    # case1(solver, human, posterior)

    # Case 2: If there are threats in each house, what does the solver do?
    # case2(solver, human, posterior)

    # Case 3: Choose a specific threat probability and check what happens
    case3(solver, human, posterior, prior=0.5)
    pass

if __name__ == "__main__":
    main()

import _context
from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, walk
import json
import datetime
from scipy.stats import beta
import seaborn as sns
sns.set_theme(style='white', context='paper')


def analyze(parent_direc: str):

    # Most shapes are (num_simulations, num_sites or num_sites+1)
    # Step 1: Accumulate data
    # Get the list of subdirectories
    direc_list = []
    for root, dirs, files in walk(parent_direc):
        for dir in dirs:
            try:
                datetime.datetime.strptime(dir, "%Y%m%d-%H%M%S")
                direc_list.append(path.join(root, dir))
                # print(direc_list[-1])
            except:
                pass
    
    # Get the grid stepsize
    args_file = path.join(direc_list[0], 'args.json')
    with open(args_file, 'r') as f:
        args = json.load(f)

    grid_stepsize = args['grid_step']

    # Compute w_star
    h = args['hl']
    c = args['tc']
    wh_rob = args['health_weight_robot']
    wh_hum = args['health_weight_human']
    d_star = (1-wh_rob) * c / (wh_rob * h)

    # Initialize data
    trust_est = []
    trust_fb = []
    threat_level = []

    for direc in direc_list:
        args_file = path.join(direc, 'args.json')
        data_file = path.join(direc, 'data.pkl')
        reader = PickleReader(data_file)
        reader.read_data()
        data = reader.data
        with open(args_file, 'r') as f:
            args = json.load(f)

        d = args['threat_level']

        threat_level.append(d)
        trust_fb.append(np.mean(data['trust feedback'][:, -1]))
        trust_est.append(np.mean(data['trust estimate'][:, -1]))

    # Step 2: Plot
    # Things to plot on the 3-d graph
    # x-axis is wh_rob
    # y-axis is wh_hum
    # Plot ending trust (actual and estimate)
    fig, ax = plt.subplots()
    # ax.plot(threat_level, trust_fb, c='tab:blue', lw=3, label='Feedback')
    ax.scatter(threat_level, trust_fb, s=25, c='tab:blue', marker='o', label='Feedback')
    # ax.scatter(threat_level, trust_est, s=25, c='tab:orange', marker='o', label='estimate')
    
    if d_star < 1 and wh_hum < 0.5:
        # Region 2
        d_arr = np.array(threat_level)
        lower_bound_of_trust_increase = (1. - d_arr) * beta.cdf(d_star, 4, 28) + d_arr * beta.cdf(d_star, 28, 4)
        final_exp_alpha = 90 + lower_bound_of_trust_increase * 40 * 20
        final_exp_beta = 30 + (1 - lower_bound_of_trust_increase) * 40 * 30
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta)
        ax.plot(d_arr, final_exp_trust, lw=2, c='black', ls='dashed', label='Expected')
    elif d_star > 1 and wh_hum > 0.5:
        # Region 3
        d_arr = np.array(threat_level)
        lower_bound_of_trust_increase = (1. - d_arr) * beta.cdf(0.5, 4, 28) + d_arr * beta.cdf(0.5, 28, 4)
        final_exp_alpha = 90 + lower_bound_of_trust_increase * 40 * 20
        final_exp_beta = 30 + (1 - lower_bound_of_trust_increase) * 40 * 30
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta)
        ax.plot(d_arr, final_exp_trust, lw=2, c='black', ls='dashed', label='Expected')
    elif d_star < 1 and wh_hum > 0.5:
        # Region 4
        d_arr = np.array(threat_level)
        lower_bound_of_trust_increase = (1. - d_arr) * beta.cdf(d_star, 4, 28) + d_arr
        final_exp_alpha = 90 + lower_bound_of_trust_increase * 40 * 20
        final_exp_beta = 30 + (1 - lower_bound_of_trust_increase) * 40 * 30
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta)
        ax.plot(d_arr, final_exp_trust, lw=2, c='black', ls='dashed', label='Expected')
    else:
        # Region 1
        d_arr = np.array(threat_level)
        final_exp_alpha = 90 + 40 * 20
        final_exp_beta = 30
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta) * np.ones_like(d_arr)
        ax.plot(d_arr, final_exp_trust, lw=2, c='black', ls='dashed', label='Expected')

    ax.legend()
    ax.set_xlabel(r'$d$', fontsize=16)
    ax.set_ylabel(r'End-of-mission trust $t_N$', fontsize=16)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(r'End-of-mission trust, $w_h^r={:1.1f}$, $w_h^h={:1.1f}$'.format(wh_rob, wh_hum))

    plt.show()

def main(args: argparse.Namespace):

    # parent_direc = './data/ThreatLevel/0.7/'
    parent_direc = args.path
    analyze(parent_direc)   

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--path', type=str, help="Path to the parent directory of the data to be analysed")
    args = parser.parse_args()
    main(args)

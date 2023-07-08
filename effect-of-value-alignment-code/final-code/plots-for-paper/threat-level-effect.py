import os.path

import _context
from classes.DataReader import PickleReader
import numpy as np
import matplotlib.pyplot as plt
from os import path, walk
import json
import datetime
import seaborn as sns
from scipy.stats import beta
sns.set_theme(style='white', context='paper')


def analyze(parent_directory: str, wh_rob: float, wh_hum: float):

    # Most shapes are (num_simulations, num_sites or num_sites+1)
    # Step 1: Accumulate data
    # Get the list of subdirectories
    directory_list = []
    data_directory = os.path.join(parent_directory, 'rob{:1.1f}'.format(wh_rob), 'hum{:1.1f}'.format(wh_hum))
    for root, dirs, files in walk(data_directory):
        for directory in dirs:
            try:
                datetime.datetime.strptime(directory, "%Y%m%d-%H%M%S")
                directory_list.append(path.join(root, directory))
                # print(directory_list[-1])
            except ValueError:
                pass
    
    # Get the grid stepsize
    args_file = path.join(parent_directory, 'rob{:1.1f}'.format(wh_rob), 'hum{:1.1f}'.format(wh_hum), 'sim_params.json')
    with open(args_file, 'r') as f:
        args = json.load(f)

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

    for directory in directory_list:
        args_file = path.join(directory, 'args.json')
        data_file = path.join(directory, 'data.pkl')
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
    fig, ax = plt.subplots(layout='tight')
    ax.scatter(threat_level, trust_fb, s=64, c='tab:blue', marker='o', label='Feedback')

    trust_params = [60., 60., 10., 20.]

    line_width = 4

    if d_star < 1 and wh_hum < 0.5:
        # Region 2
        d_arr = np.array(threat_level)
        lower_bound_of_trust_increase = (1. - d_arr) * beta.cdf(d_star, 4, 28) + d_arr * beta.cdf(d_star, 28, 4)
        final_exp_alpha = trust_params[0] + lower_bound_of_trust_increase * 40 * trust_params[2]
        final_exp_beta = trust_params[1] + (1 - lower_bound_of_trust_increase) * 40 * trust_params[3]
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta)
        ax.plot(d_arr, final_exp_trust, lw=line_width, c='black', ls='dashed', label='Expected')
    elif d_star > 1 and wh_hum > 0.5:
        # Region 3
        d_arr = np.array(threat_level)
        lower_bound_of_trust_increase = (1. - d_arr) * beta.cdf(0.5, 4, 28) + d_arr * beta.cdf(0.5, 28, 4)
        final_exp_alpha = trust_params[0] + lower_bound_of_trust_increase * 40 * trust_params[2]
        final_exp_beta = trust_params[1] + (1 - lower_bound_of_trust_increase) * 40 * trust_params[3]
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta)
        ax.plot(d_arr, final_exp_trust, lw=line_width, c='black', ls='dashed', label='Expected')
    elif d_star < 1 and wh_hum > 0.5:
        # Region 4
        d_arr = np.array(threat_level)
        lower_bound_of_trust_increase = (1. - d_arr) * beta.cdf(d_star, 4, 28) + d_arr
        final_exp_alpha = trust_params[0] + lower_bound_of_trust_increase * 40 * trust_params[2]
        final_exp_beta = trust_params[1] + (1 - lower_bound_of_trust_increase) * 40 * trust_params[3]
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta)
        ax.plot(d_arr, final_exp_trust, lw=line_width, c='black', ls='dashed', label='Expected')
    else:
        # Region 1
        d_arr = np.array(threat_level)
        final_exp_alpha = trust_params[0] + 40 * trust_params[2]
        final_exp_beta = trust_params[1]
        final_exp_trust = final_exp_alpha / (final_exp_alpha + final_exp_beta) * np.ones_like(d_arr)
        ax.plot(d_arr, final_exp_trust, lw=line_width, c='black', ls='dashed', label='Expected')

    ax.legend(fontsize=16, loc='lower left')
    ax.set_xlabel(r'$d$', fontsize=16)
    ax.set_ylabel(r'End-of-mission trust $t_N$', fontsize=16)
    ax.set_ylim([-0.05, 1.05])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.set_title(r'End-of-mission trust, $w_h^r={:1.1f}$, $w_h^h={:1.1f}$'.format(wh_rob, wh_hum), fontsize=16)

    plt.show()


def main():

    parent_directory = os.path.join('..', 'varying-threat-level', 'data', 'BoundedRational', 'MidInitialTrust')
    wh_rob = 0.2
    wh_hum = 0.8
    analyze(parent_directory, wh_rob, wh_hum)


if __name__ == "__main__":

    main()

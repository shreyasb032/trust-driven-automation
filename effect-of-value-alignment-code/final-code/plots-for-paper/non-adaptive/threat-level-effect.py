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
sns.set_theme(style='whitegrid', context='paper')


def analyze(parent_directory: str, wh_rob: float, wh_hum: float, ax: plt.Axes,
            label_x: bool = False, label_y: bool = False):

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
    # fig, ax = plt.subplots(layout='tight')
    ax.scatter(threat_level, trust_fb, s=64, c='tab:blue', marker='o', label='Feedback')

    # ax.legend(fontsize=16, loc='lower left')
    if label_x:
        ax.set_xlabel(r'$d$', fontsize=16)
    if label_y:
        ax.set_ylabel(r'$t_N$', fontsize=16)

    ax.set_ylim([-0.05, 1.05])
    ax.set_title(r'$w_h^r={:1.1f}$, $w_h^h={:1.1f}$'.format(wh_rob, wh_hum), fontsize=16)


def main():

    parent_directory = os.path.join('..', '..', 'varying-threat-level', 'data', 'BoundedRational')
    # parent_directory = os.path.join('..', '..', 'varying-threat-level', 'data', 'ReversePsychology')

    # fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 6), layout='tight')
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(16, 3.5), layout='tight')

    wh_rob = 0.3
    wh_hum = 0.3
    # analyze(parent_directory, wh_rob, wh_hum, axs[0, 0], label_y=True)
    analyze(parent_directory, wh_rob, wh_hum, axs[0], label_y=True, label_x=True)

    wh_rob = 0.3
    wh_hum = 0.7
    # analyze(parent_directory, wh_rob, wh_hum, axs[0, 1])
    analyze(parent_directory, wh_rob, wh_hum, axs[1], label_x=True)

    wh_rob = 0.7
    wh_hum = 0.3
    # analyze(parent_directory, wh_rob, wh_hum, axs[1, 0], label_x=True, label_y=True)
    analyze(parent_directory, wh_rob, wh_hum, axs[2], label_x=True)

    wh_rob = 0.7
    wh_hum = 0.7
    # analyze(parent_directory, wh_rob, wh_hum, axs[1, 1], label_x=True)
    analyze(parent_directory, wh_rob, wh_hum, axs[3], label_x=True)

    plt.show()


if __name__ == "__main__":

    main()

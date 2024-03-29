import os.path
import _context
from classes.DataReader import PickleReader
import numpy as np
import matplotlib.pyplot as plt
from os import path, walk
import json
import datetime
import seaborn as sns
sns.set_theme(style='whitegrid', context='paper')


def analyze(parent_directory: str, threat_level: float, adaptive: bool = False,
            ax: plt.Axes | None = None):
    # Most shapes are (num_simulations, num_sites or num_sites+1)
    # Step 1: Accumulate data
    # Get the list of subdirectories
    directory_list = []
    data_directory = os.path.join(parent_directory, str(round(threat_level, 1)))
    # print(data_directory)
    for root, dirs, files in walk(data_directory):
        for directory in dirs:
            # print(directory)
            try:
                datetime.datetime.strptime(directory, "%Y%m%d-%H%M%S")
                directory_list.append(path.join(root, directory))
                # print(directory_list[-1])
            except ValueError:
                pass

    # print(directory_list)
    # Initialize data
    trust_est = []
    trust_fb = []
    wh_hum = []

    for directory in directory_list:
        args_file = path.join(directory, 'args.json')
        data_file = path.join(directory, 'data.pkl')
        reader = PickleReader(data_file)
        reader.read_data()
        data = reader.data
        with open(args_file, 'r') as f:
            args = json.load(f)

        # print(args_file)
        if not adaptive:
            whr = args['health_weight_robot']
            if whr != 0.5:
                continue

        whh = args['health_weight_human']
        wh_hum.append(whh)
        trust_fb.append(np.mean(data['trust feedback'][:, -1]))
        trust_est.append(np.mean(data['trust estimate'][:, -1]))

    # Step 2: Plot
    # Things to plot on the 3-d graph
    # x-axis is wh_rob
    # y-axis is wh_hum
    # Plot ending trust (actual and estimate)
    if ax is None:
        fig, ax = plt.subplots(layout='tight')

    color = 'tab:blue'
    marker = 'o'
    label = 'Non-adaptive'
    if adaptive:
        color = 'tab:orange'
        marker = '*'
        label = 'Adaptive'

    ax.scatter(wh_hum, trust_fb, s=64, c=color, marker=marker, label=label)

    return ax


def main():

    threat_level = 0.3

    parent_directory = os.path.join('..', '..', 'varying-weights', 'data', 'BoundedRational')
    ax = analyze(parent_directory, threat_level, adaptive=False, ax=None)

    parent_directory = os.path.join('..', '..', 'varying-weights', 'data', 'BoundedRational', 'Adaptive')
    ax = analyze(parent_directory, threat_level, adaptive=True, ax=ax)

    ax.legend(fontsize=16, loc='lower left')
    ax.set_xlabel(r'$w^h_h$', fontsize=16)
    ax.set_ylabel(r'$t_N$', fontsize=16)
    ax.set_ylim([-0.05, 1.05])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.set_title(r'End-of-mission trust', fontsize=16)

    # parent_directory = os.path.join('..', '..', 'varying-weights', 'data', 'ReversePsychology')
    # parent_directory = os.path.join('..', '..', 'varying-weights', 'data', 'OneStepOptimal')

    plt.show()


if __name__ == "__main__":
    main()

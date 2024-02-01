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


def analyze(parent_directory: str, wh_hum: float, ax: plt.Axes | None = None, adaptive: bool = False):
    # Most shapes are (num_simulations, num_sites or num_sites+1)
    # Step 1: Accumulate data
    # Get the list of subdirectories
    directory_list = []
    data_directory = os.path.join(parent_directory, f"rob0.5", f"hum{wh_hum:.1f}")
    if adaptive:
        data_directory = os.path.join(parent_directory, f"hum{wh_hum:.1f}")

    for root, dirs, files in walk(data_directory):
        for directory in dirs:
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
    # Plot ending trust
    if ax is None:
        fig, ax = plt.subplots(layout='tight')

    color = 'tab:blue'
    marker = 'o'
    label = 'Non-adaptive'
    if adaptive:
        color = 'tab:orange'
        marker = "*"
        label = 'Adaptive'

    ax.scatter(threat_level, trust_fb, s=64, c=color, marker=marker, label=label)

    return ax


def main():

    wh_hum = 0.7

    parent_directory = os.path.join('..', '..', 'varying-threat-level', 'data', 'BoundedRational')
    ax = analyze(parent_directory, wh_hum, ax=None, adaptive=False)

    parent_directory = os.path.join('..', '..', 'varying-threat-level', 'data', 'BoundedRational', 'Adaptive')
    ax = analyze(parent_directory, wh_hum, ax=ax, adaptive=True)

    ax.legend(fontsize=16, loc='lower left')
    ax.set_xlabel(r'$d$', fontsize=16)
    ax.set_ylabel(r'$t_N$', fontsize=16)
    ax.set_ylim([-0.05, 1.05])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.set_title(r'End-of-mission trust', fontsize=16)

    plt.show()


if __name__ == "__main__":
    main()

# Compares the trust levels for different human behavior models
import _context
from classes.DataReader import PickleReader
import numpy as np
import matplotlib.pyplot as plt
from os import path, walk
import json
import datetime
import seaborn as sns
sns.set_theme(style='white', context='paper')


def analyze(parent_directory: str, threat_level: float, color: str, behavior: str, marker: str,
            ax: plt.Axes | None = None):

    # Most shapes are (num_simulations, num_sites or num_sites+1)
    # Step 1: Accumulate data
    # Get the list of subdirectories
    directory_list = []
    data_directory = path.join(parent_directory, str(round(threat_level, 1)))
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

        whh = args['health_weight_human']
        wh_hum.append(whh)
        trust_fb.append(np.mean(data['trust feedback'][:, -1]))
        trust_est.append(np.mean(data['trust estimate'][:, -1]))

    # Step 2: Plot ending trust
    if ax is None:
        fig, ax = plt.subplots(layout='tight')

    ax.scatter(wh_hum, trust_fb, s=64, c='tab:blue', marker='o', label='Feedback')

    return ax


def main():

    threat_level = 0.3

    parent_directory = path.join('..', '..', 'varying-weights', 'data', 'BoundedRational', 'Adaptive')
    ax = analyze(parent_directory, threat_level, color='tab:blue', behavior='BRD', marker='o', ax=None)
    ax.legend(fontsize=16, loc='lower left')
    ax.set_xlabel(r'$w^h_h$', fontsize=16)
    ax.set_ylabel(r'$t_N$', fontsize=16)
    ax.set_ylim([-0.05, 1.05])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.set_title(r'End-of-mission trust', fontsize=16)

    parent_directory = path.join('..', '..', 'varying-weights', 'data', 'ReversePsychology', 'Adaptive')
    ax = analyze(parent_directory, threat_level, color='tab:orange', behavior='RP', marker='v', ax=ax)

    parent_directory = path.join('..', '..', 'varying-weights', 'data', 'OneStepOptimal', 'Adaptive')
    ax = analyze(parent_directory, threat_level, color='tab:gray', behavior='OneStep', marker='s', ax=ax)

    plt.show()


if __name__ == "__main__":
    main()

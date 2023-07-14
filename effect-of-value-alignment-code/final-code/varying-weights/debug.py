import sys

import _context
from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, walk
import json
from mpl_toolkits.mplot3d import Axes3D
import datetime
import seaborn as sns

sns.set_theme(style='white', context='paper')


def analyze(parent_direc: str, region: int):
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
            except ValueError:
                pass

    # Get the grid stepsize
    args_file = path.join(parent_direc, 'wh_start_0.00', 'sim_params.json')
    with open(args_file, 'r') as f:
        args = json.load(f)

    grid_stepsize = args['grid_step']

    # Compute w_star
    h = args['hl']
    c = args['tc']
    w_star = c / (h + c)

    # Start is included. End is not included

    # Initialize data
    wh_rob = []
    wh_hum = []
    trust_est = []
    trust_fb = []

    counts_rob = {}
    counts_hum = {}

    for direc in direc_list:
        args_file = path.join(direc, 'args.json')
        data_file = path.join(direc, 'data.pkl')
        reader = PickleReader(data_file)
        reader.read_data()
        data = reader.data
        with open(args_file, 'r') as f:
            args = json.load(f)

        _wh_hum = args['health_weight_human']
        _wh_rob = args['health_weight_robot']

        str_hum = f"{_wh_hum:.2f}"
        str_rob = f"{_wh_rob:.2f}"

        if str_hum not in counts_hum.keys():
            counts_hum[str_hum] = 1
        else:
            counts_hum[str_hum] += 1

        if str_rob not in counts_rob.keys():
            counts_rob[str_rob] = 1
        else:
            counts_rob[str_rob] += 1

        wh_hum.append(_wh_hum)
        wh_rob.append(_wh_rob)

        trust_fb.append(np.mean(data['trust feedback'][:, -1]))
        trust_est.append(np.mean(data['trust estimate'][:, -1]))

    print(counts_hum)
    print(counts_rob)
    sys.exit()
    # Step 2: Plot
    # Things to plot on the 3-d graph
    # x-axis is wh_rob
    # y-axis is wh_hum
    # Plot ending trust (actual and estimate)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wh_rob, wh_hum, trust_fb, s=25, c='tab:blue', marker='o', label='feedback')
    # ax.scatter(wh_rob, wh_hum, trust_est, s=25, c='tab:orange', marker='o', label='estimate')
    # ax.legend()
    ax.set_xlabel(r'$w_h^r$', fontsize=16)
    ax.set_ylabel(r'$w_h^h$', fontsize=16)
    ax.set_zlabel('Trust', fontsize=16)
    ax.set_zlim([-0.05, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title('End-of-mission trust')

    plt.show()


def main(args: argparse.Namespace):
    # parent_direc = './data/ThreatLevel(OldData)/0.7/'
    parent_direc = args.path
    region = args.region
    analyze(parent_direc, region)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--path', type=str, help="Path to the parent directory of the data to be analysed")
    parser.add_argument('--region', type=int, help='The region of trust to plot (0 - all regions, '
                                                   '1 - both do not care about health, '
                                                   '2 - human does not care about health,robot does, '
                                                   '3 - human cares about health, robot does not, '
                                                   '4 - both care about health) (default:0)', default=0)
    args = parser.parse_args()
    main(args)

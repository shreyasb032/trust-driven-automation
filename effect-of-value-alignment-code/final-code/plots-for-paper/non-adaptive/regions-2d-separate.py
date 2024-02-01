import sys

import _context
from classes.DataReader import PickleReader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path, walk
import json
import datetime
import seaborn as sns

sns.set_theme(style='white', context='paper')
# sns.set(font_scale=1.1)
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42


def analyze(parent_directory: str, ax: plt.Axes, cmap: str = 'plasma'):
    # Most shapes are (num_simulations, num_sites or num_sites+1)
    # Step 1: Accumulate data
    # Get the list of subdirectories
    directory_list = []
    for root, dirs, files in walk(parent_directory):
        for directory in dirs:
            try:
                datetime.datetime.strptime(directory, "%Y%m%d-%H%M%S")
                directory_list.append(path.join(root, directory))
                # print(directory_list[-1])
            except ValueError:
                pass

    # Get the grid stepsize
    args_file = path.join(parent_directory, 'wh_start_0.00', 'sim_params.json')
    with open(args_file, 'r') as f:
        args = json.load(f)

    grid_stepsize = args['grid_step']

    # Initialize data
    wh_rob = []
    wh_hum = []
    trust_est = []
    trust_fb = []

    counts_hum = {}
    counts_rob = {}

    for directory in directory_list:
        args_file = path.join(directory, 'args.json')
        data_file = path.join(directory, 'data.pkl')
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

    # print(counts_hum)
    # print(counts_rob)
    # sys.exit()
    wh_rob_list = list(set(wh_rob))
    wh_hum_list = list(set(wh_hum))
    # wh_rob_list.sort()
    # wh_hum_list.sort()
    # print(wh_rob_list)
    # print(len(wh_rob_list))
    # print(wh_hum_list)
    # print(len(wh_hum_list))
    # return
    # Step 2: Plot
    indices = set()
    # print(wh_rob)
    # print(wh_hum)

    image = np.ones((len(wh_rob_list), len(wh_hum_list)))
    for whr, whh, t in zip(wh_rob, wh_hum, trust_fb):
        idx1 = round(whr / grid_stepsize)
        idx2 = round(whh / grid_stepsize)
        if (idx1, idx2) in indices:
            # print(indices)
            print(idx1, idx2, whr, whh)
            # sys.exit()
        indices.add((idx1, idx2))
        image[idx1, idx2] = t

    # print(trust_fb)
    # ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, origin='lower', extent=(-0.05, 0.95, 0.05, 0.95))
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, origin='lower', extent=(0., 1., 0., 1.0))
    ax.set_xlabel(r'$w_h^r$', fontsize=18)
    ax.set_ylabel(r'$w_h^h$', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    return ax


def main():
    # cmap_str = 'plasma'
    # cmap_str = 'viridis'
    # cmap_str = 'inferno'
    # cmap_str = 'magma'
    # cmap_str = 'cividis'
    cmap_str = 'coolwarm'

    fig = plt.figure(layout='tight')
    ax1 = fig.add_subplot(111)
    # parent_directory = path.join("..", "..", "varying-weights", "data", "ReversePsychology", "0.7")
    parent_directory = path.join("..", "..", "varying-weights", "data", "OneStepOptimal", "0.3")
    # parent_directory = path.join("..", "..", "varying-weights", "data", "BoundedRational", "0.7")
    ax1 = analyze(parent_directory, ax1, cmap_str)

    cmap = plt.get_cmap(cmap_str)
    norm = plt.Normalize(0.0, 1.0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, location='right')
    cb.ax.set_title("scale", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    # ax1.set_title("End-of-mission Trust")

    plt.show()


if __name__ == "__main__":
    main()

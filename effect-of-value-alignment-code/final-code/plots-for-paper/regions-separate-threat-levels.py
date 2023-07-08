import _context
from classes.DataReader import PickleReader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path, walk
import json
from mpl_toolkits.mplot3d import Axes3D
import datetime
import seaborn as sns
sns.set_theme(style='white', context='paper')
# sns.set(font_scale=1.1)


def analyze(parent_directory: str, ax: Axes3D, cmap: str='plasma'):
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

    # Compute w_star
    h = args['hl']
    c = args['tc']
    w_star = c / (h + c)

    # Start is included. End is not included

    wh_hum_start = 0.05
    wh_hum_end = 0.95 + grid_stepsize
    wh_rob_start = 0.05
    wh_rob_end = 0.95 + grid_stepsize

    # Initialize data
    wh_rob = []
    wh_hum = []
    trust_est = []
    trust_fb = []

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

        wh_hum.append(_wh_hum)
        wh_rob.append(_wh_rob)

        trust_fb.append(np.mean(data['trust feedback'][:, -1]))
        trust_est.append(np.mean(data['trust estimate'][:, -1]))

    wh_rob_list = list(set(wh_rob))
    wh_rob_list.sort()
    wh_hum_list = list(set(wh_hum))
    wh_hum_list.sort()
    # print(wh_rob_list)
    # print(len(wh_rob_list))
    # print(wh_hum_list)
    # print(len(wh_hum_list))
    # return
    # Step 2: Plot
    # Things to plot on the 3-d graph
    # x-axis is wh_rob
    # y-axis is wh_hum
    # Plot ending trust (actual and estimate)
    ax.scatter3D(wh_rob, wh_hum, trust_fb, s=25, c=trust_fb, vmin=0.0, vmax=1.0,
                 cmap=mpl.colormaps[cmap], marker='o', label='feedback', depthshade=False)
    ax.set_xlabel(r'$w_h^r$', fontsize=16)
    ax.set_ylabel(r'$w_h^h$', fontsize=16)
    ax.set_zlabel('Trust', fontsize=16)
    ax.set_zlim([-0.05, 1.05])
    ax.set_xlim(wh_rob_start - grid_stepsize, wh_rob_end)
    ax.set_ylim(wh_hum_start - grid_stepsize, wh_hum_end)
    # ax.set_title(f'Threat level $d={threat_level}$')


    return ax


def main():

    cmap_str = 'plasma'
    # cmap_str = 'viridis'
    # cmap_str = 'inferno'
    # cmap_str = 'magma'
    # cmap_str = 'cividis'

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    parent_directory = path.join("..", "varying-weights", "data", "BoundedRational", "MidInitialTrust", "0.7")
    # parent_directory = path.join("..", "varying-weights", "data", "BoundedRational", "MidInitialTrust", "0.3")
    ax1 = analyze(parent_directory, ax1, cmap_str)

    cmap = plt.get_cmap(cmap_str)
    norm = plt.Normalize(0.0, 1.0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, location='left', shrink=0.6)
    cb.ax.set_title("scale")

    ax1.set_title("End-of-mission Trust")

    plt.show()


if __name__ == "__main__":
    main()

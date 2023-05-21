import _context
from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import path, walk
from Utils import col_print
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
    w_star = c / (h+c)

    # Start is included. End is not included

    wh_hum_start = 0.05
    wh_hum_end = 0.95 + grid_stepsize
    wh_rob_start = 0.05
    wh_rob_end = 0.95 + grid_stepsize

    if region == 1:
        wh_hum_end = w_star
        wh_rob_end = w_star
    elif region == 2:
        wh_hum_end = w_star
        wh_rob_start = w_star + grid_stepsize
    elif region == 3:
        wh_hum_start = w_star + grid_stepsize
        wh_rob_end = w_star
    elif region == 4:
        wh_hum_start = w_star + grid_stepsize
        wh_rob_start = w_star + grid_stepsize
        
    # Initialize data
    wh_rob = []
    wh_hum = []
    trust_est = []
    trust_fb = []

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
        
        if (_wh_hum < wh_hum_start) or (_wh_hum >= wh_hum_end) or (_wh_rob < wh_rob_start) or (_wh_rob >= wh_rob_end):
            continue
        if _wh_rob == 0 or _wh_rob == 1 or _wh_rob == w_star:
            # print('wh_rob={:.2f}'.format(_wh_rob))
            continue

        wh_hum.append(_wh_hum)
        wh_rob.append(_wh_rob)

        trust_fb.append(np.mean(data['trust feedback'][:, -1]))
        trust_est.append(np.mean(data['trust estimate'][:, -1]))

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
    ax.set_xlim([wh_rob_start - grid_stepsize, wh_rob_end])
    ax.set_ylim([wh_hum_start - grid_stepsize, wh_hum_end])
    ax.set_title('End-of-mission trust')

    plt.show()

def main(args: argparse.Namespace):

    # parent_direc = './data/ThreatLevel/0.7/'
    parent_direc = args.path
    region = args.region
    analyze(parent_direc, region)   

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--path', type=str, help="Path to the parent directory of the data to be analysed")
    parser.add_argument('--region', type=int, help='The region of trust to plot (0 - all regions, 1 - both do not care about health, 2 - human does not care about health,robot does, 3 - human cares about health, robot does not, 4 - both care about health) (default:0)', default=0)
    args = parser.parse_args()
    main(args)

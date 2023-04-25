import _context
from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import path, walk
from classes.Utils import col_print
import json
from mpl_toolkits.mplot3d import Axes3D
import datetime

def analyze(parent_direc: str, region: int):

    # Most shapes are (num_simulations, num_missions, num_sites or num_sites+1)
    # Step 1: Accumulate data
    
    # Get the list of subdirectories
    direc_list = []
    for root, dirs, files in walk(parent_direc):
        for dir in dirs:
            try:
                datetime.datetime.strptime(dir, "%Y%m%d-%H%M%S")
                direc_list.append(path.join(root, dir))
                print(direc_list[-1])
            except:
                pass
        
    # Get the grid stepsize
    args_file = path.join(direc_list[0], 'args.json')
    with open(args_file, 'r') as f:
        args = json.load(f)
    
    grid_stepsize = args['grid_step']
    num_weights = int(1/grid_stepsize) + 1
    
    # Compute w_star
    h = args['hl']
    c = args['tc']
    w_star = c / (h+c)
    
    # Startis included. End is not included
    
    wh_hum_start = 0.0
    wh_hum_end = 1.0 + grid_stepsize
    wh_rob_start = 0.0
    wh_rob_end = 1.0 + grid_stepsize
    
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
    
    num_weights = int((wh_rob_end - wh_rob_start)/grid_stepsize) + 1
    
    # Initialize data
    trust_fb = np.zeros((num_weights, num_weights), dtype=float)
    trust_est = np.zeros((num_weights, num_weights), dtype=float)
    # trust_fb_std = np.zeros((num_weights, num_weights), dtype=float)
    # trust_est_std = np.zeros((num_weights, num_weights), dtype=float)

    # health = np.zeros((num_weights, num_weights), dtype=float)
    # sim_time = np.zeros((num_weights, num_weights), dtype=float)
    # health_std = np.zeros((num_weights, num_weights), dtype=float)
    # sim_time_std = np.zeros((num_weights, num_weights), dtype=float)

    # wh_est_mean = np.zeros((num_weights, num_weights), dtype=float)
    # wh_est_mean_std = np.zeros((num_weights, num_weights), dtype=float)

    wh_rob = np.zeros((num_weights, num_weights), dtype=float)
    wh_hum = np.zeros((num_weights, num_weights), dtype=float)    

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

        idx1 = int((_wh_hum-wh_hum_start) / grid_stepsize)
        idx2 = int((_wh_rob-wh_rob_start) / grid_stepsize)
        wh_hum[idx1, idx2] = _wh_hum
        wh_rob[idx1, idx2] = _wh_rob

        trust_fb[idx1, idx2] = np.mean(data['trust feedback'][:, -1, -1])
        # trust_fb_std[idx1, idx2] = np.std(data['trust feedback'][:, -1, -1])

        trust_est[idx1, idx2] = np.mean(data['trust estimate'][:, -1, -1])
        # trust_est_std[idx1, idx2] = np.std(data['trust estimate'][:, -1, -1])
        
        # health[idx1, idx2] = np.mean(data['health'][:, -1, -1])
        # health_std[idx1, idx2] = np.std(data['health'][:, -1, -1])
        
        # sim_time[idx1, idx2] = np.mean(data['time'][:, -1, -1])
        # sim_time_std[idx1, idx2] = np.std(data['time'][:, -1, -1])
        
        # wh_est_mean[idx1, idx2] = np.mean(data['mean health weight'][:, -1, -1])
        # wh_est_mean_std[idx1, idx2] = np.std(data['mean health weight'][:, -1, -1])

    # Step 2: Plot
    # Things to plot on the 3-d graph
    # x-axis is wh_rob
    # y-axis is wh_hum
    # Plot ending trust (actual and estimate)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wh_rob.flatten(), wh_hum.flatten(), trust_fb.flatten(), c='tab:blue', marker='o', label='feedback')
    ax.scatter(wh_rob.flatten(), wh_hum.flatten(), trust_est.flatten(), c='tab:orange', marker='o', label='estimate')
    # ax.plot_surface(wh_rob.flatten(), wh_hum.flatten(), trust_fb.flatten(), c='tab:blue', label='feedback')
    # ax.plot_surface(wh_rob.flatten(), wh_hum.flatten(), trust_est.flatten(), c='tab:orange', label='estimate')
    ax.legend()
    ax.set_xlabel(r'$w_h^r$', fontsize=16)
    ax.set_ylabel(r'$w_h^h$', fontsize=16)
    ax.set_zlabel('Trust', fontsize=16)    
    ax.set_zlim([-0.05, 1.05])
    ax.set_xlim([wh_rob_start - grid_stepsize, wh_rob_end])
    ax.set_ylim([wh_hum_start - grid_stepsize, wh_hum_end])
    ax.set_title('Ending trust')

    # Plot ending health
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(wh_rob.flatten(), wh_hum.flatten(), health.flatten(), c='tab:blue', marker='o', label='health')
    # ax.set_xlabel(r'$w_h^r$', fontsize=16)
    # ax.set_ylabel(r'$w_h^h$', fontsize=16)
    # ax.set_zlabel('Health', fontsize=16)    
    # ax.set_title('Ending health')

    # # Plot ending time
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(wh_rob.flatten(), wh_hum.flatten(), sim_time.flatten(), c='tab:blue', marker='o', label='time')
    # ax.set_xlabel(r'$w_h^r$', fontsize=16)
    # ax.set_ylabel(r'$w_h^h$', fontsize=16)
    # ax.set_zlabel('Time', fontsize=16)    
    # ax.set_title('Ending time')

    # # Plot learned weight
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(wh_rob.flatten(), wh_hum.flatten(), wh_est_mean.flatten(), c='tab:blue', marker='o', label=r'$w_h^{est}$')
    # ax.set_xlabel(r'$w_h^r$', fontsize=16)
    # ax.set_ylabel(r'$w_h^h$', fontsize=16)
    # ax.set_zlabel(r'$w_h^{est}$', fontsize=16)    
    # ax.set_title('Learned Health Weight')

    plt.show()

def main(args: argparse.Namespace):

    # parent_direc = './data/non-adaptive-learner/'
    parent_direc = args.path
    region = args.region
    analyze(parent_direc, region)   

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--path', type=str, help="Path to the parent directory of the data to be analysed")
    parser.add_argument('--region', type=int, help='The region of trust to plot (0 - all regions, 1 - both do not care about health, 2 - human does not care about health,robot does, 3 - human cares about health, robot does not, 4 - both care about health) (default:0)', default=0)
    args = parser.parse_args()
    main(args)

import _context
from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import path, walk
from classes.Utils import col_print
import json

def analyze(parent_direc: str, save_figs: bool, arg_name: str):

    # Most shapes are (num_simulations, num_missions, num_sites or num_sites+1)
    # Step 1: Accumulate data
    
    # Get the list of subdirectories
    direc_list = []
    for root, dirs, files in walk(parent_direc):
        for dir in dirs:
            direc_list.append(path.join(root, dir))
        
    # Get the grid stepsize
    args_file = path.join(direc_list[0], 'args.json')
    with open(args_file, 'r') as f:
        args = json.load(f)
    
    grid_stepsize = args['grid_step']
    num_weights = int(1/grid_stepsize) + 1
    
    #Initialize data
    trust_fb = np.zeros((num_weights,), dtype=float)
    trust_est = np.zeros((num_weights,), dtype=float)
    trust_fb_std = np.zeros((num_weights,), dtype=float)
    trust_est_std = np.zeros((num_weights,), dtype=float)

    health = np.zeros((num_weights,), dtype=float)
    sim_time = np.zeros((num_weights,), dtype=float)
    health_std = np.zeros((num_weights,), dtype=float)
    sim_time_std = np.zeros((num_weights,), dtype=float)

    wh_est_mean = np.zeros((num_weights,), dtype=float)
    wh_est_mean_std = np.zeros((num_weights,), dtype=float)

    wh_hum = np.zeros((num_weights,), dtype=float)    

    for direc in direc_list:
        args_file = path.join(direc, 'args.json')
        data_file = path.join(direc, 'data.pkl')
        reader = PickleReader(data_file)
        reader.read_data()
        data = reader.data
        with open(args_file, 'r') as f:
            args = json.load(f)

        _wh_hum = args[arg_name]
        
        idx1 = int(_wh_hum / grid_stepsize)
        wh_hum[idx1] = _wh_hum

        trust_fb[idx1] = np.mean(data['trust feedback'][:, -1, -1])
        trust_fb_std[idx1] = np.std(data['trust feedback'][:, -1, -1])

        trust_est[idx1] = np.mean(data['trust estimate'][:, -1, -1])
        trust_est_std[idx1] = np.std(data['trust estimate'][:, -1, -1])
        
        health[idx1] = np.mean(data['health'][:, -1, -1])
        health_std[idx1] = np.std(data['health'][:, -1, -1])
        
        sim_time[idx1] = np.mean(data['time'][:, -1, -1])
        sim_time_std[idx1] = np.std(data['time'][:, -1, -1])
        
        wh_est_mean[idx1] = np.mean(data['mean health weight'][:, -1, -1])
        wh_est_mean_std[idx1] = np.std(data['mean health weight'][:, -1, -1])

    # Step 2: Plot
    # Things to plot on the 2-d graph
    # x-axis is wh_hum
    # Plot ending trust (actual and estimate)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(wh_hum, trust_fb, c='tab:blue', marker='o', label='feedback')
    ax.scatter(wh_hum, trust_est, c='tab:orange', marker='o', label='estimate')
    ax.legend()
    ax.set_xlabel(r'$w_h^h$', fontsize=16)
    ax.set_ylabel('Trust', fontsize=16)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('Ending trust')

    # Plot ending health
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(wh_hum, health, c='tab:blue', label='health')
    ax.set_xlabel(r'$w_h^h$', fontsize=16)
    ax.set_ylabel(r'Health', fontsize=16)
    ax.set_title('Ending health')

    # Plot ending time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(wh_hum, sim_time, c='tab:blue', label='time')
    ax.set_xlabel(r'$w_h^h$', fontsize=16)
    ax.set_ylabel('Time', fontsize=16)
    ax.set_title('Ending time')

    # Plot learned weight
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(wh_hum, wh_est_mean, c='tab:blue', label=r'$w_h^{est}$')
    ax.set_xlabel(r'$w_h^h$', fontsize=16)
    ax.set_ylabel(r'$w_h^{est}$', fontsize=16)
    ax.set_title('Learned Health Weight')

    plt.show()

def main(args: argparse.Namespace):

    parent_direc = './data/adaptive-learner/'
    arg_name = 'health_weight_human'

    if args.type == 'a-priori':
        parent_direc = './data/a-priori-alignment/'
        arg_name = 'health_weight'

    save_figs = args.save_figs
    analyze(parent_direc, save_figs, arg_name)                

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--save-figs', type=bool, help="Flag to set or unset saving figures (default: False)", default=True)
    parser.add_argument('--type', type=str, help="Whether to plot adaptive or a-priori (default: adaptive)", default='adaptive')
    args = parser.parse_args()
    main(args)

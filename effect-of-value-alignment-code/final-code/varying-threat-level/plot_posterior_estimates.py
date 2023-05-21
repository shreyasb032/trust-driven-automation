import _context
from classes.DataReader import PickleReader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='whitegrid')
from os import path, walk
import json
import datetime
import argparse

def analyze(parent_direc: str):
    
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
    
    args_file = path.join(direc_list[0], 'args.json')
    with open(args_file, 'r') as f:
        args = json.load(f)

    wh_hum = args['health_weight_human']
    
    # Initialize data
    mean_estimates = [[], [], []]
    map_estimates = [[], [], []]
    threat_level = []
    
    for direc in direc_list:
        args_file = path.join(direc, 'args.json')
        data_file = path.join(direc, 'data.pkl')
        reader = PickleReader(data_file)
        reader.read_data()
        data = reader.data
        with open(args_file, 'r') as f:
            args = json.load(f)

        d = args['threat_level']
        threat_level.append(d)

        mean_estimates[0].append(np.mean(data["mean health weight"][:, 1], axis=0))
        mean_estimates[1].append(np.mean(data["mean health weight"][:, 20], axis=0))
        mean_estimates[2].append(np.mean(data["mean health weight"][:, -1], axis=0))
        
        map_estimates[0].append(np.mean(data["map health weight"][:, 1], axis=0))
        map_estimates[1].append(np.mean(data["map health weight"][:, 20], axis=0))
        map_estimates[2].append(np.mean(data["map health weight"][:, -1], axis=0))

    fig, ax = plt.subplots(layout='tight', figsize=(9, 7))
    palette = sns.color_palette('deep')
    
    ax.scatter(threat_level, mean_estimates[0], s=25, marker='o', label='0', color=palette[0])
    ax.scatter(threat_level, mean_estimates[1], s=25, marker='o', label='20', color=palette[1])
    ax.scatter(threat_level, mean_estimates[2], s=25, marker='o', label='40', color=palette[2])

    # ax.scatter(threat_level, map_estimates[0], s=25, marker='*', label='0', color=palette[0])
    # ax.scatter(threat_level, map_estimates[1], s=25, marker='*', label='20', color=palette[1])
    # ax.scatter(threat_level, map_estimates[2], s=25, marker='*', label='40', color=palette[2])
    
    ax.plot(threat_level, [wh_hum] * len(threat_level), lw=2, ls='dotted', c='black')
    
    # h1, = ax.plot([], [], marker='*', color='black')
    # h2, = ax.plot([], [], marker='o', color='black')
    
    h3, = ax.plot([], [], c=palette[0])
    h4, = ax.plot([], [], c=palette[1])
    h5, = ax.plot([], [], c=palette[2])
    
    ax.legend(handles = [h3, h4, h5], labels=['0', '20', '40'])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Prior Threat level d')
    ax.set_ylabel('Estimate of health weight')
    
    plt.show()

def main(args: argparse.Namespace):
    
    # parent_direc = './data/ThreatLevel/0.7/'
    parent_direc = args.path
    analyze(parent_direc)   

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--path', type=str, help="Path to the parent directory of the data to be analysed")
    args = parser.parse_args()
    main(args)

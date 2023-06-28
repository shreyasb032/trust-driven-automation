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
from copy import copy
import datetime

def analyze(parent_direc: str):

    # Most shapes are (num_simulations, num_missions, num_sites or num_sites+1)    

    # Step 1: Accumulate data
    
    # Get the list of subdirectories
    direc_list = []
    for root, dirs, _ in walk(parent_direc):
        for dir in dirs:
            try:
                datetime.datetime.strptime(dir, "%Y%m%d-%H%M%S")
                direc_list.append(path.join(root, dir))
            except:
                pass

    # Get the metadata
    args_file = path.join(direc_list[0], 'args.json')
    with open(args_file, 'r') as f:
        args = json.load(f)

    # Compute d_star
    h = args['hl']
    c = args['tc']
    wh_rob = args['health_weight_robot']
    d_star = ((1. - wh_rob) * c) / (wh_rob * h)
    

    for direc in direc_list:
        table_data = []
        row = ["d", "d_star", "Recommendation", "Action", "Threat", "Trust-fb", "Trust-est", "Perf-est", "Perf-act", "wh_est"]
        table_data.append(row)

        args_file = path.join(direc, 'args.json')
        with open(args_file, 'r') as f:
            args = json.load(f)

        data_file = path.join(direc, 'data.pkl')
        reader = PickleReader(data_file)
        reader.read_data()
        data = reader.data
        
        i = 6
        print("********************SIMULATION {:d}************************".format(i))

        trust_fb = data['trust feedback'][i, :-1]
        trust_est = data['trust estimate'][i, :-1]
        recs = data['recommendation'][i,:]
        acts = data['actions'][i, :]
        ds = data['after scan level'][i, :]
        threats = data['threat'][i, :]
        perf_est = data['performance estimates'][i, :]
        perf_act = data['performance actual'][i, :]
        wh_mean = data['mean health weight'][i, 1:]
        
        for fb, est, rec, act, d, threat, pe, pa, wh in zip(trust_fb, trust_est, recs, acts, ds, threats, perf_est, perf_act, wh_mean):
            row = []
            row.append("{:.2f}".format(d))
            row.append("{:.2f}".format(d_star))
            row.append("{:d}".format(rec))
            row.append("{:d}".format(act))
            row.append("{:d}".format(threat))
            row.append("{:.2f}".format(fb))
            row.append("{:.2f}".format(est))
            row.append("{:d}".format(pe))
            row.append("{:d}".format(pa))
            row.append("{:.2f}".format(wh))
            table_data.append(copy(row))
        
        col_print(table_data)

        input("Press Enter to continue...")


def main(args: argparse.Namespace):

    # parent_direc = './data/non-adaptive-learner/'
    parent_direc = args.path
    analyze(parent_direc)

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')
    parser.add_argument('--path', type=str, help="Path to the parent directory of the data to be analysed")
    args = parser.parse_args()
    main(args)

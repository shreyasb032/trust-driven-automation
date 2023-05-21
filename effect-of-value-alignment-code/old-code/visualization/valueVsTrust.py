import numpy as np
import matplotlib.pyplot as plt
import _context
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, NumericInput
from bokeh.plotting import figure
from classes.Utils import simParams
from typing import Dict
from copy import deepcopy

# What I want to do here: Plot the value function as a function of trust
# How to do that? Define trust parameters, number of stages, discount factor, reward function
# Compute possible value of trust and plot the value
# Given a trust value, can we compute the next stage trust value? Yes based on the performance and trust params
# Can we express the value function as a function of trust? Maybe
# Value at last stage depends on: Threat Level, Reward Function, Trust, Human's Reward function,
# So, for a given trust value, I can compute this for a=0 and a=1. The value will be the max of these two
# Going backward, 

# For ease of plotting, lets assume that N=4 here. We want to vary theta, threat levels, reward weights for the robot and the human
# And see how the value function changes
# I want to plot the q-value for each action, and the value function on the same plot

# TODO: Make it interactive using bokeh/plotly. Parameters: wh_hum, wh_rob, threat_level, df, kappa, hl, tc
# Parameters:
# wh_hum - slider between 0.0 and 1.0 with 0.05 steps
# wh_rob - slider between 0.0 and 1.0 with 0.05 steps
# threat_level - slider between 0.0 and 1.0 with 0.1 steps
# df - slider between 0.0 and 1.0 with 0.05 steps
# kappa - numeric input
# hl - numeric input
# tc - numeric input
# k - site to plot

def get_lists(trust_params, sim_params, lists):
    
    wc_hum = 1.0 - sim_params.wh_hum
    wc_rob = 1.0 - sim_params.wh_rob
    N = sim_params.N

    value_matrix = np.zeros((N+1, N+1), dtype=float)
    action_matrix = np.zeros((N, N+1), dtype=int)
    # q_value_matrix = np.zeros((N+1, N+1, 2), dtype=float)
    # Compute the q-value function
    # Compute the trust
    # Plot the q-value function vs trust
    for t in reversed(range(N)):

        # Possible vals at stage t
        possible_alphas = trust_params[0] + np.arange(t+1) * trust_params[2]
        possible_betas = trust_params[1] + (t - np.arange(t+1)) * trust_params[3]

        # Compute some extra values if the human model is disuse or bounded rational
        # Estimated expected immediate rewards for human for choosing to NOT USE and USE RARV respectively

        # The below are expected rewards based on the threat level
        r0_hum = -sim_params.wh_hum * sim_params.hl * sim_params.threat_level
        r1_hum = -wc_hum * sim_params.tc

        # The below are actual observable rewards based on threat presence
        r0_no_threat = 0
        r0_threat = -sim_params.wh_hum * sim_params.hl

        # Probability of NOT USING RARV (Proportional to)
        p0 = np.exp(sim_params.kappa * r0_hum)
        # Probability of USING RARV (Proportional to)
        p1 = np.exp(sim_params.kappa * r1_hum)

        # Normalizing
        p0 /= (p0+p1)
        p1 = 1. - p0

        for i, alpha in enumerate(possible_alphas):

            beta = possible_betas[i]
            trust = alpha / (alpha + beta)
            lists['trust'][t].append(trust)

            phl = 0.
            pcl = 0.
            ptl = 0.

            ######### CASE 1: Expected reward-to-go to recommend to NOT USE RARV ###########
            # Probability of health loss
            # Probability of NOT USING RARV * Probability of Threat Presence
            phl = (trust + (1. - trust) * p0) * sim_params.threat_level

            # Probability of time loss
            # Probability of using RARV
            pcl = (1. - trust) * p1

            # Expected immediate reward to recommend to not use RARV
            r0 = -phl * sim_params.wh_rob * sim_params.hl - pcl * wc_rob * sim_params.tc

            # probability of trust gain
            pti = sim_params.threat_level * int(r0_threat > r1_hum) + (1.0 - sim_params.threat_level) * int(r0_no_threat > r1_hum)
            
            # probability of trust loss
            ptl = 1. - pti
            
            # Trust gain reward
            trust_gain_reward = pti * sim_params.wt_rob * np.sqrt(N - t)
            
            # Add the trust gain reward
            r0 += trust_gain_reward

            # Store the expected one step reward for not using RARV
            lists['one_step_0'][t].append(r0)

            # Trust increase
            next_stage_reward = pti * value_matrix[t+1, i+1]

            # Trust decrease
            next_stage_reward += ptl * value_matrix[t+1, i]
            
            # Add the discounted future reward
            r0 += sim_params.df * next_stage_reward

            ############### Expected reward to go to recommend to USE RARV #############
            # Probability of health loss
            # Probability of NOT USING RARV * Probability of Threat Presence
            phl = (1. - trust) * p0 * sim_params.threat_level

            # Probability of time loss
            # Probability of using RARV
            pcl = trust + (1. - trust) * p1

            # Probability of trust gain
            pti = sim_params.threat_level * int(r0_threat < r1_hum) + (1.0 - sim_params.threat_level) * int(r0_no_threat < r1_hum)

            # Probability of trust loss
            ptl = 1. - pti

            # Expected immediate reward to recommend to USE RARV
            r1 = -phl * sim_params.wh_rob * sim_params.hl - pcl * wc_rob * sim_params.tc

            # Trust gain reward
            trust_gain_reward = pti * sim_params.wt_rob * np.sqrt(N - t)
            
            # Add the trust gain reward
            r1 += trust_gain_reward
            
            # Store the one step reward for using the RARV
            lists['one_step_1'][t].append(r1)

            # Trust increase
            next_stage_reward =  pti * value_matrix[t+1, i+1] 

            # Trust decrease
            next_stage_reward += ptl * value_matrix[t+1, i]
            
            # Add the discounted future reward
            r1 += sim_params.df * next_stage_reward

            action_matrix[t, i] = int(r1 > r0)
            lists['action'][t].append(action_matrix[t, i])

            value_matrix[t, i] = max(r1, r0)
            lists['value'][t].append(value_matrix[t, i])

            lists['q_val_0'][t].append(r0)
            lists['q_val_1'][t].append(r1)
    
    return lists

def setup_widgets():
    widgets = {}
    
    # Inputs
    wh_hum_inp = NumericInput(title="wh_hum", value=0.7, mode='float')
    widgets['wh_hum_inp'] = wh_hum_inp
    
    wh_rob_inp = NumericInput(title="wh_rob", value=0.7, mode='float')
    widgets['wh_rob_inp'] = wh_rob_inp

    wt_rob_inp = NumericInput(title="wt_rob", value=0.0, mode='float')
    widgets['wt_rob_inp'] = wt_rob_inp

    threat_level_inp = NumericInput(title='threat_level', value=0.7, mode='float')
    widgets['threat_level_inp'] = threat_level_inp

    df_inp = NumericInput(title='df', value=0.8, mode='float')
    widgets['df_inp'] = df_inp

    kappa_inp = NumericInput(title='kappa', value=0.2, mode='float')
    widgets['kappa_inp'] = kappa_inp

    hl_inp = NumericInput(title='hl', value=10.0, mode='float')
    widgets['hl_inp'] = hl_inp

    tc_inp = NumericInput(title='tc', value=10.0, mode='float')
    widgets['tc_inp'] = tc_inp

    k_inp = NumericInput(title='site_num', value=0, mode='int')
    widgets['k_inp'] = k_inp
    
    N_inp = NumericInput(title='N', value=100, mode='int')
    widgets['N_inp'] = N_inp

    return widgets

def update_data(lists: Dict, sim_params: simParams):

    for alpha_0, beta_0 in zip(sim_params.alpha_0_list, sim_params.beta_0_list):
            for ws in sim_params.ws_list:
                for wf in sim_params.wf_list:
                    trust_params = [alpha_0, beta_0, ws, wf]
                    lists = get_lists(trust_params, sim_params, lists)

    return lists

def update_sim_params(sim_params: simParams, widgets: Dict):

    # Human's health weight    
    sim_params.wh_hum = widgets['wh_hum_inp'].value

    # Robot's health and trust weights
    sim_params.wh_rob = widgets['wh_rob_inp'].value
    sim_params.wt_rob = widgets['wt_rob_inp'].value
    
    sim_params.hl = widgets['hl_inp'].value
    sim_params.tc = widgets['tc_inp'].value
    sim_params.threat_level = widgets['threat_level_inp'].value
    sim_params.kappa = widgets['kappa_inp'].value
    sim_params.df = widgets['df_inp'].value
    sim_params.k = widgets['k_inp'].value
    sim_params.N = widgets['N_inp'].value

    return sim_params

def sort_lists(lists: Dict):
    
    sorted_arrays_dict = deepcopy(lists)

    for i, trust_list in enumerate(lists['trust']):
        trust_array = np.array(trust_list)
        idxs = np.argsort(trust_array)

        val_array = np.array(lists['value'][i])
        q_val_0_array = np.array(lists['q_val_0'][i])
        q_val_1_array = np.array(lists['q_val_1'][i])
        action_array = np.array(lists['action'][i])
        one_step_0_array = np.array(lists['one_step_0'][i])
        one_step_1_array = np.array(lists['one_step_1'][i])

        trust_array = np.take_along_axis(trust_array, idxs, axis=0)
        val_array = np.take_along_axis(val_array, idxs, axis=0)
        q_val_0_array = np.take_along_axis(q_val_0_array, idxs, axis=0)
        q_val_1_array = np.take_along_axis(q_val_1_array, idxs, axis=0)
        action_array = np.take_along_axis(action_array, idxs, axis=0)
        one_step_0_array = np.take_along_axis(one_step_0_array, idxs, axis=0)
        one_step_1_array = np.take_along_axis(one_step_1_array, idxs, axis=0)

        sorted_arrays_dict['trust'][i] = trust_array
        sorted_arrays_dict['value'][i] = val_array
        sorted_arrays_dict['q_val_0'][i] = q_val_0_array
        sorted_arrays_dict['q_val_1'][i] = q_val_1_array
        sorted_arrays_dict['action'][i] = action_array
        sorted_arrays_dict['one_step_0'][i] = one_step_0_array
        sorted_arrays_dict['one_step_1'][i] = one_step_1_array
    
    return sorted_arrays_dict


def update_sources(attr_name, old, new):

    global sim_params, lists, sorted_arrays_dict
    
    for data_lists in lists.values():
        for data_list in data_lists:
            data_list.clear()

    sim_params = update_sim_params(sim_params, widgets)
    lists = update_data(lists, sim_params)
    sorted_arrays_dict = sort_lists(lists)
    k = widgets['k_inp'].value
    val_source.data = dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['value'][k])
    q_0_source.data = dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['q_val_0'][k])
    q_1_source.data = dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['q_val_1'][k])
    one_step_0_source.data = dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['one_step_0'][k])
    one_step_1_source.data = dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['one_step_1'][k])

################ Create empty simulation parameters #############
sim_params = simParams()
#################################################################
################# Setup the widgets #############################
widgets = setup_widgets()
#################################################################
################## Setup the constants ##########################
# Number of sites
sim_params.N = 100

# List of alpha_0s
alpha_step = 5
max_N = 100 // alpha_step
sim_params.alpha_0_list = [float(alpha_step * i) for i in range(1, max_N)]
sim_params.beta_0_list = [100. - sim_params.alpha_0_list[i] for i in range(max_N - 1)]
sim_params.ws_list = [20.0]
sim_params.wf_list = [30.0]

###################################################################
################## Initialize the data lists ######################
# data lists
value_lists = [[] for _ in range(sim_params.N)]
q_value_lists_0 = [[] for _ in range(sim_params.N)]
q_value_lists_1 = [[] for _ in range(sim_params.N)]
trust_lists = [[] for _ in range(sim_params.N)]
action_lists = [[] for _ in range(sim_params.N)]
one_step_lists_0 = [[] for _ in range(sim_params.N)]
one_step_lists_1 = [[] for _ in range(sim_params.N)]

lists = {}
lists['value'] = value_lists
lists['q_val_0'] = q_value_lists_0
lists['q_val_1'] = q_value_lists_1
lists['trust'] = trust_lists
lists['action'] = action_lists
lists['one_step_0'] = one_step_lists_0
lists['one_step_1'] = one_step_lists_1
######################################################################
######### Initialize the data based on default values ################
sim_params = update_sim_params(sim_params, widgets)
lists = update_data(lists, sim_params)
sorted_arrays_dict = sort_lists(lists)
#######################################################################
################### Initialize the plots ##############################
# Setup data sources
k = widgets['k_inp'].value
val_source = ColumnDataSource(data=dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['value'][k]))
q_0_source = ColumnDataSource(data=dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['q_val_0'][k]))
q_1_source = ColumnDataSource(data=dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['q_val_1'][k]))
one_step_0_source = ColumnDataSource(data=dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['one_step_0'][k]))
one_step_1_source = ColumnDataSource(data=dict(x=sorted_arrays_dict['trust'][k], y=sorted_arrays_dict['one_step_1'][k]))
#######################################################################
############# Setup rules for updating the plots ######################
# Set up plots
plot = figure(height=800, width=800, title='Value at site {:d}'.format(k),
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[0.0,1.0], #y_range=[-16.0,0.0],
            x_axis_label='Trust',
            y_axis_label='Value')

# plot.line('x', 'y', source=val_source, line_width=3, line_alpha=0.6, line_color='blue', legend_label='Value')
plot.line('x', 'y', source=q_0_source, line_width=3, line_alpha=0.6, line_color='orangered', legend_label='q-value for NOT USE')
plot.line('x', 'y', source=q_1_source, line_width=3, line_alpha=0.6, line_color='seagreen', legend_label='q-value for USE')


plot2 = figure(height=800, width=800, title='One step rewards at site {:d}'.format(k),
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[0.0,1.0], #y_range=[-16.0,0.0],
            x_axis_label='Trust',
            y_axis_label='One step reward')

plot2.line('x', 'y', source=one_step_0_source, line_width=3, line_alpha=0.6, line_color='orangered', legend_label='NOT USE')
plot2.line('x', 'y', source=one_step_1_source, line_width=3, line_alpha=0.6, line_color='seagreen', legend_label='USE')

for w in widgets.values():
    w.on_change('value', update_sources)

# Set up layouts and add to document
inputs = column(list(widgets.values()))
curdoc().add_root(row(inputs, plot, plot2, width=1800))

#######################################################################
# Plot stuff here
# for i in range(int(min(sim_params.N, 10))):
#     fig, ax = plt.subplots()
#     # print(i, len(trust_lists[i]))
#     # # Sorting
#     # trust = np.array(trust_lists[i])
#     # val = np.array(value_lists[i])
#     # q_val_0 = np.array(q_value_lists_0[i])
#     # q_val_1 = np.array(q_value_lists_1[i])

#     # idxs = np.argsort(trust)
#     # trust = np.take_along_axis(trust, idxs, axis=0)
#     # val = np.take_along_axis(val, idxs, axis=0)
#     # q_val_0 = np.take_along_axis(q_val_0, idxs, axis=0)
#     # q_val_1 = np.take_along_axis(q_val_1, idxs, axis=0)

#     ax.plot(sorted_arrays_dict['trust'][i], sorted_arrays_dict['value'][i], c='tab:blue', label='value')
#     ax.plot(sorted_arrays_dict['trust'][i], sorted_arrays_dict['q_val_0'][i], c='tab:green', label='q-value 0')
#     ax.plot(sorted_arrays_dict['trust'][i], sorted_arrays_dict['q_val_1'][i], c='tab:orange', label='q-value 1')
#     ax.set_ylabel('Value')
#     ax.set_xlabel('Trust')
#     ax.legend()

# plt.show()

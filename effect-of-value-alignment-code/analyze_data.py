from classes.DataReader import PickleReader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import path
from classes.Utils import col_print


def plot(fig, ax: plt.Axes, mean, std=None, color=None, title=None, x_label=None, y_label=None, plot_label=None):
    ax.plot(mean, label=plot_label, c=color, linewidth=2)

    if std is not None:
        ax.fill_between(np.arange(len(mean)), mean + std, mean - std, color=color, alpha=0.7)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True)

    return fig, ax


def plot_posterior(data: dict, fill: bool = False):
    fig, ax = plt.subplots()

    posterior_dists = data['posterior']
    weights = data['weights'][0, 0, 0, :]

    num_missions = posterior_dists.shape[1]
    num_sites = posterior_dists.shape[2]
    colors = list(mcolors.TABLEAU_COLORS)

    if num_missions > 1:

        # mean and std have shape (num_missions, num_weights)
        mean = np.mean(posterior_dists, axis=0)[:, -1, :]
        std = np.std(posterior_dists, axis=0)[:, -1, :]

        for i in range(num_missions):
            ax.plot(weights, mean[i, :], linewidth=2, label='Mission #{:d}'.format(i + 1), c=colors[i])
            if fill:
                ax.fill_between(weights, mean[i, :] + std[i, :], mean[i, :] - std[i, :], alpha=0.5, color=colors[i])

    else:
        # mean and std have shape (num_sites, num_weights)
        mean = np.mean(posterior_dists, axis=0)[0, :, :]
        std = np.std(posterior_dists, axis=0)[0, :, :]

        for i in range(1, num_sites):
            if i % 10 == 0:
                ax.plot(weights, mean[i, :], linewidth=2, label='Site #{:d}'.format(i), c=colors[i // 10 - 1])
                if fill:
                    ax.fill_between(weights, mean[i, :] + std[i, :], mean[i, :] - std[i, :], alpha=0.5, color=colors[i])

    ax.legend()
    ax.set_title("Posterior Distribution after each mission", fontsize=16)
    ax.set_xlabel(r'Health Weight $w_h$', fontsize=14)
    ax.set_ylabel(r'$P(w_h)$', fontsize=14)
    ax.grid(True)

    return fig


def plot_trust_after_each_site(data: dict):
    fig, ax = plt.subplots()

    trust_feedback = data['trust feedback']
    mean = np.mean(trust_feedback, axis=0)
    std = np.std(trust_feedback, axis=0)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:blue', plot_label='feedback',
                   title="Trust after each site", x_label='Site index', y_label='Trust')

    trust_estimate = data['trust estimate']
    mean = np.mean(trust_estimate, axis=0)
    std = np.std(trust_estimate, axis=0)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:orange', plot_label='estimate',
                   title="Trust after each site", x_label='Site index', y_label='Trust')

    ax.legend()
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    return fig


def plot_trust_after_each_mission(data: dict):
    fig, ax = plt.subplots()

    trust_feedback = data['trust feedback']
    mean = np.mean(trust_feedback, axis=0)
    std = np.std(trust_feedback, axis=0)

    fig, ax = plot(fig, ax, mean[:, -1], std[:, -1], 'tab:blue', plot_label='feedback', title="Trust at end of mission",
                   x_label='Mission index', y_label='Trust')

    trust_estimate = data['trust estimate']
    mean = np.mean(trust_estimate, axis=0)
    std = np.std(trust_estimate, axis=0)

    fig, ax = plot(fig, ax, mean[:, -1], std[:, -1], 'tab:orange', plot_label='estimate',
                   title="Trust at end of mission", x_label='Mission index', y_label='Trust')

    ax.legend()
    ax.set_ylim([-0.05, 1.05])

    return fig


def plot_mean_weight_after_each_site(data: dict):
    fig, ax = plt.subplots()

    wh_means = data['mean health weight']  # Shape (num_simulations, num_missions, N+1)
    mean = np.mean(wh_means, axis=0)  # Shape (num_missions, N+1)
    std = np.std(wh_means, axis=0)  # Shape (num_missions, N+1)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:blue', plot_label='mean weight',
                   title="Estimated health weight after each site", x_label='Site index', y_label='Mean health weight')

    ax.set_ylim([-0.05, 1.05])

    return fig


def plot_mean_weight_after_each_mission(data: dict):
    fig, ax = plt.subplots()

    wh_means = data['mean health weight']  # Shape (num_simulations, num_missions, N+1)
    mean = np.mean(wh_means, axis=0)  # Shape (num_missions, N+1)
    std = np.std(wh_means, axis=0)  # Shape (num_missions, N+1)

    fig, ax = plot(fig, ax, mean[:, -1], std[:, -1], 'tab:blue', plot_label='mean weight',
                   title="Estimated health weight after each mission", x_label='Mission index',
                   y_label='Mean health weight')
    ax.set_ylim([-0.05, 1.05])

    return fig


def plot_health_after_each_site(data: dict):
    fig, ax = plt.subplots()

    health = data['health']  # Shape (num_simulations, num_missions, N+1)
    mean = np.mean(health, axis=0)  # Shape (num_missions, N+1)
    std = np.std(health, axis=0)  # Shape (num_missions, N+1)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:blue', plot_label='health',
                   title="Soldier health after each site", x_label='Site index', y_label='Health')

    return fig


def plot_time_after_each_site(data: dict):
    fig, ax = plt.subplots()

    sim_time = data['time']  # Shape (num_simulations, num_missions, N+1)
    mean = np.mean(sim_time, axis=0)  # Shape (num_missions, N+1)
    std = np.std(sim_time, axis=0)  # Shape (num_missions, N+1)

    fig, ax = plot(fig, ax, mean.flatten(), std.flatten(), 'tab:blue', plot_label='time',
                   title="Time in simulation after each site", x_label='Site index', y_label='Time')

    return fig


def analyze(data: dict, dir: str, save_figs: bool):
    # Most shapes are (num_simulations, num_missions, num_sites or num_sites+1)

    ############################## TRUST At end of mission ##################################################

    # plot_trust_after_each_mission(data)

    ############################ Trust after each site #######################################################
    fig = plot_trust_after_each_site(data)
    if save_figs:
        fig_name = path.join(dir, "trust.png")
        fig.savefig(fig_name)

    ########################################### POSTERIOR ################################################

    fig = plot_posterior(data)
    if save_figs:
        fig_name = path.join(dir, "posterior.png")
        fig.savefig(fig_name)

    ######################################### Mean weight after each site #############################################

    fig = plot_mean_weight_after_each_site(data)
    if save_figs:
        fig_name = path.join(dir, "wh-mean.png")
        fig.savefig(fig_name)

    ########################################### Mean weight after each mission #######################################

    # plot_mean_weight_after_each_mission(data)

    ################################### Health and Time after each site ##############################################

    fig = plot_health_after_each_site(data)
    if save_figs:
        fig_name = path.join(dir, "health.png")
        fig.savefig(fig_name)

    fig = plot_time_after_each_site(data)
    if save_figs:
        fig_name = path.join(dir, "time.png")
        fig.savefig(fig_name)

    # plt.show()

    # data['recommendation']
    # data['actions']
    # data['prior threat level']
    # data['after scan level']
    # data['threat'] 
    # data['map health weight']
    # data['map health weight probability']
    # data['performance estimates']
    # data['performance actual']


def main(args: argparse.Namespace):
    filepath = args.path
    PRINT_TRUST_PARAMS = args.print_trust_params
    PRINT_TRIALS = args.print_trials

    if filepath is None:
        raise "filepath cannot be none"

    direc = path.dirname(filepath)

    reader = PickleReader(filepath)
    reader.read_data()

    save_figs = args.save_figs

    analyze(reader.data, direc, save_figs)

    if PRINT_TRUST_PARAMS:
        i = 0
        parameter_estimates = reader.data["trust parameter estimates"]
        num_simulations, num_missions, _, _ = parameter_estimates.shape

        while i < num_simulations:
            for j in range(num_missions):
                print(parameter_estimates[i, j, -1, :])

            i += 1
            input("Press Enter...")

    if PRINT_TRIALS:
        t_feedback = reader.data['trust feedback']
        t_estimate = reader.data['trust estimate']
        prior = reader.data['prior threat level']
        after_scan = reader.data['after scan level']
        threat = reader.data['threat']
        health = reader.data['health']
        time_ = reader.data['time']
        recs = reader.data['recommendation']
        acts = reader.data['actions']
        p_est = reader.data['performance estimates']
        p_act = reader.data['performance actual']
        mean_wh = reader.data['mean health weight']

        num_simulations, num_missions, num_sites = p_est.shape

        for i in range(num_simulations):
            print()
            print("Simulation #{:d}".format(i))
            print()

            for j in range(num_missions):
                print()
                print("Mission #{:d}".format(j))
                print()

                # Initialize data
                table_data = [['prior', 'after_scan', 'threat', 'rec', 'action', 'health', 'time', 'trust-fb',
                               'trust-est', 'perf_act', 'perf_est', 'wh_mean']]

                for k in range(num_sites):
                    row = []
                    row.append("{:.2f}".format(prior[i, j, k]))
                    row.append("{:.2f}".format(after_scan[i, j, k]))
                    row.append("{:d}".format(threat[i, j, k]))
                    row.append("{:d}".format(recs[i, j, k]))
                    row.append("{:d}".format(acts[i, j, k]))
                    row.append("{:d}".format(health[i, j, k]))
                    row.append("{:d}".format(time_[i, j, k]))
                    row.append("{:.2f}".format(t_feedback[i, j, k]))
                    row.append("{:.2f}".format(t_estimate[i, j, k]))
                    row.append("{:d}".format(p_act[i, j, k]))
                    row.append("{:d}".format(p_est[i, j, k]))
                    row.append("{:.2f}".format(mean_wh[i, j, k]))
                    table_data.append(row)

                col_print(table_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exploring the stored simulated data')

    parser.add_argument('--path', required=True, help='Filepath for the data file. No default')
    parser.add_argument('--print-trust-params',
                        help='Flag to control whether to print the trust parameters (default: False)', default=False)
    parser.add_argument('--save-figs', type=bool,
                        help="Flag to set or unset saving figures (default: False)", default=True)
    parser.add_argument('--print-trials', type=bool,
                        help='Flag to control printing of the details of all trial in all simulations (default: False)',
                        default=False)

    args = parser.parse_args()

    main(args)

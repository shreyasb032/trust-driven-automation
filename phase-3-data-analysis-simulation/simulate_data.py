from typing import List
from classes.Human import HumanBase, SigmoidHuman
from classes.Learner import WeightsLearner
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_theme(context='talk', style='white')


def simulate_learning(human: HumanBase, learner: WeightsLearner,
                      health_list: List[int], time_list: List[int]):

    learned_weights = []
    true_weights = []
    health_used = []
    time_used = []

    for health in health_list:
        for time in time_list:
            state = np.array([[health, time]], dtype=float)
            true_wh = human.get_wh(state)
            if true_wh > 0.5:
                true_weights.append(true_wh)
                health_used.append(health)
                time_used.append(time)
                learned_wh = learner.simulate_and_learn(health, time)
                learned_weights.append(learned_wh)

    return true_weights, learned_weights, health_used, time_used


def scatter_plot(true_weights: List[float], learned_weights: List[float]):

    true_weights = np.array(true_weights)
    index_array = np.argsort(true_weights)

    learned_weights = np.array(learned_weights)
    true_weights_plotting = true_weights[index_array]
    learned_weights_plotting = learned_weights[index_array]

    fig, ax = plt.subplots(layout='tight')
    ax.scatter(np.arange(true_weights.shape[0]), true_weights_plotting, c='red', label='True')
    ax.scatter(np.arange(true_weights.shape[0]), learned_weights_plotting, c='green', label='Learned')
    ax.set_xlabel('Index')
    ax.set_ylabel('Health Weight')
    ax.set_title('Learned health weights')
    ax.legend(loc='lower right')
    ax.grid(True)

    return fig, ax


def plot_imshow(true_weights: np.ndarray, learned_weights: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    ax1.imshow(true_weights, origin='lower', vmin=0.0, vmax=1.0)
    ax1.set_ylabel("Time")
    ax1.set_xlabel('Health')
    ax1.set_title('True Weights')
    im = ax2.imshow(learned_weights, origin='lower', vmin=0.0, vmax=1.0)
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='5%', pad=0.2)
    ax2.set_ylabel("Time")
    ax2.set_xlabel('Health')
    ax2.set_title('Learned Weights')
    fig.colorbar(im, ax=(ax1, ax2), orientation='horizontal', fraction=0.1, location='bottom')

    return fig, ax1, ax2


def main():

    human = SigmoidHuman(kappa=1.0, seed=256, noise=True)
    learner = WeightsLearner(human, num_workers=200)

    health_list = [20*i for i in range(1, 6)]
    health_list.insert(0, 10)
    time_list = health_list.copy()

    true_weights, learned_weights, health_used, time_used = simulate_learning(human, learner, health_list, time_list)

    # for true_wh, learned_wh in zip(true_weights, learned_weights):
    #     print(true_wh, learned_wh)

    fig, ax = scatter_plot(true_weights, learned_weights)

    true_weights_2d = np.ones((len(health_list), len(time_list)), dtype=float) * 0.3
    learned_weights_2d = true_weights_2d.copy()

    for idx, (health, time) in enumerate(zip(health_used, time_used)):
        health_idx = health_list.index(health)
        time_idx = time_list.index(time)
        # true_weights_2d[time_idx, health_idx] = true_weights[idx]
        learned_weights_2d[time_idx, health_idx] = learned_weights[idx]

    for i, health in enumerate(health_list):
        for j, time in enumerate(time_list):
            true_weights_2d[j, i] = human.get_wh(np.array([[health, time]]))

    fig, ax1, ax2 = plot_imshow(true_weights_2d, learned_weights_2d)
    plt.show()


if __name__ == "__main__":
    main()

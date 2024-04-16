import numpy as np
from classes.Human import sigmoid
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='white')


def main():

    weights_list = [[-0.01, 0.015]]

    health_list = [10 * i for i in range(11)]
    time_list = health_list.copy()
    xx, yy = np.meshgrid(health_list, time_list)
    z_list = []

    for weights in weights_list:
        zz = sigmoid(xx * weights[0] + yy * weights[1])
        z_list.append(zz)

    for zz in z_list:
        fig, ax = plt.subplots(layout='tight')
        plt.imshow(zz, origin='lower', extent=(0, 100, 0, 100))
        plt.xlabel('Health')
        plt.ylabel("Time")
        plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()

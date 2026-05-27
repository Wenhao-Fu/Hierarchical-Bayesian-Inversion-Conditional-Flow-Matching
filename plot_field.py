import numpy as np
import h5py
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator


def plot_field(field, path):
    config = {
        "font.family": 'serif',
        "font.size": 10.5,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    # plot generated target image
    fig, ax = plt.subplots(5, 6, figsize=(20, 14))
    ax = ax.flatten()
    a = []
    for i in range(30):
        a.append(ax[i].imshow(field[i].reshape(64, 64), plt.get_cmap('jet'), vmin=0, vmax=1))
        # a.append(ax[i].imshow(field[i].reshape(64, 64), plt.get_cmap('jet'), vmin=-3, vmax=3))
        ax[i].axis()

        ax[i].xaxis.set_major_locator(MultipleLocator(15))
        ax[i].yaxis.set_major_locator(MultipleLocator(15))

        x = np.array([58, 58, 58, 58])
        y = np.array([10, 25, 40, 50])
        ax[i].scatter(x, y, s=20, c='white', marker='^')

        x = np.array([0, 8, 8, 8, 20])
        y = np.array([0, 10, 30, 50, 20])
        ax[i].scatter(x, y, s=20, c='white', marker='o')

        for spine in ax[i].spines.values():
            spine.set_linewidth(1)

        ax[i].tick_params(axis='both', direction='in')
        ax[i].tick_params(axis='both', labelsize=10.5, width=1)

        cbar = fig.colorbar(a[i], ax=ax[i], fraction=0.046, pad=0.046)
        cbar.outline.set_linewidth(1)
        cbar.ax.tick_params(labelsize=10.5, width=1)
        cbar.ax.tick_params(axis='both', direction='in')

        ticks = cbar.ax.get_yticklabels()

        for tick in ticks:
            tick.set_fontname('Times New Roman')

        # plt.show()
        plt.subplots_adjust(wspace=0.25, hspace=0.2)
        plt.yticks(fontproperties='Times New Roman', size=10.5)
        plt.xticks(fontproperties='Times New Roman', size=10.5)
    plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.01)
    plt.close()


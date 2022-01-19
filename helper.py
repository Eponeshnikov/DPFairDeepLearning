import numpy as np
import matplotlib.pyplot as plt

def plot_results(results):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,4))
    x = np.arange(len(results))
    width = .4
    plot_names, plot_values, sdt_values = [], [], []
    for name in results:
        plot_names.append(name)
        metrics, stds = results[name]
        plot_values.append(metrics)
        sdt_values.append(stds)
    
    plot_values = np.array(plot_values)
    sdt_values = np.array(sdt_values)
    #print(plot_values)
    ax[0].bar(x, plot_values[:, 0], width, yerr=sdt_values[:, 0], label='Accuracy')
    ax[1].bar(x-width/3, plot_values[:, 1], width/3,  yerr=sdt_values[:, 1], label='$\Delta_{DP}$')
    ax[1].bar(x, plot_values[:, 2], width/3, label='$\Delta_{EOD}$')
    ax[1].bar(x+width/3, plot_values[:, 3], width/3, label='$\Delta_{EOP}$')

    for i in [0,1]:
        ax[i].set_xticks(x)  
        ax[i].set_xticklabels(plot_names) 
        ax[i].legend()
    fig.tight_layout()
    plt.show()
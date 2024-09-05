from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from data import ENDO_NAMES, OBS_VAR_NAMES

cp = sns.color_palette()

def plot_preds(
        y_gt:np.ndarray, 
        y_hats: Dict[str, np.ndarray], 
        n_timestep_forecast: int = 1
    ) -> plt.Figure:
    fig, axs = plt.subplots(8,1,figsize=(10,15))
    
    axs[0].plot([0,1],[-1,-1],label='GT',color='black')
    for i, method in enumerate(y_hats):
        axs[0].plot([0,1],[-1,-1],label=method,color=cp[i])
    axs[0].legend(ncol=6, loc='center')
    axs[0].set_axis_off()
    axs[0].set_ylim([0,1])

    for i, ax in enumerate(axs[1:]):
        for j, (method, y_hat) in enumerate(y_hats.items()):
            if j == 0:
                ax.plot(y_gt[n_timestep_forecast-1:len(y_hat)+n_timestep_forecast-1,i], label='GT', color='black')
                # ax.plot(y_gt[:len(y_hat),i], label='GT', color='black')
            ax.plot(y_hat[:,n_timestep_forecast-1,i], label=method, color=cp[j])
        # xtl = ax.get_xticklabels()
        xticks = np.arange(0,215,4)
        ax.set_xticks(xticks)
        years = np.arange(1948,2002,1)
        new_xtl = [y if i % 4 == 0 else None for i, y in enumerate(years)]
        ax.set_xticklabels(new_xtl)
        desc = ENDO_NAMES[OBS_VAR_NAMES[i]]
        long_name, tex_name = desc['long_name'], desc['tex']
        ax.set_title(f'{tex_name}: {long_name}')
    fig.tight_layout()
    return fig

def plot_rmses(
    data: pd.DataFrame   
) -> plt.Figure:
    fig,axs = plt.subplots(2,4,figsize=(12,6))
    for i, method in enumerate(data.method.unique()):
        axs[0,0].plot([0,1],[-1,-1],label=method,color=cp[i])
    axs[0,0].legend(loc='center')
    axs[0,0].set_axis_off()
    axs[0,0].set_ylim([0,1])
    for i, (var_name, ax) in enumerate(zip(OBS_VAR_NAMES,axs.flatten()[1:])):
        sns.lineplot(data=data, x='t', y=var_name, hue='method', ax=ax)
        ax.get_legend().remove()
        desc = ENDO_NAMES[OBS_VAR_NAMES[i]]
        long_name, tex_name = desc['long_name'], desc['tex']
        ax.set_title(f'{tex_name}: {long_name}')
        ax.set_xticks(np.arange(10)) # [0,1,2,3])
        ax.set_ylabel('RMSE')
    fig.tight_layout()
    return fig
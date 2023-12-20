import torch
import numpy as np
from typing import List, Dict, Any
from xpag.tools.utils import DataType, datatype_convert
import os
import time
import csv
import matplotlib.pyplot as plt


def plot_latent_codes(env, file_path, goals_fn, nb_goals=50):
    
    import matplotlib.cm as cm
                    
    # plot latent mean and variances

    _, axs = plt.subplots(1,2, figsize=(12, 5))
    env.plot(axs[0])
    
    size = int(env.size_max[0] + 1)
    
    input_obs = goals_fn(nb_samples_per_grid=nb_goals)[:,::-1]
    latent_obs = env.convert_2D_to_embed(input_obs)
    
    split_real = np.split(input_obs, size**2)
    #split_latent = np.split(latent_obs, size**2)
    colors = cm.rainbow(np.linspace(0, 1, size**2))
    
    for x, c in zip(split_real, colors):
        axs[0].scatter(x[:,0], x[:,1], color=c)
        
    sigma = env.embed.sigma.numpy()
    mu = env.embed.mu.numpy()
    split_sig = np.split(sigma, size**2)
    split_mu = np.split(mu, size**2)
    
    import math
    
    for s, m, c in zip(split_sig, split_mu, colors):
        for j in range(s.shape[0]):
            u=m[j][0]     #x-position of the center
            v=m[j][1]    #y-position of the center
            a=3*s[j][0]     #radius on the x-axis
            b=3*s[j][1]    #radius on the y-axis
            
            t = np.linspace(0, 2*math.pi, 100)
            axs[1].plot(u+a*np.cos(t) , v+b*np.sin(t),color=c, alpha=0.8)
    
    plt.show()
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()
    
    
    
    
    
def plot_grid_coverage(env, success, env_step, plot_step, save_dir,tag='grid_eval', writer=None):
    
    size = int(env.size_max[0] + 1)
    success_array = np.array(success).reshape(size**2,-1,1)
    succ = success_array.mean(axis=1)
    
    fig,ax = plt.subplots(figsize=(15, 10))
    
    env.plot(ax)
    plt.imshow(succ.reshape(size,-1).swapaxes(0,1),
               origin='lower',
               extent=(-0.5,size-0.5,-0.5,size-0.5),
               vmin=0,
               vmax=1, 
               cmap='RdBu', 
               alpha=0.5
            )
    #plt.title("Average Success ({}) : {} %".format(method,str(succ.mean())[:4]))
    cbar = plt.colorbar()
    for (v,k),label in np.ndenumerate(succ.reshape(size,-1).swapaxes(0,1)):
        plt.text(k,v,round(label,2),ha='center',va='center')
    plt.show()
    cbar.set_label('proba', rotation=270, labelpad=15)
    os.makedirs(os.path.join(os.path.expanduser(save_dir), "plots", tag), 
                exist_ok=True
                )
    
    grid_eval_path = os.path.join(
            os.path.expanduser(save_dir),
            "plots", tag,
            f"{env_step:12}.png".replace(" ", "0")
    )
    plt.savefig(grid_eval_path, bbox_inches='tight')
    
    if writer is not None:
        writer.add_figure('Eval/coverage', fig, plot_step)
import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings("error", category=FutureWarning)
import sys, subprocess, os
#from packaging import version
#from IPython import get_ipython
import matplotlib
from ipywidgets import interact
from IPython.display import display, Image, clear_output
import jax
import seaborn as sns
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import yaml
# verifying GPU backend for jax:

assert(jax.lib.xla_bridge.get_backend().platform == 'gpu')

# xpag:
import xpag
from xpag.wrappers import gym_vec_env, GoalEnvWrapper, custom_vec_env, ResetDoneWrapper, EmbedVecWrapper

from xpag.tools import train_vae_model, ae_plot_gen, compare_vae_obs

# make setters:
from make_setters import *

# 2D mazes sibrivalry
from custom_envs.sibrivalry.toy_maze import PointMaze2D, PixelObsWrapper
from custom_envs.sibrivalry.ant_maze import AntMazeEnv
from utils import tensor_ant_maze_random

# VAE
sys.path.append("..")
from vae_lab.vae_model import VariationalAutoencoder, VariationalEncoder


@hydra.main(version_base=None, config_path="configs", config_name="pretrain_vae")
def train_vae(cfg : DictConfig) -> None:
    
    save_dir = os.path.join(
        os.path.expanduser('~'), 
        'Git',
        'xpag-tutorials', 
        'pretrained_vae', 
        'num_obs_'+ str(cfg.vae.num_data)+'_beta_'+str(cfg.vae.beta)+ '_' + cfg.dir
        )
    
    GEN_PATH = os.path.join(save_dir, 'vae_gen')
    COMPARE_PATH = os.path.join(save_dir, 'vae_compare')
    DIST_EVO_PATH = os.path.join(save_dir, 'dist_evo')
    STATE_DICT = os.path.join(save_dir, 'state_dict')
    os.makedirs(GEN_PATH, exist_ok=True)
    os.makedirs(COMPARE_PATH, exist_ok=True)
    os.makedirs(DIST_EVO_PATH, exist_ok=True)
    
    writer = SummaryWriter(log_dir=save_dir)
        
    
    
    
    ### Get conf variables ###
    maze_type = cfg.env.maze_type
    max_episode_steps = cfg.env.max_episode_steps
    dist_thresh = cfg.env.dist_thresh_latent
    d = cfg.env.latent_dim_obs
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    
    
    ### Load VAE ###
    
    # Full VAE (encoder/decoder) on GPU for optimization
    vae = VariationalAutoencoder(latent_dims=d, in_channel=3)
    vae.to(device)
    
    # Encoder on CPU for environment interaction
    encoder_obs = VariationalEncoder(latent_dims=d, in_channel=3, device=torch.device("cpu"))
    
    # Transfert state dict (on CPU)
    cpu_encoder_state_dict = {k: v.cpu() for k, v in vae.encoder.state_dict().items()}
    encoder_obs.load_state_dict(cpu_encoder_state_dict)
    
    
    ### Load Env. ###
    env_fn = lambda: ResetDoneWrapper(PointMaze2D(maze_type=maze_type, max_steps=max_episode_steps, dist_threshold=dist_thresh))
    eval_env_fn = lambda: ResetDoneWrapper(PointMaze2D(test=True,maze_type=maze_type, max_steps=max_episode_steps, dist_threshold=dist_thresh))
        
        
    env, eval_env, mult_eval_env, env_info = custom_vec_env(env_fn, eval_env_fn, max_episode_steps, cfg.env.num_envs, cfg.env.num_eval_rollout, embed=encoder_obs)
    mult_goal_fn = eval_env.tensor_point_maze_random
    
    
    env_info['observation_dim'] = d
    env_info['achieved_goal_dim'] = d
    env_info['desired_goal_dim'] = d
    eval_env.plot = eval_env.maze.plot
    
    
    ### Train VAE ###
    
    vae_opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae.lr)#, weight_decay=1e-5)
    
    if cfg.from_dataset:
        # TODO : remove hardcode path
        data_dir = os.path.join(os.path.expanduser('~'),  
                    'Git',
                    'xpag-tutorials',
                    'results',
                    'trajectories',
                    'dataset',
                    'PBCS_0',
                    'square_pbcs_0_expert_traj_2.npy',
                      )
        
        desired_goals = np.load(data_dir)
 
    else:
        # Sample uniform from env : Oracle ONLY FOR MAZES
        num_goals = cfg.vae.num_data
        env_size = env_info["maze_size"]
        num_grid = int((env_size[0]+1)**2)
        num_goals_per_grid = num_goals//num_grid
        desired_goals = mult_goal_fn(num_goals_per_grid)
        #desired_goals = tensor_ant_maze_random(num_goals_per_grid)
    
    
    pixel_obs = np.swapaxes(env.convert_2D_to_pixel(desired_goals),3,1)
    
    obs_batch = torch.from_numpy(pixel_obs)
    obs_batch = obs_batch.type(torch.float)
    train_dataset = data.TensorDataset(obs_batch)
    train_dataloader = data.DataLoader(train_dataset,batch_size=len(obs_batch))
    
    losses = train_vae_model(vae, vae_opt, train_dataloader, nb_steps=cfg.vae.num_epoch, beta=cfg.vae.beta)
    
    cpu_encoder_state_dict = {k: v.cpu() for k, v in vae.encoder.state_dict().items()}
    for e in [env, eval_env, mult_eval_env]:
        e.embed.load_state_dict(cpu_encoder_state_dict)
        
    state_dict = {
        'encoder': vae.encoder.state_dict(),
        'decoder': vae.decoder.state_dict()
    }
    torch.save(state_dict, STATE_DICT)
        
    
    ae_plot_gen(step=0,
                    plot_step=0,
                    vae=vae,
                    path=GEN_PATH,
                    writer=writer    
                )
    
    real_obs_from_latent = compare_vae_obs(step=0,
                                                plot_step=0,
                                                real_obs=desired_goals,
                                                pixel_obs=obs_batch,
                                                encoder=env.embed,
                                                decoder=vae.decoder,
                                                path=COMPARE_PATH,
                                                writer=writer,
                                                plot_images_similarity=True
                                            )
    
    noise = np.random.randn(real_obs_from_latent.shape)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_style("white")
    sns.kdeplot(x=real_obs_from_latent[:,0], y=real_obs_from_latent[:,1],shade=True, thresh=0.1, cbar=True)
    eval_env.plot(ax)
    ax.scatter(real_obs_from_latent[:,0], real_obs_from_latent[:,1], c='r')
    #ax.scatter(desired_goals[:,0], desired_goals[:,1], c='g')
    plt.show()
    plt.savefig(DIST_EVO_PATH+f'/step_0', bbox_inches='tight')
    writer.add_figure('Eval/latent_goal_sampling', fig, 0)
    
    plt.clf()
    
    
    import matplotlib.cm as cm
                
    fig, axs = plt.subplots(1,2, figsize=(12, 5))
    eval_env.plot(axs[0])
    
    nb_goals = 50
    size = int(eval_env.size_max[0] + 1)
    
    #input_obs = mult_goal_fn(nb_samples_per_grid=nb_goals)
    input_obs = desired_goals
    latent_obs = eval_env.convert_2D_to_embed(input_obs)
    
    split_size = 10
    
    split_real = np.split(input_obs, split_size)
    split_latent = np.split(latent_obs, split_size)
    colors = cm.rainbow(np.linspace(0, 1, split_size))
    
    for x, c in zip(split_real, colors):
        axs[0].scatter(x[:,0], x[:,1], color=c)
    
    for x, c in zip(split_latent, colors):
        axs[1].scatter(x[:,0], x[:,1], color=c)
        
    #sns.set_style("white")
    #sns.kdeplot(x=latent_obs[:,0], y=latent_obs[:,1],shade=True, thresh=0.1)
                    
        
    plt.show()
    plt.savefig(DIST_EVO_PATH+f'/real_vs_latent', bbox_inches='tight')
    plt.clf()
    
    sigma = eval_env.embed.sigma.numpy()
    mu = eval_env.embed.mu.numpy()
    split_sig = np.split(sigma, split_size)
    split_mu = np.split(mu, split_size)
    import math
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for s, m, c in zip(split_sig, split_mu, colors):
        for i in range(s.shape[0]):
            u=m[i][0]     #x-position of the center
            v=m[i][1]    #y-position of the center
            a=3*s[i][0]     #radius on the x-axis
            b=3*s[i][1]    #radius on the y-axis
            
            t = np.linspace(0, 2*math.pi, 100)
            plt.plot(u+a*np.cos(t) , v+b*np.sin(t),color=c)
            plt.show()
        plt.savefig(DIST_EVO_PATH+f'/sigma_latent', bbox_inches='tight')


if __name__ == "__main__":
    train_vae()

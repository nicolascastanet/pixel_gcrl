import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings("error", category=FutureWarning)
import sys, subprocess, os

from IPython.display import display, Image, clear_output
import jax

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import yaml
# verifying GPU backend for jax:

assert(jax.lib.xla_bridge.get_backend().platform == 'gpu')

# xpag:
import xpag
from xpag.agents import TQC, SAC, TD3
from xpag.wrappers import custom_vec_env, ResetDoneWrapper
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.tools import learn

# make setters:
from make_setters import *

# 2D mazes sibrivalry
from custom_envs.sibrivalry.toy_maze import PointMaze2D

# VAE
sys.path.append("..")
from vae.vae_model import VariationalAutoencoder, VariationalEncoder

# remove warnings from tensorflow_probability, a library used by the TQC and SAC agents in xpag
# ("WARNING:root:The use of `check_types` is deprecated and does not have any effect.)
    


# specify a config file with hydra library
@hydra.main(version_base=None, config_path="configs", config_name="base_config_sib_point_maze")
def train_agent(cfg : DictConfig) -> None:
    import logging
    logger = logging.getLogger()

    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()
    logger.addFilter(CheckTypesFilter())

    maze_type = cfg.env.maze_type
    max_episode_steps = cfg.env.max_episode_steps
    dist_thresh = cfg.env.dist_thresh_latent
    reward_type = cfg.env.reward
    action_scale = np.array([0.95,0.95])
    
    if cfg.env.from_pixel:
        
        d = cfg.env.latent_dim_obs
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {device}')
        
        # Full VAE (encoder/decoder) on GPU for optimization
        vae = VariationalAutoencoder(latent_dims=d, in_channel=3)
        vae.to(device)
        
        # Encoder on CPU for environment interaction
        encoder_obs = VariationalEncoder(latent_dims=d, in_channel=3, device=torch.device("cpu"))
        
        if cfg.vae.pretrain:
            path = os.path.join(
                    os.path.expanduser('~'), 
                    'Git',
                    'pixel_gcrl', 
                    'pretrained_vae', 
                    'num_obs_500_beta_2',
                    'state_dict')
            state = torch.load(path)
            vae.encoder.load_state_dict(state['encoder'])
            vae.decoder.load_state_dict(state['decoder'])
        
        # Transfert state dict (on CPU)
        cpu_encoder_state_dict = {k: v.cpu() for k, v in vae.encoder.state_dict().items()}
        encoder_obs.load_state_dict(cpu_encoder_state_dict)
        
        embed = vae

    else:
        embed = None
        encoder_obs = None

    #dummy_env = PointMaze2D(maze_type=maze_type, max_steps=max_episode_steps)
    env_fn = lambda: ResetDoneWrapper(PointMaze2D(maze_type=maze_type, max_steps=max_episode_steps, dist_threshold=dist_thresh, action_scale=action_scale))
    eval_env_fn = lambda: ResetDoneWrapper(PointMaze2D(test=True,maze_type=maze_type, max_steps=max_episode_steps, dist_threshold=dist_thresh, action_scale=action_scale))
        
        
    env, eval_env, mult_eval_env, env_info = custom_vec_env(env_fn, 
                                                            eval_env_fn, 
                                                            max_episode_steps, 
                                                            cfg.env.num_envs, 
                                                            cfg.env.num_eval_rollout, 
                                                            embed=encoder_obs, 
                                                            reward=reward_type,
                                                            cnn_policy=cfg.env.cnn_policy
                                                        )
    mult_goal_fn = eval_env.tensor_point_maze_random
    
    if cfg.env.from_pixel:
        env_info['observation_dim'] = d
        env_info['achieved_goal_dim'] = d
        env_info['desired_goal_dim'] = d
        if d==2:
            def plot_projection(x):
                return x
        else:
            plot_projection = None

        eval_env.plot = eval_env.maze.plot
    
    else:
        eval_env.plot = eval_env.maze.plot
        def plot_projection(x):
            return x

    
    agent = eval(cfg.rl_algo)(
        env_info['observation_dim'] if not env_info['is_goalenv']
        else env_info['observation_dim'] + env_info['desired_goal_dim'],
        env_info['action_dim'],
        {
            "hidden_dims_actor": tuple(cfg.actor.actor_layers),
            "hidden_dims_critic": tuple(cfg.actor.critic_layers),
            "actor_lr": cfg.actor.actor_lr,
            "critic_lr": cfg.actor.critic_lr,
            "tau": cfg.actor.tau,
            "seed": cfg.seed
        }
    )
    
    sampler_need_encoding = True if (cfg.env.from_pixel and not cfg.vae.pretrain) else False
    sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER(eval_env,
                                                                              sampler_need_encoding)
    buffer = DefaultEpisodicBuffer(
        max_episode_steps=env_info['max_episode_steps'],
        buffer_size=cfg.actor.buffer_size,
        sampler=sampler
    )

    save_dir = os.path.join(
        os.path.expanduser('~'), 'Git','pixel_gcrl','results', cfg.method.name)

    #setter = UniformSetter()
    setter = eval(cfg.method.make_setter)(buffer,
                                env,
                                eval_env,
                                env_info,
                                save_dir,
                                cfg
                                )
    

    start_training_after_x_steps = env_info['max_episode_steps'] * 50
    
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

    os.makedirs(os.path.join(save_dir), exist_ok=True)
    filename = os.path.join(
                os.path.expanduser(save_dir),
                "setter_conf.yaml"
            )
    OmegaConf.save(cfg, filename)

    learn(
        env,
        eval_env,
        env_info,
        agent,
        buffer,
        setter,
        batch_size=cfg.actor.batch_size,
        gd_steps_per_step=cfg.actor.gd_steps_per_step,
        start_training_after_x_steps=start_training_after_x_steps,
        max_steps=cfg.actor.max_steps,
        evaluate_every_x_steps=cfg.actor.evaluate_every_x_steps,
        save_agent_every_x_steps=cfg.actor.save_agent_every_x_steps,
        save_dir=save_dir,
        save_episode=cfg.save_episode,
        plot_projection=plot_projection,
        custom_eval_function=None,
        additional_step_keys=None,
        seed=cfg.seed,
        mult_eval_env=mult_eval_env,
        force_eval_goal_fn=mult_goal_fn,
        plot_goals=cfg.plot_goals,
        vae_obs=embed,
        conf=cfg,
    )


if __name__ == "__main__":
    train_agent()






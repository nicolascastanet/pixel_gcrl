import torch
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal

import math
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KernelDensity
# xpag:
import xpag
from xpag.buffers import Buffer
from xpag.setters import RigSetter, DefaultSetter, UniformSetter, SvggMCMCSetter, SvggBufferSetter, SvggSetter, DensitySetter, RandomSetter, GoalGanSetter
from xpag.tools import MLP, OCSVM, MultivariateGeneralizedGaussian, OCSVM_0
from xpag.svgg import SVGD, RBF, AlphaBetaDifficulty, MinDensity, GoalGanTrainer, LowDensity


def make_svgg(
    buffer : Buffer,
    env,
    eval_env,
    env_info,
    save_dir,
    cfg
):
    
    assert  torch.cuda.is_available()
    device = torch.device("cuda")

    #  Criterion -> Sucess model / Prior
    success_model = MLP(input_size= 2*env_info['desired_goal_dim'], 
                    layers = cfg.method.sp_model.layers
                ).to(device)
          
    optim_model = torch.optim.Adam(success_model.parameters(),
                                   weight_decay=cfg.method.sp_model.weight_decay
                                    )

    if cfg.method.prior.type == 'ocsvm':
        prior = OCSVM(kernel=RBF(gamma=cfg.method.prior.gamma),
                  sk_model=OneClassSVM(nu=cfg.method.prior.ood_percent,
                                       gamma=cfg.method.prior.gamma
                                    )
                )
        
        prior_need_opti = True
        prior_ready = False
        
    elif cfg.method.prior.type == 'normal':
        
        mean = torch.zeros(env_info['desired_goal_dim']).to(torch.device("cuda"))
        cov = torch.eye(env_info['desired_goal_dim']).to(torch.device("cuda"))
        prior = MultivariateNormal(mean, cov)
        
        prior_need_opti = False
        prior_ready = True

    def part_to_goals(part):
        return part.cpu().numpy()

    # Init particles distribution
    if env_info["name"] == "FetchReach-v3":
        init_grip_pos = eval_env.initial_gripper_xpos[:3]
        init_state = torch.tensor(init_grip_pos).type(torch.FloatTensor)
        init_std = cfg.method.particles.init_std
        dist_init = Normal(init_state, torch.ones(env_info["desired_goal_dim"])*init_std)

        dist_init_criterion = None
        terminated = False
        goal_on_fetch_table = False

    elif env_info["name"] == "FetchPush-v2":
        init_grip_pos = eval_env.initial_gripper_xpos[:2]#.repeat(cfg.method.particles.num,1)
        #object_xpos += np.random.uniform(
        #            -0.15, 0.15, size=(cfg.method.particles.num, 2)
        #        )
        #init_state = torch.tensor([init_grip_pos[0], init_grip_pos[1]]#, 0.42469975]
        #                        ).type(torch.FloatTensor) # middle right of the table
        
        init_state = torch.tensor([1.15, init_grip_pos[1]]#, 0.42469975]
                                ).type(torch.FloatTensor) # middle right of the table

        #init_tensor = torch.tensor([init_grip_pos[0], 0.8]).type(torch.FloatTensor)
        dist_init = Uniform(init_state-torch.tensor([0.02,0.2]), init_state+torch.tensor([0.02,0.2]))
        dist_init_criterion = dist_init
        terminated = False
        goal_on_fetch_table = True

        def part_to_goals(part):
            table_z_pos_fetch = np.full((part.shape[0],1),0.42469975)# Table height
            return np.concatenate((part.cpu().numpy(), table_z_pos_fetch),axis=1)
        
    elif env_info["name"] == "FetchPickAndPlace-v2":
        pass
 
    else:
        # Gym Gmazes
        init_state = torch.tensor([0.0, 0.0]).type(torch.FloatTensor)
        dist_init = Uniform(torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5])) # TODO remove hard coded shape
        dist_init_criterion = None
        terminated = True
        goal_on_fetch_table = False
        
    if cfg.method.mode == "difficulty":
        criterion = AlphaBetaDifficulty(alpha = cfg.method.criterion.alpha,
                                    beta = cfg.method.criterion.beta,
                                    model = success_model,
                                    prior = prior,
                                    init_state = init_state,
                                    dist_init = dist_init_criterion,
                                    temp = cfg.method.criterion.temp
                                )
        
        criterion.difficulty_mode = cfg.method.difficulty_mode
        
        if env_info["from_pixels"]:
            criterion.convert_to_embed = eval_env.convert_2D_to_embed
        
        
    elif cfg.method.mode == "low_density":
        criterion = LowDensity(alpha = cfg.method.criterion.alpha,
                                    beta = cfg.method.criterion.beta,
                                    prior = prior,
                                    temp = cfg.method.criterion.temp
                                )
        
    criterion.table_fetch = goal_on_fetch_table
    # Particles and SVGD
    
    init_part = 2 * dist_init.sample((cfg.method.particles.num,)).to(torch.device("cuda"))
    
    if env_info["from_pixels"]:
        init_part = torch.from_numpy(
                eval_env.convert_2D_to_embed(init_part.cpu().numpy())
            ).type(torch.FloatTensor).to(torch.device("cuda"))
    
    #nit_part = prior.sample((cfg.method.particles.num,)).to(torch.device("cuda"))    
    particles = init_part.clone()
    kernel = RBF(sigma=None)
    optim_part = torch.optim.Adam([particles], lr=cfg.method.particles.lr)

    svgd = SVGD(P = criterion, 
                K = kernel, 
                optimizer = optim_part,
                epoch = cfg.method.particles.num_steps_per_opt
            )

    # SVGG Setter
    setter = SvggSetter(cfg.env.num_envs,
                        eval_env,
                        buffer,
                        particles,
                        svgd,
                        criterion,
                        success_model,
                        optim_model,
                        prior,
                        cfg.method.plot,
                        part_to_goals,
                        save_dir
                    )

    setter.particles_oe = cfg.method.particles.optimize_every_x_steps // cfg.env.num_envs
    setter.prior_oe = cfg.method.prior.optimize_every_x_steps // cfg.env.num_envs
    setter.prior_batch_size = cfg.method.prior.batch_size
    setter.model_oe = cfg.method.sp_model.optimize_every_x_steps // cfg.env.num_envs
    setter.model_bs = cfg.method.sp_model.batch_size
    setter.model_hl = cfg.method.sp_model.history_length
    setter.model_k_steps = cfg.method.sp_model.k_steps_optimization

    setter.prior_need_opti = prior_need_opti
    setter.prior_ready = prior_ready
    setter.terminated = terminated
    setter.table_fetch = goal_on_fetch_table

    if not cfg.method.particles.annealed:
        setter.annealed_freq = math.inf
    else:
        setter.annealed_freq = cfg.method.particles.annealed_freq
        
    if cfg.method.mode == "low_density":
        setter.model_oe = math.inf
        setter.model_ready = True
        

    return setter

def make_mcmc_svgg(
    buffer : Buffer,
    env,
    eval_env,
    env_info,
    save_dir,
    cfg
):
    
    assert  torch.cuda.is_available()
    device = torch.device("cuda")

    #  Criterion -> Sucess model / Prior
    success_model = MLP(input_size= 2*env_info['desired_goal_dim'], 
                    layers = cfg.method.sp_model.layers
                ).to(device)
          
    optim_model = torch.optim.Adam(success_model.parameters(),
                                   weight_decay=cfg.method.sp_model.weight_decay
                                    )

    prior = OCSVM(kernel=RBF(gamma=cfg.method.prior.gamma),
                  sk_model=OneClassSVM(nu=cfg.method.prior.ood_percent,
                                       gamma=cfg.method.prior.gamma
                                    )
                )

    def part_to_goals(part):
        return part.cpu().numpy()

    # Init particles distribution
    if env_info["name"] == "FetchReach-v3":
        init_grip_pos = eval_env.initial_gripper_xpos[:3]
        init_state = torch.tensor(init_grip_pos).type(torch.FloatTensor)
        init_std = cfg.method.particles.init_std
        dist_init = Normal(init_state, torch.ones(env_info["desired_goal_dim"])*init_std)

        dist_init_criterion = None
        terminated = False
        goal_on_fetch_table = False

    elif env_info["name"] == "FetchPush-v2":
        init_grip_pos = eval_env.initial_gripper_xpos[:2]#.repeat(cfg.method.particles.num,1)
        #object_xpos += np.random.uniform(
        #            -0.15, 0.15, size=(cfg.method.particles.num, 2)
        #        )
        #init_state = torch.tensor([init_grip_pos[0], init_grip_pos[1]]#, 0.42469975]
        #                        ).type(torch.FloatTensor) # middle right of the table
        
        init_state = torch.tensor([1.15, init_grip_pos[1]]#, 0.42469975]
                                ).type(torch.FloatTensor) # middle right of the table

        #init_tensor = torch.tensor([init_grip_pos[0], 0.8]).type(torch.FloatTensor)
        dist_init = Uniform(init_state-torch.tensor([0.02,0.2]), init_state+torch.tensor([0.02,0.2]))
        dist_init_criterion = dist_init
        terminated = False
        goal_on_fetch_table = True

        def part_to_goals(part):
            table_z_pos_fetch = np.full((part.shape[0],1),0.42469975)# Table height
            return np.concatenate((part.cpu().numpy(), table_z_pos_fetch),axis=1)
        
    elif env_info["name"] == "FetchPickAndPlace-v2":
        pass


    else:
        # Gym Gmazes
        init_state = torch.tensor([0.0, 0.0]).type(torch.FloatTensor)
        dist_init = Uniform(torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5])) # TODO remove hard coded shape
        dist_init_criterion = None
        terminated = True
        goal_on_fetch_table = False

    criterion = AlphaBetaDifficulty(alpha = cfg.method.criterion.alpha,
                                    beta = cfg.method.criterion.beta,
                                    model = success_model,
                                    prior = prior,
                                    init_state = init_state,
                                    dist_init = dist_init_criterion,
                                    temp = cfg.method.criterion.temp
                                )
    criterion.table_fetch = goal_on_fetch_table
    
    # SVGG MCMC Setter
    setter = SvggMCMCSetter(cfg.env.num_envs,
                        eval_env,
                        buffer,
                        criterion,
                        success_model,
                        optim_model,
                        prior,
                        cfg.method.plot,
                        part_to_goals,
                        save_dir
                    )

    setter.prior_oe = cfg.method.prior.optimize_every_x_steps // cfg.env.num_envs
    setter.prior_batch_size = cfg.method.prior.batch_size
    setter.model_oe = cfg.method.sp_model.optimize_every_x_steps // cfg.env.num_envs
    setter.model_bs = cfg.method.sp_model.batch_size
    setter.model_hl = cfg.method.sp_model.history_length
    setter.model_k_steps = cfg.method.sp_model.k_steps_optimization

    setter.terminated = terminated
    setter.table_fetch = goal_on_fetch_table    

    return setter


def make_svgg_buffer(
    buffer : Buffer,
    env,
    eval_env,
    env_info,
    save_dir,
    cfg
):
    
    assert  torch.cuda.is_available()
    device = torch.device("cuda")

    #  Criterion -> Sucess model / Prior
    success_model = MLP(input_size= 2*env_info['desired_goal_dim'], 
                    layers = cfg.method.sp_model.layers
                ).to(device)
          
    optim_model = torch.optim.Adam(success_model.parameters(),
                                   weight_decay=cfg.method.sp_model.weight_decay
                                    )
    
    # Init states distribution
    if env_info["name"] == "FetchReach-v3":
        init_grip_pos = eval_env.initial_gripper_xpos[:3]
        init_state = torch.tensor(init_grip_pos).type(torch.FloatTensor)
        init_std = cfg.method.particles.init_std
        dist_init = Normal(init_state, torch.ones(env_info["desired_goal_dim"])*init_std)

        dist_init_criterion = None
        terminated = False
        goal_on_fetch_table = False

    elif env_info["name"] == "FetchPush-v2":
        init_grip_pos = eval_env.initial_gripper_xpos[:2]#.repeat(cfg.method.particles.num,1)
        #object_xpos += np.random.uniform(
        #            -0.15, 0.15, size=(cfg.method.particles.num, 2)
        #        )
        #init_state = torch.tensor([init_grip_pos[0], init_grip_pos[1]]#, 0.42469975]
        #                        ).type(torch.FloatTensor) # middle right of the table
        
        init_state = torch.tensor([1.15, init_grip_pos[1]]#, 0.42469975]
                                ).type(torch.FloatTensor) # middle right of the table

        #init_tensor = torch.tensor([init_grip_pos[0], 0.8]).type(torch.FloatTensor)
        dist_init = Uniform(init_state-torch.tensor([0.02,0.2]), init_state+torch.tensor([0.02,0.2]))
        dist_init_criterion = dist_init
        terminated = False
        goal_on_fetch_table = True

        def part_to_goals(part):
            table_z_pos_fetch = np.full((part.shape[0],1),0.42469975)# Table height
            return np.concatenate((part.cpu().numpy(), table_z_pos_fetch),axis=1)
        
    elif env_info["name"] == "FetchPickAndPlace-v2":
        pass
        
        #init_grip_pos = eval_env.initial_gripper_xpos[:2]#.repeat(cfg.method.particles.num,1)
        ##object_xpos += np.random.uniform(
        ##            -0.15, 0.15, size=(cfg.method.particles.num, 2)
        ##        )
        #init_state = torch.tensor([init_grip_pos[0], init_grip_pos[1]]#, 0.42469975]
        #                        ).type(torch.FloatTensor) # middle right of the table
#
        ##init_tensor = torch.tensor([init_grip_pos[0], 0.8]).type(torch.FloatTensor)
        #dist_init = Uniform(init_state-0.15, init_state+0.15)
        #dist_init_criterion = dist_init
        #terminated = False
        #goal_on_fetch_table = False

    else:
        # Gym Gmazes
        init_state = torch.tensor([0.0, 0.0]).type(torch.FloatTensor)
        dist_init = Uniform(torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5])) # TODO remove hard coded shape
        dist_init_criterion = None
        terminated = True
        goal_on_fetch_table = False

    criterion = AlphaBetaDifficulty(alpha = cfg.method.criterion.alpha,
                                    beta = cfg.method.criterion.beta,
                                    model = success_model,
                                    prior = None,
                                    init_state = init_state,
                                    dist_init = dist_init_criterion,
                                    temp = cfg.method.criterion.temp
                                )
    criterion.table_fetch = goal_on_fetch_table
    
    # SVGG Setter
    setter = SvggBufferSetter(cfg.env.num_envs,
                        eval_env,
                        buffer,
                        criterion,
                        success_model,
                        optim_model,
                        cfg.method.plot,
                        save_dir
                    )

    setter.model_oe = cfg.method.sp_model.optimize_every_x_steps // cfg.env.num_envs
    setter.model_bs = cfg.method.sp_model.batch_size
    setter.model_hl = cfg.method.sp_model.history_length
    setter.model_k_steps = cfg.method.sp_model.k_steps_optimization

    setter.terminated = terminated
    setter.table_fetch = goal_on_fetch_table

    return setter


def make_goal_gan(
    buffer : Buffer,
    env,
    eval_env,
    env_info,
    save_dir,
    cfg
):
    
    assert  torch.cuda.is_available()
    device = torch.device("cuda")
    
    
    def part_to_goals(part):
        return part.cpu().numpy()
    goal_on_fetch_table = False
    if env_info["name"] == "FetchReach-v3":
        init_grip_pos = eval_env.initial_gripper_xpos[:3]
        init_state = torch.tensor(init_grip_pos).type(torch.FloatTensor)
        
        dist_init_gg = None
        terminated = False
        goal_on_fetch_table = False
        gan_input_dim = env_info['desired_goal_dim']

    elif env_info["name"] == "FetchPush-v2":
        
        init_grip_pos = eval_env.initial_gripper_xpos[:2]
        init_state = torch.tensor([1.15, init_grip_pos[1]]
                                ).type(torch.FloatTensor)

        dist_init = Uniform(init_state-torch.tensor([0.02,0.2]), init_state+torch.tensor([0.02,0.2]))
        dist_init_gg = dist_init
        terminated = False
        goal_on_fetch_table = True
        gan_input_dim = dist_init.batch_shape[0]
        
        def part_to_goals(part):
            table_z_pos_fetch = np.full((part.shape[0],1),0.42469975)
            return np.concatenate((part.cpu().numpy(), table_z_pos_fetch),axis=1)
        
    else:
        init_state = torch.tensor([0.0, 0.0]).type(torch.FloatTensor)
        dist_init_gg = Uniform(torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5])) # TODO remove hard coded shape
        terminated = True
        goal_on_fetch_table = False
        gan_input_dim = env_info['desired_goal_dim']
        
        
    #  Criterion -> Sucess model / Prior
    success_model = MLP(input_size= 2*env_info['desired_goal_dim'], 
                    layers = cfg.method.sp_model.layers
                ).to(device)
          
    optim_model = torch.optim.Adam(success_model.parameters(),
                                   weight_decay=cfg.method.sp_model.weight_decay
                                    )

    gan_discriminator = MLP(input_size= 2*gan_input_dim,
                            output_size=1,
                            layers = cfg.method.gan.disc_layers
                ).to(device)

    gan_generator = MLP(input_size= cfg.method.gan.noise_dim,
                        output_size=gan_input_dim,
                        layers = cfg.method.gan.gen_layers
                ).to(device)


    
        
        
    gan_trainer = GoalGanTrainer(
                        gan_discriminator,
                        gan_generator,
                        init_state,
                        dist_init_gg,
                        cfg.method.gan.batch_size,
                        cfg.method.gan.history_length,
                        cfg.method.gan.optimize_every_x_steps,
                        cfg.method.gan.k_steps,
                        cfg.method.gan.noise_dim,
                        goal_on_fetch_table
                    )
    
    gan_trainer.p_min = cfg.method.gan.p_min
    gan_trainer.p_max = cfg.method.gan.p_max
    
    
    setter = GoalGanSetter(
                        cfg.env.num_envs,
                        eval_env,
                        buffer,
                        success_model,
                        optim_model,
                        gan_trainer,
                        gan_generator,
                        gan_discriminator,
                        cfg.method.plot,
                        part_to_goals,
                    )
    
    if cfg.method.gan.prior:
        setter.prior = OCSVM(kernel=RBF(gamma=cfg.method.prior.gamma),
                  sk_model=OneClassSVM(nu=cfg.method.prior.ood_percent,
                                       gamma=cfg.method.prior.gamma
                                    )
                )
        
    
    setter.noise_dim = cfg.method.gan.noise_dim
    setter.gan_oe = cfg.method.gan.optimize_every_x_steps // cfg.env.num_envs
    setter.model_oe = cfg.method.sp_model.optimize_every_x_steps // cfg.env.num_envs
    setter.model_bs = cfg.method.sp_model.batch_size
    setter.model_hl = cfg.method.sp_model.history_length
    setter.model_k_steps = cfg.method.sp_model.k_steps_optimization

    setter.save_dir = save_dir
    setter.terminated = terminated

    return setter
    

    





def make_mega(
    buffer : Buffer,
    env,
    eval_env,
    env_info,
    save_dir,
    cfg
):
    density = KernelDensity(
                    kernel=cfg.method.criterion.kernel, 
                    bandwidth=cfg.method.criterion.bandwidth
                    )
    setter = DensitySetter(
                        cfg.env.num_envs,
                        eval_env,
                        buffer,
                        density
    )

    setter.density_oe = cfg.method.criterion.optimize_every_x_steps // cfg.env.num_envs
    setter.num_ag_candidate = cfg.method.criterion.num_ag_candidate
    setter.density_batch_size = cfg.method.criterion.batch_size
    setter.save_dir = save_dir
    setter.randomize = cfg.method.criterion.randomize
    setter.alpha = cfg.method.criterion.alpha_skew

    return setter


def make_random(
        buffer : Buffer,
        env,
        eval_env,
        env_info,
        save_dir,
        cfg):
    
    return RandomSetter(buffer, cfg.env.num_envs, eval_env)


def make_uniform(
        buffer : Buffer,
        env,
        eval_env,
        env_info,
        save_dir,
        cfg
    ): 
    return UniformSetter(eval_env, cfg.env.num_envs)

def make_rig(
        buffer : Buffer,
        env,
        eval_env,
        env_info,
        save_dir,
        cfg
    ):
    
    
    setter = RigSetter(num_envs=cfg.env.num_envs,
                       eval_env=eval_env,
                       input_dim=cfg.env.latent_dim_obs
                     )
    
    setter.reward = cfg.env.reward
    setter.terminated = True
    
    return setter

def make_default(*args):
    return DefaultSetter()

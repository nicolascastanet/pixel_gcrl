# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .maze_env import Env
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, kl
from torch.utils import data
import matplotlib
from typing import Optional, Union, List
import io
import cv2
import matplotlib.pyplot as plt


class Obstacle:
  def __init__(self, top_left, bottom_right):
    self.top_left = top_left
    self.bottom_right = bottom_right

  def in_collision(self, points):
    if len(points.shape) == 1:
      return self.top_left[0] <= points[0] <= self.bottom_right[0] and \
           self.bottom_right[1] <= points[1] <= self.top_left[1]
    else:
      return np.logical_and(np.logical_and(np.logical_and(
        self.top_left[0] <= points[:,0], points[:,0] <= self.bottom_right[0]),
        self.bottom_right[1] <= points[:,1]), 
        points[:,1] <= self.top_left[1])


  def get_patch(self):
    # Create a Rectangle patch
    rect = matplotlib.patches.Rectangle((self.top_left[0], self.bottom_right[1]),
                                        self.bottom_right[0] - self.top_left[0],
                                        self.top_left[1] - self.bottom_right[1],
                                        linewidth=1,
                                        edgecolor='k',
                                        hatch='x',
                                        facecolor='none')
    return rect


class SimpleMazeEnv(gym.Env):
  """This is a long horizon (80+ step optimal trajectories), but also very simple point navigation task"""
  def __init__(self, test=False):
    self.pos_min = 0.0
    self.pos_max = 1.0
    self.dx = dx = 0.025
    self.dy = dy = 0.025
    s2 = 1/np.sqrt(2)*dx
    self.action_vec = np.array([[dx, 0], [-dx, 0], [0, dy], [0, -dy], [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]])
    self.dist_threshold = 0.10
    self.obstacles = [Obstacle([0.2, 0.8], [0.4, 0]), Obstacle([0.6, 1], [0.8, 0.2])]
    self.action_space = gym.spaces.Discrete(8)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    goal_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    self.observation_space = gym.spaces.Dict({
      'observation': observation_space,
      'desired_goal': goal_space,
      'achieved_goal': goal_space
    })

    self.s_xy = self.get_free_point()
    self.g_xy = self.get_free_point()

    self.max_steps = 250
    self.num_steps = 0
    self.test = test

  def seed(self, seed=None):
    np.random.seed(seed)

  def compute_reward(self, achieved_goal, desired_goal, info):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    goal_rew = -(d >= self.dist_threshold).astype(np.float32)
    coll_rew = self.in_collision(achieved_goal) * -20
    return goal_rew + coll_rew
  
  def render(self):
    raise NotImplementedError

  def step(self, action):
    #d_pos = action / np.linalg.norm(action) * self.dx
    d_pos = self.action_vec[action]
    self.s_xy = np.clip(self.s_xy + d_pos, self.pos_min, self.pos_max)
    reward = self.compute_reward(self.s_xy, self.g_xy, None)

    info = {}
    self.num_steps += 1

    if self.test:
      done = np.allclose(0., reward)
      info['is_success'] = done
    else:
      done = False
      info['is_success'] = np.allclose(0., reward)

    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
        'observation': self.s_xy,
        'achieved_goal': self.s_xy,
        'desired_goal': self.g_xy,
    }

    return obs, reward, done, info
      
  def in_collision(self, points):
    return np.any([x.in_collision(points) for x in self.obstacles], axis=0)

  def get_free_point(self):
    max_tries = 100
    point = np.random.rand(2)
    tries = 0
    while self.in_collision(point):
      point = np.random.rand(2)
      tries += 1
      if tries >= max_tries:
        return None
    return point

  def reset(self):
    self.num_steps = 0
    self.s_xy = self.get_free_point()
    self.g_xy = self.get_free_point()
    return {
        'observation': self.s_xy,
        'achieved_goal': self.s_xy,
        'desired_goal': self.g_xy,
    }
    
    
class PixelObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_shape = 82*82*3
        self.action_space = gym.spaces.Box(-0.95, 0.95, (img_shape, ))
        observation_space = gym.spaces.Box(-np.inf, np.inf, (img_shape, ))
        goal_space = gym.spaces.Box(-np.inf, np.inf, (img_shape, ))
        self.observation_space = gym.spaces.Dict({
          'observation': observation_space,
          'desired_goal': goal_space,
          'achieved_goal': goal_space
        })
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        env.maze.plot(ax=ax,lw=3)
        plt.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=16.5)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[np.where(img != 255)] = 0
        
        self.maze_struct_pixels = img
        self.range = 2
        
        
    def pixel_conversion(self,state):
        rg = self.range
        x_maze, y_maze = state
        x,y = int(65-y_maze*11.4),int(20+x_maze*11.4)
        state_pix = np.copy(self.maze_struct_pixels)
        state_pix[x-rg:x+rg,y-rg:y+rg,:] = [255,0,0]
        
        return state_pix

    def observation(self, obs):
        obs['observation'] = self.pixel_conversion(obs['observation'])
        obs['achieved_goal'] = self.pixel_conversion(obs['achieved_goal'])
        obs['desired_goal'] = self.pixel_conversion(obs['desired_goal'])
        
        return obs
      
    

class PointMaze2D(gym.Env):
  """Wraps the Sibling Rivalry 2D point maze in a gym goal env.
  Keeps the first visit done and uses -1/0 rewards.
  """
  def __init__(self, test=False, max_steps=50, maze_type='square_pbcs_0', dist_threshold=None, reward_type="sparse", action_scale=np.array([-0.95,0.95])):
    super().__init__()
    self._env = Env(n=50, maze_type=maze_type, use_antigoal=False, ddiff=False, ignore_reset_start=True)
    self.maze = self._env.maze
    if dist_threshold is None:
      self.dist_threshold = 0.15
    else:
      self.dist_threshold = dist_threshold
    self.size_max = max(self.maze._locs)
    self.size_min = min(self.maze._locs)

    self.action_scale = action_scale
    self.action_space = gym.spaces.Box(action_scale[0], action_scale[1], (2, ))
    
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    goal_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })
    
    self.reward_type = reward_type

    self.s_xy = np.array(self.maze.sample_start())
    self.g_xy = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.max_steps = max_steps
    self.max_episode_steps = max_steps
    self.num_steps = 0
    self.test = test

  def get_rgb_maze(self):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    self.maze.plot(ax=ax,lw=3)
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=16.5)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[np.where(img != 255)] = 0
    
    return img

  def change_maze_type(self, new_maze_type):
    self._env = Env(n=50, maze_type=new_maze_type, use_antigoal=False, ddiff=False, ignore_reset_start=True)
    self.maze = self._env.maze
    

  def seed(self, seed=None):
    return self.maze.seed(seed=seed)

  def step(self, action):
    action = np.clip(action, -self.action_scale, +self.action_scale)
  
    try:
      s_xy = np.array(self.maze.move(tuple(self.s_xy), tuple(action)))
    except:
      print('failed to move', tuple(self.s_xy), tuple(action))
      raise

    self.s_xy = s_xy
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1

    if self.test:
      done = np.allclose(0., reward)
      info['is_success'] = np.array(float(done))
    else:
      done = 0.0
      info['is_success'] = np.array(float(np.allclose(0., reward)))

    if self.num_steps >= self.max_steps and not done:
      done = 1.0
      info['TimeLimit.truncated'] = 1.0
      
    else:
      info['TimeLimit.truncated'] = 0.0
        
    terminated = info['is_success']
    truncated = info['TimeLimit.truncated']
    
    obs = {
        'observation': s_xy,
        'achieved_goal': s_xy,
        'desired_goal': self.g_xy,
    }

    return obs, reward, terminated, truncated, info

  def reset(self, 
            seed: Optional[Union[int, List[int]]] = None
          ):
    self.num_steps = 0
    s_xy = np.array(self.maze.sample_start())
    self.s_xy = s_xy
    g_xy = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.g_xy = g_xy
    return {
        'observation': s_xy,
        'achieved_goal': s_xy,
        'desired_goal': g_xy,
    }, {}

  def render(self):
    raise NotImplementedError
  

  def compute_reward(self, achieved_goal, desired_goal, info, *args, **kwargs):
    
    if self.reward_type == "sparse":
      d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
      reward = -(d >= self.dist_threshold).astype(np.float32)
      
    elif self.reward_type == "dense":
      reward = - np.linalg.norm(achieved_goal - desired_goal, axis=-1).astype(np.float32)
    
    return reward
  
  def is_success(self, achieved_goal, desired_goal, info=None):
    
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    sparse_reward = -(d >= self.dist_threshold).astype(np.float32)
    return sparse_reward + 1
  
  def tensor_point_maze_random(self, nb_samples_per_grid=10):    
    d_x_max = self.size_max[0]+1
    d_y_max = self.size_max[1]+1
    d_x_min = self.size_min[0]
    d_y_min = self.size_min[1]
    grid_size = int(min(d_x_max, d_y_max))
    
    x = np.linspace(d_x_min, d_x_max, grid_size+1)
    y = np.linspace(d_y_min, d_y_max, grid_size+1)
    random_goals = []

    for i in range(len(x)-1):
        for j in range(len(y)-1):
            #if i == 0 or j == 0 or j==(len(y)-2):
            data = np.random.uniform(low=[x[i],y[j]], high=[x[i+1],y[j+1]], size=(nb_samples_per_grid,2))
            random_goals.append(data)

    return np.array(random_goals).reshape(-1,2) - np.array([1,1])/2



class PointMazeND(PointMaze2D):
  def __init__(self,model,test=False, max_steps=50, maze_type='square_large_0'):
    super().__init__(test=test, max_steps=max_steps, maze_type=maze_type)

    model.eval_mode = True
    self.goal_dim = model.hidden_shape
    self.encoder = model.encoder
    self.decoder = model.decoder

    self.action_space = gym.spaces.Box(-0.95, 0.95, (2, ))
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    goal_space = gym.spaces.Box(-np.inf, np.inf, (self.goal_dim, ))
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })

    self.s_xy = np.array(self.maze.sample_start())
    real_goal = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.g_xy = self.encoder(real_goal).cpu().numpy()


  def step(self, action):
    try:
      s_xy = np.array(self.maze.move(tuple(self.s_xy), tuple(action)))
    except:
      print('failed to move', tuple(self.s_xy), tuple(action))
      raise

    self.s_xy = s_xy
    reward = self.compute_reward(s_xy, self.decoder(self.g_xy).cpu().numpy(), None)
    info = {}
    self.num_steps += 1

    if self.test:
      done = np.allclose(0., reward)
      info['is_success'] = done
    else:
      done = False
      info['is_success'] = np.allclose(0., reward)

    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
        'observation': s_xy,
        'achieved_goal': self.encoder(s_xy).cpu().numpy(),
        'desired_goal': self.g_xy,
    }

    return obs, reward, done, info

  def reset(self):
    self.num_steps = 0
    s_xy = np.array(self.maze.sample_start())
    self.s_xy = s_xy
    g_xy = self.encoder(np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold)))
    self.g_xy = g_xy
    return {
        'observation': s_xy,
        'achieved_goal': self.encoder(s_xy).cpu().numpy(),
        'desired_goal': g_xy,
    }





    

  



# 2 ==> 8 ==> 2
class AE(torch.nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super().__init__()
         
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.enc = nn.Sequential(
            nn.Linear(input_shape, hidden_shape)
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden_shape, input_shape)
        )

    def encoder(self, x):
      if isinstance(x,np.ndarray):
          x = torch.FloatTensor(x).to(self.device)
      return self.enc(x)

 
    def decoder(self, x):
      if isinstance(x,np.ndarray):
          x = torch.FloatTensor(x).to(self.device)
      return self.dec(x)


    def forward(self, x):
        if isinstance(x,np.ndarray):
          x = torch.FloatTensor(x).to(self.device)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


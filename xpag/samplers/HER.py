# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import torch
import numpy as np
import inspect
from xpag.samplers.sampler import Sampler


def encode_transitions(transitions, encoder):
    """
    transitions : np.array batch of n transition
    encoder : encoding funtion
    When working whin an representation function learned online, 
    we need to re-compute representation when learning the policy
    """
    
    assert 'observation.init_ag' in transitions
    
    ag_s = encoder(transitions['observation.init_obs'])
    ag_s_next = encoder(transitions['next_observation.init_obs'])
    
    transitions['observation.observation'] = np.copy(ag_s)
    transitions['observation.achieved_goal'] = np.copy(ag_s)
    #transitions['observation.desired_goal'] = encoder(transitions['observation.init_dg'])
    
    transitions['next_observation.observation'] = np.copy(ag_s_next)
    transitions['next_observation.achieved_goal'] = np.copy(ag_s_next)
    #transitions['next_observation.desired_goal'] = encoder(transitions['next_observation.init_dg'])
    
    return transitions


class HER(Sampler):
    def __init__(
        self,
        eval_env,
        need_encoding:bool = False,
        replay_strategy: str = "future"
    ):
        super().__init__()
        self.replay_strategy = replay_strategy
        self.replay_k = 4.0
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + self.replay_k))
        else:
            self.future_p = 0

        compute_reward = eval_env.compute_reward
        # Check if compute func has enough params
        if len(inspect.signature(compute_reward).parameters) < 4:
            def wrapped_reward_func(achieved_goal, goal, info, *args, **kwargs):
                return compute_reward(achieved_goal, goal, info)

            self.reward_func = wrapped_reward_func
        else:
            self.reward_func = compute_reward
            
        
        self.eval_env = eval_env
        self.need_encoding = need_encoding

    def sample(self, buffers, batch_size_in_transitions, **kwargs):
        rollout_batch_size = buffers["episode_length"].shape[0]
        batch_size = batch_size_in_transitions
        # select rollouts and steps
        episode_idxs = np.random.choice(
            np.arange(rollout_batch_size),
            size=batch_size,
            replace=True,
            p=buffers["episode_length"][:, 0, 0]
            / buffers["episode_length"][:, 0, 0].sum(),
        )
        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
        t_samples = np.random.randint(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        
        
        #import ipdb;ipdb.set_trace()
        #if self.need_encoding:
        #    # TODO : remove hardcoded embeding function
        #    transitions = encode_transitions(transitions, self.eval_env.convert_2D_to_embed)
        #import ipdb;ipdb.set_trace()
        
        # HER indexes
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        future_offset = np.random.uniform(size=batch_size) * (
            t_max_episodes - t_samples
        )
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        # replace desired goal with achieved goal
        future_ag = buffers["next_observation.achieved_goal"][
            episode_idxs[her_indexes], future_t
        ]

        
        transitions["observation.desired_goal"][her_indexes] = future_ag
        
        # check for additional input to compute reward
        if "next_observation.sig_ag" in transitions:
            input_dict = {"sig_ag" : transitions["next_observation.sig_ag"],
                            "mu_ag" : transitions["next_observation.mu_ag"]
                                }
        
        else:
            input_dict = None
                
        # recomputing rewards
        transitions["reward"] = np.expand_dims(
            self.reward_func(
                transitions["next_observation.achieved_goal"],
                transitions["observation.desired_goal"],
                transitions["action"],
                transitions["next_observation.observation"],
                other_args = input_dict
            ),
            1,
        )
        
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        transitions["observation"] = np.concatenate(
            [
                transitions["observation.observation"],
                transitions["observation.desired_goal"],
            ],
            axis=1,
        )
        transitions["next_observation"] = np.concatenate(
            [
                transitions["next_observation.observation"],
                transitions["observation.desired_goal"],
            ],
            axis=1,
        )

        return transitions

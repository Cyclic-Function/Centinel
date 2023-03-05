#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:49:59 2023

@author: yossarian
"""

from marl_control_envs import bunch_v0

import argparse
import os
# import pprint

# import control_envs
import gymnasium as gym
import numpy as np

import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Optional, Tuple, List

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    PPOPolicy,
    MultiAgentPolicyManager,
)
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-count', type=float, default=2)

    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=200)     # was 5
    parser.add_argument('--step-per-epoch', type=int, default=75000)   # was 150000
    parser.add_argument('--episode-per-collect', type=int, default=26)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=20)  # was 16
    parser.add_argument('--test-num', type=int, default=100)  # was 100
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.05)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]
    return args

def get_single_agent(args: argparse.Namespace, env) -> BasePolicy:
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    critic = Critic(
        Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
        device=args.device
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    agent = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
    )
    
    # if args.resume_path:
    #     agent.load_state_dict(torch.load(args.resume_path))
    # TODO: uncomment this
    
    return agent
    

def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    gym_attrs: Dict[str, any] = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:

    env = get_packaged_env(gym_attrs)
    
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    
    args.max_action = env.action_space.high[0]

    if agents is None:
        agents = [get_single_agent(args, env) for _ in range(gym_attrs['num_agents'])]
    
    policy = MultiAgentPolicyManager(agents, env, action_scaling=True,
                                     action_bound_method='clip')
    
    return policy, env.agents

def get_packaged_env(gym_attrs=None, render_mode=None, callable=False):
    # return gym.make('control_envs/ContinuousCartPole-v0', render_mode=render_mode)
    def get_env(render_mode=None):
        return PettingZooEnv(bunch_v0.env(gym_attrs=gym_attrs, render_mode=render_mode))

    if callable:
        return get_env
    else:
        return get_env(render_mode=render_mode)

def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    gym_attrs: Dict[str, any] = None
) -> Tuple[dict, BasePolicy]:
    train_envs = DummyVectorEnv([get_packaged_env(gym_attrs=gym_attrs, callable=True) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_packaged_env(gym_attrs=gym_attrs, callable=True) for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, agents = get_agents(
        args, agents=agents, gym_attrs=gym_attrs
    )
    
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs), ignore_obs_next=False)  ### ignore_obs_next=True
    )
    test_collector = Collector(policy, test_envs)
    # TODO: add exploration noise?
    # log
    log_path = os.path.join(args.logdir, "bunch", "ppo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False
    
    def reward_metric(rews):
        return rews[:, 0]

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                # "optim": optim.state_dict(),
            }, ckpt_path
        )
        return ckpt_path

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint["model"])
            # optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    rank = {
        agents[0]: 1,
        agents[1]: 1,
    }
    ############################################################################
    # In deterministic Centinel, if A has rank 1 and B has rank 9, then for    #
    # every 1 epoch that A trains, B trains 9 epochs. Epoch here is in         #
    # Tianshou terminology.                                                    #
    # In monte carlo Centinel, A would have 0.1 probabilitiy to be picked for  #
    # training 1 epoch and B would have 0.9 probability to be picked for       #
    # training                                                                 #
    ############################################################################

    for ag in agents:
        policy.freeze(ag, grad_freeze=True)

    res_log = False

    reps = 200
    for i in range(reps):
        print('-----------------------------------------------------------------')
        print(f"Rep: {i}")

        for ag in agents:
            policy.unfreeze(ag, grad_unfreeze=True)

            trainer_ag = OnpolicyTrainer(
                policy,
                train_collector,
                test_collector,
                rank[ag],
                args.step_per_epoch,
                args.repeat_per_collect,
                args.test_num,
                args.batch_size,
                episode_per_collect=args.episode_per_collect,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger,
                resume_from_log=res_log,
                save_checkpoint_fn=save_checkpoint_fn,
                reward_metric=reward_metric,
            )
            for epoch, epoch_stat, info in trainer_ag:
                print(f"Epoch: {epoch}")
                print(epoch_stat)
            
            policy.freeze(ag, grad_freeze=True)
            res_log = True
            

        save_info = {
            ag: policy.policies[ag].state_dict() for ag in agents
        }
        save_info['gym_attrs'] = gym_attrs
        torch.save(
            save_info, os.path.join(log_path, f'last_policy_{i}.pth')
        )

  
import math

gym_attrs = {
    'num_agents': 2,
    'target_manager': 'TargetManagerMean',
    # 'target_manager': 'TargetManagerDebug2D',
    'reward_type': 'dist_centinel_exp',
    # 'reward_type': 'end_dist_cooperative',
    # 'reward_split': 0.5,
    'max_steps': 200,
    'env_type': 'parallel',
    'max_error_radius': 0.05,
    'termination_reward':500,
}

args = get_args()
policy = train_agent(args, gym_attrs=gym_attrs)
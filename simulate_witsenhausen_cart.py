#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:58:47 2023

@author: yossarian
"""

from marl_control_envs import witsenhausen_cart_v0

import argparse
import os
# import pprint

# import control_envs
import gymnasium as gym
import numpy as np
from sklearn.linear_model import LinearRegression

import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Optional, Tuple

from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    MultiAgentPolicyManager,
    # RandomPolicy,
    PPOPolicy
)
from tianshou.trainer import onpolicy_trainer, OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='control_envs/ContinuousCartPole-v0')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=1)     # was 5
    parser.add_argument('--step-per-epoch', type=int, default=30000)   # was 150000
    parser.add_argument('--episode-per-collect', type=int, default=26)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=1)  # was 16
    parser.add_argument('--test-num', type=int, default=1)  # was 100
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
    agent_weak: Optional[BasePolicy] = None,
    agent_strong: Optional[BasePolicy] = None,
    gym_attrs: Dict[str, any] = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    pass
    # currently only implemented for one agent
    
    # env = gym.make(args.task)
    env = get_packaged_env(gym_attrs=gym_attrs)
    
    # args.state_shape = env.observation_space.shape or env.observation_space.n
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    # print('old')
    # print(env.observation_space.shape or env.observation_space.n)
    # print('new')
    # print(args.state_shape)
    # print('+')
    
    args.action_shape = env.action_space.shape or env.action_space.n
    
    args.max_action = env.action_space.high[0]
    # print(args.max_action, 'maxi')
    
    if agent_weak is None:
        agent_weak = get_single_agent(args, env)
    if agent_strong is None:
        agent_strong = get_single_agent(args, env)
    # TODO: single instance of env or multiple?
    
    agents = [agent_weak, agent_strong]
    policy = MultiAgentPolicyManager(agents, env, action_scaling=True,
                                     action_bound_method='clip')
    
    # print(env.agents, 'Ag')   # what is this
    
    return policy, env.agents

def get_packaged_env(gym_attrs=None, render_mode=None, callable=False, test_reward=False):
    # return gym.make('control_envs/ContinuousCartPole-v0', render_mode=render_mode)
    def get_env(render_mode=None):
        return PettingZooEnv(witsenhausen_cart_v0.env(gym_attrs=gym_attrs, render_mode=render_mode, test_reward=test_reward))

    if callable:
        return get_env
    else:
        return get_env(render_mode=render_mode)

def train_agent(
    args: argparse.Namespace = get_args(),
    agent_weak: Optional[BasePolicy] = None,
    agent_strong: Optional[BasePolicy] = None,
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
        args, agent_weak=agent_weak, agent_strong=agent_strong, gym_attrs=gym_attrs
    )
    
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)
    # TODO: add exploration noise?
    # log
    log_path = os.path.join(args.logdir, "wits_cartpole", "ppo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False

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
    
    if args.watch_demo:
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "last_policy.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            gym_attrs = checkpoint['gym_attrs']
            # gym_attrs['k'] = 0.4
            
            if 'survival_bonus' in gym_attrs:
                gym_attrs['survival_reward'] = gym_attrs['survival_bonus']
                del gym_attrs['survival_bonus']
            
            print(gym_attrs)
            
            env = get_packaged_env(gym_attrs=gym_attrs)
            agent_weak = get_single_agent(args, env)
            agent_weak.load_state_dict(checkpoint["agent_weak"])
            agent_strong = get_single_agent(args, env)
            agent_strong.load_state_dict(checkpoint["agent_strong"])
            # optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        trainer = None
        policy, agents = get_agents(
            args, agent_weak=agent_weak, agent_strong=agent_strong, gym_attrs=gym_attrs
        )
        # print(policy.policies[agents[1]].state_dict())
    
    return trainer, policy, gym_attrs

def watch(
    args: argparse.Namespace = get_args(),
    policy=None,
    gym_attrs: Dict[str, any] = None
) -> None:
    
    # print('how')
    
    env1 = get_packaged_env(gym_attrs=gym_attrs, render_mode="human", test_reward=True)
    
    env = DummyVectorEnv([lambda: env1])
    policy.eval()
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    # print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    env.close()
    
    return env1.env.env.env.env

args = get_args()
args.watch_demo = True
result, policy, gym_attrs = train_agent(args)
gym_attrs['debug_params'] = ['track trajectory']
ev = watch(args, policy, gym_attrs=gym_attrs)
print(gym_attrs)

'''
works well, changing max_steps, term and surv reward
tryign k=0.2
'''
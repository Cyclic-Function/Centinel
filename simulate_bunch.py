from marl_control_envs import bunch_v0

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

from typing import Dict, Optional, Tuple, List

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
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=100)     # was 5
    parser.add_argument('--step-per-epoch', type=int, default=150000)   # was 150000
    parser.add_argument('--episode-per-collect', type=int, default=64)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[1024, 1024, 1024, 1024])
    parser.add_argument('--training-num', type=int, default=16)  # was 16
    parser.add_argument('--test-num', type=int, default=16)  # was 100
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

def get_packaged_env(gym_attrs=None, render_mode=None, test_reward=False, callable=False):
    # return gym.make('control_envs/ContinuousCartPole-v0', render_mode=render_mode)
    def get_env(render_mode=None):
        return PettingZooEnv(bunch_v0.env(gym_attrs=gym_attrs, render_mode=render_mode, test_reward=test_reward))

    if callable:
        return get_env
    else:
        return get_env(render_mode=render_mode)

def retrieve_agents(
    args: argparse.Namespace = get_args()
) -> Tuple[dict, BasePolicy]:
    log_path = os.path.join(args.logdir, "brunch", "ppo")
    
    print(f"Loading agent under {log_path}")
    ckpt_path = os.path.join(log_path, "last_policy.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        gym_attrs = checkpoint['gym_attrs']
        # gym_attrs['num_agents'] = 2
        # gym_attrs['target_manager'] = 
        
        policy, agents = get_agents(
            args, gym_attrs=gym_attrs
        )
        
        for i in agents:
            policy.policies[i].load_state_dict(checkpoint[i])
        
        print("Successfully restore policy and optim.")
    else:
        print("Fail to restore policy and optim.")
        policy = None
        gym_attrs = None
    
    return policy, gym_attrs

def watch(
    args: argparse.Namespace = get_args(),
    policy=None,
    gym_attrs: Dict[str, any] = None
) -> None:
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
policy, gym_attrs = retrieve_agents(args)

if policy is not None:
    env = watch(args, policy, gym_attrs=gym_attrs)
else:
    assert False

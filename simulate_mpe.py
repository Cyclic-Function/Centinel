import argparse
import os
from copy import deepcopy
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from marl_control_envs import bunch_v0


"""This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_single_agent(args: argparse.Namespace) -> BasePolicy:
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    return agent
    

def get_agents(
    args: argparse.Namespace = get_args(),
    n_agents: int = 2,
    agents: List[Optional[BasePolicy]] = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    if agents is None:
        agents = [get_single_agent(args) for _ in range(n_agents)]
    
    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents


def get_env(render_mode=None):
    return PettingZooEnv(bunch_v0.env(
               local_ratio=0.5, max_cycles=25, continuous_actions=False
           ))

def train_agent(
    args: argparse.Namespace = get_args(),
    agents: List[Optional[BasePolicy]] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    n_agents = 2

    # ======== agent setup =========
    policy, agents = get_agents(
        args, n_agents=n_agents, agents=agents
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    # train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "bunch", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "tic_tac_toe", "dqn", "policy.pth"
            )
        torch.save(
            policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return False

    # def train_fn(epoch, env_step):
    #     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    # def test_fn(epoch, env_step):
    #     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    agents_state_dict = {f'agent_{i}': policy.policies[agents[i]].state_dict() for i in range(n_agents)}
    torch.save(agents_state_dict, os.path.join(log_path, "finished_policy.pth"))

    return result, policy


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agents: List[Optional[BasePolicy]] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    policy, optim, agents = get_agents(
        args, agents=agents
    )
    policy.eval()
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")
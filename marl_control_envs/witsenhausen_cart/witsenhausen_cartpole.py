#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:53:26 2023

@author: yossarian
"""

import math

import gymnasium
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.utils import seeding

import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from typing import Optional, Any, Dict

from marl_control_envs.witsenhausen_cart.witsenhausen_dynamics import WitsenhausenDynamics
from marl_control_envs.witsenhausen_cart.witsenhausen_rewards import WitsenhausenRewardUSquare
from marl_control_envs.witsenhausen_cart.witsenhausen_render import WitsenhausenRender


class WitsenhausenCartPole:
    def __init__(
            self, np_random, gym_attrs: Dict[str, any], metadata: Dict[str, Any],
            render_mode: Optional[str] = None, test_reward=False):
        valid_gym_attrs = (
            'max_steps', 'force_scaling', 'k', 'survival_reward', 'termination_reward',
            'reward_type', 'theta_threshold_radians', 'init_state_sd',
            'agent_strong_unobservable_states', 'agent_strong_obs_noise_sd',
            'gravity', 'debug_params', 'env_type'
        )
        assert set(gym_attrs.keys()) <= set(valid_gym_attrs), f'bruh {set(gym_attrs.keys()) - set(valid_gym_attrs)}'
        
        self.num_agents = 2
        self.agents = ['agent_weak', 'agent_strong']
        self.possible_agents = self.agents[:]
        
        self.steps_beyond_terminated = None
        self.step_count = None
        self.max_steps = gym_attrs.get('max_steps', 500)
        
        self.force_scaling = gym_attrs.get('force_scaling', 15.0)
        
        k = gym_attrs.get('k', 0.4)    # Witsenhausen parameter
        reward_scale = 1e3
        survival_reward = gym_attrs.get('survival_reward', 175.0)/reward_scale
        termination_reward = gym_attrs.get('termination_reward', -500.0)
        test_reward = test_reward
        reward_type = gym_attrs.get('reward_type', 'u_square')
        assert reward_type in ('u_square', 'energy')
        if reward_type == 'u_square':
            self.reward_handler = WitsenhausenRewardUSquare(
                self.agents, k, survival_reward, termination_reward,
                test_reward, reward_scale/self.max_steps
            )
        elif reward_type == 'energy':
            assert False
        
        self.theta_threshold_radians = gym_attrs.get('theta_threshold_radians', 24 * 2 * math.pi / 360)       # TODO: IMP was 12
        self.x_threshold = 2.4
        
        self.max_action = 1.0
        self.action_spaces = {
            i: spaces.Box(
                low=-self.max_action, high=self.max_action,
                shape=(1,), dtype=np.float32,
               )
            for i in self.agents
        }
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        obs_high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_spaces = {
            i: spaces.Dict(
                {'observation': spaces.Box(-obs_high, obs_high, dtype=np.float32)}
               )
            for i in self.agents
        }
        
        epsilon = 0.025
        init_state_sd = gym_attrs.get('init_state_sd', ['epsilon', 'epsilon', self.theta_threshold_radians/4, 'epsilon'])
        init_state_sd = np.array([epsilon if sd == 'epsilon' else sd for sd in init_state_sd])
        agent_strong_unobservable_states = gym_attrs.get(
            'agent_strong_unobservable_states',
            []
        )
        agent_strong_obs_noise_sd = gym_attrs.get(
            'agent_strong_obs_noise_sd',
            [0, 0, self.theta_threshold_radians/(4*5), 0]
        )
        gravity = gym_attrs.get('gravity', 9.8)
        length = 0.5  # actually half the pole's length
        self.witsenhausen_dynamics = WitsenhausenDynamics(
            np_random, self.agents, init_state_sd,
            agent_strong_unobservable_states, agent_strong_obs_noise_sd,
            gravity=gravity, length=length
        )
        self.witsenhausen_render = WitsenhausenRender(
            metadata, length, self.x_threshold, render_mode
        )
        
        self.debug_params = gym_attrs.get('debug_params', [])
        
        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}   # TODO: why does this exist

        self.render_mode = render_mode
        
        if 'track trajectory' in self.debug_params:
            self.traj = np.zeros(self.max_steps)
        
        self.env_type = gym_attrs.get('env_type', 'AEC')
        assert self.env_type in ('AEC', 'parallel'), 'Please pick a valid env_type, dumbass'
        if self.env_type == 'parallel':
            self.force_history = {
                i: None
                for i in self.agents
            }
    
    def reset(self):
        self.witsenhausen_dynamics.reset()
                
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}   # TODO: why does this exist
        
        self.steps_beyond_terminated = None
        self.step_count = 0
        
        if self.env_type == 'parallel':
            self.force_history = {
                i: None
                for i in self.agents
            }

        if self.render_mode == 'human':
            self.render()
    
    def observe(self, agent):
        return self.witsenhausen_dynamics.observe(agent)
    
    def step(self, action, agent):
        assert self.witsenhausen_dynamics.state is not None, 'Call reset before using step method.'
        assert agent in self.agents, 'Please pick a valid agent'
        
        action = np.clip(action, -self.max_action, self.max_action)
        force = action[0]*self.force_scaling
        if 'agent_strong zero' in self.debug_params:
            if agent == self.agent_strong:
                force = 0
        
        if self.env_type == 'AEC':
            self.witsenhausen_dynamics.update_state(force)
        elif self.env_type == 'parallel':
            self.force_history[agent] = force
            if agent == self.agents[-1]:
                force = sum(self.force_history.values())
                self.witsenhausen_dynamics.update_state(force)
                self.force_history = {i: None for i in self.agents}
        
        x = self.witsenhausen_dynamics.state[0]
        theta = self.witsenhausen_dynamics.state[2]

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = False
        
        if 'track trajectory' in self.debug_params:
            self.traj[self.step_count] = action
        
        global_reward = 0.0
        for i in self.agents:
            # reset rewards
            self.rewards[i] = 0.0
        
        if not terminated:
            global_reward += self.reward_handler.get_reward(
                agent, force,
                self.witsenhausen_dynamics.tau,
                self.witsenhausen_dynamics.state
            )
            
            self.step_count += 1
            if self.step_count >= self.max_steps:
                truncated = True
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            global_reward += self.reward_handler.termination_reward
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
        
        for i in self.agents:
            self.rewards[i] += global_reward
            self.terminations[i] = terminated
            self.truncations[i] = truncated
            self.infos[i] = {}
        
        if self.render_mode == "human":
            if self.env_type == 'AEC':
                self.render()
            elif self.env_type == 'parallel':
                if agent == self.agents[-1]:
                    self.render()
    
    def render(self):
        self.witsenhausen_render.render(self.witsenhausen_dynamics.state)
    
    def close(self):
        self.witsenhausen_render.close()

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    def __init__(
        self, gym_attrs: Dict[str, any] = None,
        render_mode: Optional[str] = None,
        test_reward: Optional[bool] = False,
    ):
        super().__init__()
        self.seed()
        
        self.metadata = {
            'render_modes': ['human', 'rgb_array'],
            'name': 'witsenhausen_cartpole_v0',
            'is_parallelizable': False,     # TODO: make this true!
            'render_fps': 50,
        }
        
        self.gym_attrs = gym_attrs if gym_attrs is not None else dict()
        
        self.render_mode = render_mode
        self.test_reward = test_reward
        self.set_env()
        
        self.agents = self.env.agents[:]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        self.action_spaces = self.env.action_spaces
        self.observation_spaces = self.env.observation_spaces
        
        self.update_env_vars()
    
    def observe(self, agent):
        obs = self.env.observe(agent)
        return obs

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
    
    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        
        self.env.reset()
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = {i: 0 for i in self.agents}
        
        self.update_env_vars()
    
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection

        self.env.step(action, agent)
        # select next agent and observe
        self.agent_selection = self._agent_selector.next()
        
        self.update_env_vars()

        self._accumulate_rewards()
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def update_env_vars(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_env(self):
        self.env = WitsenhausenCartPole(
            self.np_random, self.gym_attrs, self.metadata, self.render_mode,
            self.test_reward
        )

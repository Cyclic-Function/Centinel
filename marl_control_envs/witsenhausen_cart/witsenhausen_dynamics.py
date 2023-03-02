#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:51:21 2023

@author: yossarian
"""

import math
import numpy as np

class WitsenhausenDynamics:
    def __init__(
            self, np_random, agents, init_state_sd,
            agent_strong_unobservable_states, agent_strong_obs_noise_sd,
            gravity=9.8
        ):
        self.np_random = np_random
        self.agent_weak, self.agent_strong = agents
        
        self.init_state_sd = init_state_sd
        self.agent_strong_unobservable_states = agent_strong_unobservable_states
        self.agent_strong_obs_noise_sd = agent_strong_obs_noise_sd
        self.state = None
        self.agent_strong_observation_noise = None
        
        self.gravity = gravity
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.tau = 0.01  # seconds between state updates        
        self.kinematics_integrator = 'euler'
    
    def reset(self):
        self.state = self.np_random.normal(
            loc=0.0,
            scale=self.init_state_sd
        )
        self.agent_strong_observation_noise = self.np_random.normal(
            loc=0.0,
            scale=self.agent_strong_obs_noise_sd
        )
    
    def observe(self, agent):
        assert self.state is not None
        assert agent in (self.agent_weak, self.agent_strong)
        
        if agent == self.agent_weak:
            return self.state
        elif agent == self.agent_strong:
            noise = self.agent_strong_observation_noise
            strong_state = np.copy(self.state)
            strong_state[self.agent_strong_unobservable_states] = 0
            
            return strong_state + noise
    
    def update_state(self, force):
        x, x_dot, theta, theta_dot = self.state
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            dx = self.tau * x_dot
            x = x + dx
            x_dot = x_dot + self.tau * xacc
            dtheta = self.tau * theta_dot
            theta = theta + dtheta
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'semi-implicit euler':  # semi-implicit euler
            dx = self.tau * x_dot
            x = x + dx
            x_dot = x_dot + self.tau * xacc
            theta_dot = theta_dot + self.tau * thetaacc
            dtheta = self.tau * theta_dot
            theta = theta + dtheta
        else:
            assert False, 'Please pick a valid integrator dumbass'

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

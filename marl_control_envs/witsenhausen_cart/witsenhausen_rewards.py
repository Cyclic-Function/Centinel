#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:52:20 2023

@author: yossarian
"""

class WitsenhausenRewardUSquare:
    def __init__(self, agents, k, survival_reward, termination_reward, test_reward, reward_scale):
        self.agent_weak, self.agent_strong = agents
        
        self.k = k
        self.survival_reward = survival_reward
        self.termination_reward = termination_reward
        self.test_reward = test_reward
        self.reward_scale = reward_scale
    
    def get_reward(self, agent, u, dt, state):
        if self.test_reward:
            agent_survival_reward = 0.0
        else:
            agent_survival_reward = self.survival_reward
        
        if agent == self.agent_weak:
            agent_k = 0.0
        elif agent == self.agent_strong:
            agent_k = self.k
        
        theta = state[2]
        
        return (agent_survival_reward - theta**2 - (agent_k**2)*(u**2)*dt)*self.reward_scale

class WitsenhausenRewardEnergy:
    pass

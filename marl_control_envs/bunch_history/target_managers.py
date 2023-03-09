#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:02:48 2023

@author: yossarian
"""

import numpy as np


class TargetManagerCoordinates:
    '''
    There can be different target managers. This one is for 2 agents only.
    Each agent_1 knows x and agent_2 knows y.
    '''
    def __init__(self, np_random, agents, pos_max=1.0):
        assert len(agents) == 2
        self.num_agents = len(agents)
        
        self.np_random = np_random
        
        self.pos_max = pos_max
        
        self.agent_0, self.agent_1 = agents
        self.agent_local_target = {i:None for i in agents}
        self.global_target = None
    
    def reset(self):
        """
        Implicitly assumed coordinates between -1, 1
        """
        x = self.np_random.uniform(-self.pos_max, self.pos_max)
        y = self.np_random.uniform(-self.pos_max, self.pos_max)
        
        self.agent_local_target = {
            self.agent_0: np.array([x, 0.0]),
            self.agent_1: np.array([0.0, y])
        }
        
        self.global_target = np.array([x, y])
    
    def get_local_target(self, agent):
        return self.agent_local_target[agent]

class TargetManagerDebug2D:
    '''
    Both agents know exact targets
    '''
    def __init__(self, np_random, agents, pos_max=1.0):
        assert len(agents) == 2
        
        self.np_random = np_random
        
        self.pos_max = pos_max
        
        self.agent_0, self.agent_1 = agents
        self.agent_local_target = {i:None for i in agents}
        self.global_target = None
    
    def reset(self):
        """
        Implicitly assumed coordinates between -1, 1
        """
        x = self.np_random.uniform(-self.pos_max, self.pos_max)
        y = self.np_random.uniform(-self.pos_max, self.pos_max)
        
        self.agent_local_target = {
            self.agent_0: np.array([x, y]),
            self.agent_1: np.array([x, y])
        }
        
        self.global_target = np.array([x, y])
    
    def get_local_target(self, agent):
        return self.agent_local_target[agent]


class TargetManagerMean:
    '''
    There can be different target managers.
    Go to the mean coordinate of the agents.
    assuming 2d coordinate system
    '''
    def __init__(self, np_random, agents, pos_max=1.0):
        self.num_agents = len(agents)
        
        self.np_random = np_random
        
        self.pos_max = pos_max
        
        self.agents = agents
        self.agent_local_target = {i: None for i in self.agents}
        self.global_target = None
    
    def reset(self):
        self.agent_local_target = {
            agent: self.np_random.uniform(-self.pos_max, self.pos_max, size=2)
            for agent in self.agents
        }
        self.global_target = np.mean([self.agent_local_target[agent] for agent in self.agents], axis=0)
    
    def get_local_target(self, agent):
        return self.agent_local_target[agent]

class TargetManagerMeanImpossible:
    '''
    There can be different target managers.
    Go to the mean coordinate of the agents.
    assuming 2d coordinate system
    '''
    def __init__(self, np_random, agents, pos_max=1.0):
        assert len(agents) == 2, 'only tested for 2 agents'
        self.num_agents = len(agents)
        
        self.np_random = np_random
        
        self.pos_max = pos_max
        
        self.agents = agents
        self.agent_local_target = {i: None for i in self.agents}
        self.global_target = None
    
    def reset(self):
        self.agent_local_target = {
            agent: self.np_random.uniform(-self.pos_max, self.pos_max, size=2)
            for agent in self.agents
        }
        
        self.global_target = (np.mean([self.agent_local_target[agent] for agent in self.agents], axis=0) + self.np_random.uniform(-self.pos_max, self.pos_max, size=2))/2.0
    
    def get_local_target(self, agent):
        return self.agent_local_target[agent]

class TargetManagerWeightedMean:
    '''
    There can be different target managers.
    Go to the mean coordinate of the agents.
    assuming 2d coordinate system
    '''
    def __init__(self, np_random, agents, weights, pos_max=1.0):
        assert len(agents) == 2, 'only tested for 2 agents'
        self.num_agents = len(agents)
        
        self.np_random = np_random
        
        self.pos_max = pos_max
        
        self.agents = agents
        self.agent_local_target = {i: None for i in self.agents}
        self.global_target = None
        
        self.weights = weights
    
    def reset(self):
        self.agent_local_target = {
            agent: self.np_random.uniform(-self.pos_max, self.pos_max, size=2)
            for agent in self.agents
        }
        
        self.global_target = np.average(
            [self.agent_local_target[agent] for agent in self.agents],
            weights=self.weights, axis=0
        )
    
    def get_local_target(self, agent):
        return self.agent_local_target[agent]


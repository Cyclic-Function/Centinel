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
        
        # self.initial_dists = {}
        # self.final_dists = {}
        
        self.init_pos = {}
    
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
        
        # self.initial_dists = {}
        # self.final_dists = {}
        
        self.init_pos = {}
    
    def add_init_pos(self, agent, init_pos_ag):
        self.init_pos[agent] = init_pos_ag
    
    def get_init_pos(self, agent=None):
        if agent is None:
            return self.init_pos
        else:
            return self.init_pos[agent]
    
    # def add_initial_dist(self, agent, pos_init):
    #     """
    #     Must call reset first
    #     """
    #     self.initial_dists[agent] = np.linalg.norm(pos_init - self.global_target)
    
    # def add_final_dist(self, agent, pos_final):
    #     self.final_dists[agent] = np.linalg.norm(pos_final - self.global_target)
        
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
        
        # self.initial_dists = {}
        # self.final_dists = {}
        
        self.init_pos = {}
    
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
        
        # self.initial_dists = {}
        # self.final_dists = {}
        
        self.init_pos = {}
    
    def add_init_pos(self, agent, init_pos_ag):
        self.init_pos[agent] = init_pos_ag
    
    def get_init_pos(self, agent=None):
        if agent is None:
            return self.init_pos
        else:
            return self.init_pos[agent]
    
    # def add_initial_dist(self, agent, pos_init):
    #     """
    #     Must call reset first
    #     """
    #     self.initial_dists[agent] = np.linalg.norm(pos_init - self.global_target)
    
    # def add_final_dist(self, agent, pos_final):
    #     self.final_dists[agent] = np.linalg.norm(pos_final - self.global_target)
        
    def get_local_target(self, agent):
        return self.agent_local_target[agent]
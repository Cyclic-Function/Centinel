#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:36:45 2023

@author: yossarian
"""

from marl_control_envs import bunch_v0
import numpy as np

env = bunch_v0.env(render_mode='human')
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = np.array([0.2, 0.3], dtype=np.float32)
    env.step(action)

env.close()
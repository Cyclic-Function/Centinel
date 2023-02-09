"""
big brain stuff bouta happen
"""

from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

# from marl_control_envs.bunch.target_managers import TargetManagerDebug2D as TargetManager

import marl_control_envs.bunch.target_managers as tm

from typing import Optional, Any, Dict


class Bunch:
    class FinderAgent:
        def __init__(self, np_random, dt=0.01, pos_max=1.0):
            self.np_random = np_random
            
            self.dt = dt
            self.m = 1.0
            self.force_scale = 10.0
            
            self.pos_max = pos_max
            
            self.state = None
            
            self.kinematics_integrator = 'euler'
            
            self.init_state = None
        
        def reset(self):
            self.state = np.concatenate([
                self.np_random.uniform(-self.pos_max, self.pos_max, size=2),
                [0.0, 0.0]      # xdot, ydot
            ])
            self.init_state = np.copy(self.state)
        
        def get_init_pos(self, magnitude=False):
            if magnitude:
                return np.linalg.norm(self.init_state[0:2])
            else:
                return self.init_state[0:2]
        
        def get_pos(self, magnitude=False):
            if magnitude:
                return np.linalg.norm(self.state[0:2])
            else:
                return self.state[0:2]
        
        def get_vel(self, magnitude=False):
            if magnitude:
                return np.linalg.norm(self.state[2:4])
            else:
                return self.state[2:4]
            
        def update_state(self, F):  # vector force
            Fx, Fy = F*self.force_scale
            x, y, xdot, ydot = self.state
            
            xacc = Fx/self.m
            yacc = Fy/self.m
            
            if self.kinematics_integrator == 'euler':
                dx = self.dt * xdot
                x = x + dx
                xdot = xdot + self.dt * xacc
                
                if x > self.pos_max:
                    x = self.pos_max
                    xdot = 0.0
                elif x < -self.pos_max:
                    x = -self.pos_max
                    xdot = 0.0
                
                dy = self.dt * ydot
                y = y + dy
                ydot = ydot + self.dt * yacc
                
                if y > self.pos_max:
                    y = self.pos_max
                    ydot = 0.0
                elif y < -self.pos_max:
                    y = -self.pos_max
                    ydot = 0.0
            # else:
            #     assert False, 'pick a valid integrator dumbass'
            
            self.state = np.array([x, y, xdot, ydot])


    def __init__(self, np_random, gym_attrs, metadata, render_mode=None, test_reward=False):
        self.np_random = np_random
        self.metadata = metadata
        self.test_reward = test_reward
        
        self.num_agents = gym_attrs['num_agents']
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.possible_agents = self.agents.copy()
        
        self.max_action = np.array(
            [1.0, 1.0],
            dtype=np.float32
        )
        self.action_spaces = {
            i: spaces.Box(
                low=-self.max_action, high=self.max_action,
                shape=(2,), dtype=np.float32,
               )
            for i in self.agents
        }
          
        self.pos_max = 1.0    # max x or y
        self.vel_max = np.finfo(np.float32).max
        single_agent_high = np.array(
            [
                self.pos_max*1.1,     # x
                self.pos_max*1.1,     # y
                self.vel_max,         # xdot
                self.vel_max,         # ydot
            ],dtype=np.float32,
        )
        local_target = np.array(
            [
                self.pos_max*1.1,     # local target x
                self.pos_max*1.1,     # local target y
            ],dtype=np.float32,
        )
        all_agent_high = np.resize(single_agent_high, len(single_agent_high)*self.num_agents)
        obs_high = np.concatenate([local_target, all_agent_high])
        assert obs_high.dtype == np.float32
        
        self.observation_spaces = {
            i: spaces.Dict(
                {"observation": spaces.Box(-obs_high, obs_high, dtype=np.float32)}
               )
            for i in self.agents
        }
        # print(f'obspace: {self.observation_spaces}')
        # TODO: make only self velocity observable but others velocity unobservable?
        
        # termination conditions, termination = good
        self.pos_max_error = gym_attrs.get('max_error_radius', 0.01)
        self.vel_max_error = gym_attrs.get('max_error_velocity', 0.02)
        
        self.finder_agents = {i: self.FinderAgent(self.np_random) for i in self.agents}
        
        target_manager_type = gym_attrs['target_manager']
        if target_manager_type == 'TargetManagerDebug2D':
            self.target_manager = tm.TargetManagerDebug2D(self.np_random, self.agents)
        elif target_manager_type == 'TargetManagerCoordinates':
            self.target_manager = tm.TargetManagerCoordinates(self.np_random, self.agents)
        else:
            assert False, 'as fast as a glacier, like always'
        
        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        
        if self.test_reward:
            self.termination_reward = 0.0
        else:
            self.termination_reward = gym_attrs.get('termination_reward', 500.0)
        
        # TODO change test reward to consider reward_type
        # TODO add reward type in the step function
        
        self.reward_type = gym_attrs.get('reward_type', 'cooperative')
        # print(self.reward_type)
        # currently, there are two reward types
        # cooperative means that agents get the cooperative reward,
        # ie sum of rewards of agent 0 and 1
        # centinel reward is for situations where the other agents are frozen,
        # so agent can only optimise its own behaviour, so the reward is only
        # for that agent        
        
        self.render_mode = render_mode
        
        self.screen_length = 600
        self.screen = None
        self.clock = None
        self.isopen = True

        self.steps_beyond_terminated = None
        
        self.agent_colours = {
            self.agents[0]: (0, 255, 0), 
            self.agents[1]: (0, 0, 255),
        }       # TODO: generalise to n agents!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.target_colour = (255, 0, 0)
        
        self.step_count = 0
        single_agent_max_steps = gym_attrs.get('max_steps', 500)
        self.max_steps = single_agent_max_steps*self.num_agents        # TODO: add termination conditions!
    
    def reset(self):
        self.target_manager.reset()
        for i in self.agents:
            self.finder_agents[i].reset()
        
        self.steps_beyond_terminated = None
        
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        
        self.step_count = 0
        
        if self.render_mode == "human":
            self.render()
    
    def observe(self, agent):
        local_target = self.target_manager.get_local_target(agent)
        all_agent_states = np.concatenate([self.finder_agents[i].state for i in self.agents])
        obs = np.concatenate([local_target, all_agent_states])
        
        # if agent == self.agents[1]:
        #     print(f'{agent}, loc: {local_target}, glob: {self.target_manager.global_target} sel: {all_agent_states[0:2]}, oth: {all_agent_states[4:6]}')
        
        # print(f'step: {self.step_count}, obs: {all_agent_states}')
        return obs
    
    def step(self, action, agent):
        assert self.finder_agents[agent].state is not None, "Call reset before using step method."
        
        action = np.clip(action, -self.pos_max, self.pos_max)
        self.finder_agents[agent].update_state(action)
        
        truncated = False
        # all_agents_within_max_pos_error = np.all(
        #     [np.linalg.norm(global_target - self.finder_agents[i].get_pos()) < self.pos_max_error for i in self.agents]
        # )
        # all_agents_within_max_vel_error = np.all(
        #     [self.finder_agents[i].get_vel(magnitude=True) < self.vel_max_error for i in self.agents]
        # )
        # terminated = bool(
        #     all_agents_within_max_pos_error
        #     and all_agents_within_max_vel_error
        # )
        terminated = False
        ###############################
        ###############################
        # TERMINATED IS FALSE MY DEAR #
        ###############################
        ###############################
        
        agents_init_pos = {i: self.finder_agents[i].get_init_pos() for i in self.agents}
        agents_cur_pos = {i: self.finder_agents[i].get_pos() for i in self.agents}
        
        global_target = self.target_manager.global_target
                
        for i in self.agents:
            # reset rewards
            self.rewards[i] = 0.0
        
        global_reward = 0.0
        
        if not terminated:
            if self.step_count >= self.max_steps:
                truncated = True
            
            if self.reward_type == 'dist_cooperative':
                global_reward += -np.mean([
                    np.linalg.norm(global_target - agents_cur_pos[i]) for i in self.agents
                ])/self.num_agents  # more normalising conditions - due to cyclic nature
            elif self.reward_type == 'dist_centinel':
                # in this mode, the other agent is frozen, so only current
                # agent's reward is reported, rather than sum of everyone's
                # reward
                self.rewards[agent] += -np.linalg.norm(global_target - agents_cur_pos[agent])
            elif self.reward_type == 'end_dist_cooperative':
                # WARN: NO TERMINATION CONDITION
                if truncated:
                    global_reward += -np.mean([
                        np.linalg.norm(global_target - agents_cur_pos[i]) for i in self.agents
                    ])
            elif self.reward_type == 'end_dist_centinel':
                if truncated:
                    for i in self.agents:
                        self.reward[i] += -np.linalg.norm(global_target - agents_cur_pos[i])
            elif self.reward_type == 'prop_cooperative':
                global_reward += np.mean([
                    (agents_init_pos[i] - agents_cur_pos[i])/agents_init_pos[i]
                    for i in self.agents
                ])*100/self.num_agents
            elif self.reward_type == 'prop_centinel':
                self.rewards[agent] = (agents_init_pos[agent] - agents_cur_pos[agent])/agents_init_pos[agent]
            elif self.reward_type == 'end_prop_cooperative':
                # only reward when termination/truncation
                # WARN: NO TERMINATION CONDITION
                if truncated:
                    global_reward += np.mean([
                        (agents_init_pos[i] - agents_cur_pos[i])/agents_init_pos[i]
                        for i in self.agents
                    ])*100  # normalise to 100        
            elif self.reward_type == 'end_prop_centinel':
                # only reward when termination/truncation
                # WARN: NO TERMINATION CONDITION
                if truncated:
                    for i in self.agents:
                        self.rewards[i] += 100*(agents_init_pos[i] - agents_cur_pos[i])/agents_init_pos[i]
            elif self.reward_type == 'equisplit_centinel':
                x_cur_err = abs(global_target[0] - agents_cur_pos[agent][0])
                y_cur_err = abs(global_target[1] - agents_cur_pos[agent][1])
                
                x_init_err = abs(global_target[0] - agents_init_pos[agent][0])
                y_init_err = abs(global_target[1] - agents_init_pos[agent][1])
                
                x_perc_error = (x_init_err - x_cur_err)/x_init_err
                y_perc_error = (y_init_err - y_cur_err)/y_init_err
                
                if -x_perc_error > 100:
                    x_normalised_error = 0.0
                else:
                    x_normalised_error = 1/(1 + np.exp(-x_perc_error))
                
                if -y_perc_error > 100:
                    y_normalised_error = 0.0
                else:
                    y_normalised_error = 1/(1 + np.exp(-y_perc_error))
                
                self.rewards[agent] += 0.5*x_normalised_error + 0.5*y_normalised_error
            else:
                assert False, 'really?'
            
            self.step_count += 1
        elif self.steps_beyond_terminated is None:
            # TODO: termination condition not implemented yet
            self.steps_beyond_terminated = 0
            # reward = self.termination_reward
            # assert False, 'set termination reward?'
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
            # purely cooperative, so same rewards for all
            self.rewards[i] += global_reward
            self.terminations[i] = terminated
            self.truncations[i] = truncated
            self.infos[i] = {}
        
        # self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()
        # self._clear_rewards()
        # TODO: above comments should be in raw_env?

        if self.render_mode == "human":
            self.render()
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_length, self.screen_length)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_length, self.screen_length))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        scale = lambda length: length*self.screen_length/2.0
        unnormalise = lambda coords: (coords + self.pos_max)*self.screen_length/2.0
        
        agent_radius = scale(0.015)
        target_radius = scale(0.010)
        # print(f'agent radius: {agent_radius}, target radius: {target_radius}')
        
        if self.finder_agents[self.agents[0]].state is None:
            return None
        
        self.surf = pygame.Surface((self.screen_length, self.screen_length))
        self.surf.fill((255, 255, 255))
        
        for i in self.agents:
            agent_x, agent_y = unnormalise(self.finder_agents[i].get_pos())
            # print(f'{i}: {agent_x}, {agent_y}')
            gfxdraw.aacircle(
                self.surf,
                int(agent_x),
                int(agent_y),
                int(agent_radius),
                self.agent_colours[i],
            )
            gfxdraw.filled_circle(
                self.surf,
                int(agent_x),
                int(agent_y),
                int(agent_radius),
                self.agent_colours[i],
            )
        
        target_x, target_y = unnormalise(self.target_manager.global_target)
        gfxdraw.aacircle(
            self.surf,
            int(target_x),
            int(target_y),
            int(target_radius),
            self.target_colour,
        )
        gfxdraw.filled_circle(
            self.surf,
            int(target_x),
            int(target_y),
            int(target_radius),
            self.target_colour,
        )

        gfxdraw.hline(
            self.surf, 0,
            int(self.screen_length),
            int(self.screen_length/2),
            (0, 0, 0)
        )
        gfxdraw.vline(
            self.surf, int(self.screen_length/2),
            0,
            int(self.screen_length),
            (0, 0, 0)
        )
        # gfxdraw.vline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    def __init__(self, 
                 gym_attrs: Dict[str, Any], 
                 render_mode: Optional[str] = None,
                 test_reward: Optional[bool] = False,
    ):
        super().__init__()
        
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "name": "cooperative_cartpole_v0",
            "is_parallelizable": False,     # TODO: make this true!
            "render_fps": 50,
        }
        self.test_reward = test_reward
        self.seed()     # TODO: this seed stuff may cause issues
                        # Assuming seed is externally set
        
        self.gym_attrs = gym_attrs
        self.render_mode = render_mode
        self.set_env()
        
        self.agents = self.env.agents.copy()
        self.possible_agents = self.agents.copy()
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
        # self._reset_cumulative_rewards()
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

        # ################################################## self._reset_cumulative_rewards()
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
    
    # def _reset_cumulative_rewards(self):
    #     self._cumulative_rewards = {i: 0 for i in self.agents}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        # TODO: print this seed and see if it is different in VecEnvs
    
    def set_env(self):
        self.env = Bunch(
            self.np_random, self.gym_attrs, self.metadata, self.render_mode,
            self.test_reward
        )

import math

import gymnasium
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.classic_control import utils
from gymnasium.utils import seeding

import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from typing import Optional, Any, Dict


class CooperativeCartPole:
    """
    ## Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.
    ## Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.
    - 0: Push cart to the left
    - 1: Push cart to the right
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ## Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ## Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.
    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ## Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    ## Arguments
    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```
    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """
    
    
    ###################
    ################
    ###########
    #######
    # TODO: make the discrete action space continuous?
    # TODO: reset must have Gaussian not uniform noise
    # TODO: truncation needs to be implemented manually I think
    

    def __init__(self, np_random, metadata: Dict[str, Any], render_mode: Optional[str] = None):
        self.np_random = np_random
        self.metadata = metadata
        
        self.num_agents = 2
        self.agents = ["weak_controller", "strong_controller"]
        ######## TODO: wtf is this below?
        self.possible_agents = self.agents[:]
        
        self.min_action = -1.0
        self.max_action = 1.0
        
        self.gravity = 9.8      # TODO: was 9.8 IMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.tau = 0.02  # seconds between state updates
        
        self.force_mag = 15.0       # TODO: used to be 10.0
        self.k = 2.0    # Witsenhausen parameter
        
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 24 * 2 * math.pi / 360       # TODO: IMP was 12
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        
        self.action_spaces = {
            i: spaces.Box(
                low=self.min_action, high=self.max_action,
                shape=(1,), dtype=np.float32,
               )
            for i in self.agents
        }
        self.observation_spaces = {
            i: spaces.Dict(
                {"observation": spaces.Box(-high, high, dtype=np.float32)}
               )
            for i in self.agents
        }
        
        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}   # TODO: why does this exist

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
        
        self.step_count = 0
        self.max_steps = 200
        
        # print(self.state, 'init')

    def step(self, action, agent):
        assert self.state is not None, "Call reset before using step method."
        assert agent in self.agents, "Invalid agent selected"
        
        # print(self.state, 'step')
        
        x, x_dot, theta, theta_dot = self.state
        # force = self.force_mag if action == 1 else -self.force_mag
        force = self.force_mag*min(max(action[0], self.min_action), self.max_action)
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

        if self.kinematics_integrator == "euler":
            dx = self.tau * x_dot
            x = x + dx
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == "semi-implicit euler":  # semi-implicit euler
            dx = self.tau * x_dot
            x = x + dx
            x_dot = x_dot + self.tau * xacc
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        else:
            assert False, "pick a valid integrator idiot"

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        truncated = False
        
        reward = 0.0

        if not terminated:
            if agent == "weak_controller":
                reward = -x**2 - abs((self.k**2)*force*dx)      # TODO: should this be abs?
            elif agent == 'strong_controller':
                reward = -x**2 - abs((1000.0)*force*dx)
                # reward = -x**2 - abs((self.k**2)*force*dx)
            
            self.step_count += 1
            
            if self.step_count >= self.max_steps:
                truncated = True
            
                # TODO: the above line is only for debugging
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -1e6
            # reward = 0.0        # TODO: delete this
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            # reward = 0.0
        
        for i in self.agents:
            # purely cooperative, so same rewards for all
            self.rewards[i] += reward
            self.terminations[i] = terminated
            self.truncations[i] = truncated
            self.infos[i] = {}
        
        # self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()
        # self._clear_rewards()
        # TODO: above comments should be in raw_env

        if self.render_mode == "human":
            self.render()
        # return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        
        # TODO: reset seed in the other class
        
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        epsilon = 0.025
        # self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.state = self.np_random.normal(
            loc=[0.0, 0.0, 0.0, 0.0],
            scale=[epsilon, epsilon, self.theta_threshold_radians/4, epsilon]
        )
        self.steps_beyond_terminated = None
        
        # print(low)
        # print('+')
        # print(high)
        # print('-')
        # epsilon = 0.025
        # print(self.theta_threshold_radians/4)
        # print(self.np_random.normal(
        #     loc=[0.0, 0.0, 0.0, 0.0],
        #     scale=[epsilon, epsilon, self.theta_threshold_radians/4, epsilon]
        # ))
        # print(options)
        # pleb
        
        # print(self.state, 'reset')
        
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}   # TODO: why does this exist
        
        self.step_count = 0

        if self.render_mode == "human":
            self.render()
        # return np.array(self.state, dtype=np.float32), {}
    
    def observe(self, agent):
        # TODO: might be source of an error
        # TODO: argument agent is not used
        # return {"observation": np.array(self.state, dtype=np.float32)}
        # TODO: note IMP change it should be [self.state] or self.state
        return np.array(self.state, dtype=np.float32)

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
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

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
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "name": "cooperative_cartpole_v0",
            "is_parallelizable": False,     # TODO: make this true!
            "render_fps": 50,
        }   # TODO: is this fine or should it be outside the function?
        
        self.seed()     # TODO: this seed stuff may cause issues
                        # Assuming seed is externally set
        self.render_mode = render_mode
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
        # print('obs', obs)
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
        self._reset_cumulative_rewards()
        
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

        self._reset_cumulative_rewards()
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
    
    def _reset_cumulative_rewards(self):
        self._cumulative_rewards = {i: 0 for i in self.agents}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        # TODO: print this seed and see if it is different in VecEnvs
    
    def set_env(self):
        self.env = CooperativeCartPole(
            self.np_random, self.metadata, self.render_mode)
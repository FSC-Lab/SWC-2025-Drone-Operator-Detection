# noqa: D212, D415

# SEARCH ENVIRONMENT
# Describe Environment Model

import numpy as np
import copy
from gymnasium.utils import EzPickle

from Environment.core import Agent, Landmark, World
from Environment.simple_env import SimpleEnv, make_env

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn

# CONFIGURATION


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        max_cycles=25, 
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            # dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "search_env_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

# ENVIRONMENT PARAMETERS
# Environment Setup
N = 3  # Num Searchers
Ob = 0  # Num Obstacles
T = 1  # Num Targets

scale = 5000 # Distance Scale Factor
r_b = 315 / scale  # Target Size Radius  (for rendering)
r_o = 0.05 / scale  # Obstacle Size Radius

# Searcher
r_d = 50 / scale   # Searcher Size Radius (for rendering)
r_in_min = 1000    # Initial Minimum Distance
r_in_max = 4500    # Initial Maximum Distance


def generate_positions(N, r_in_min, r_in_max):
    init_pos = []

    for _ in range(N):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(r_in_min, r_in_max)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        init_pos.append(np.array([x, y]))

    return init_pos

init_pos = generate_positions(N, r_in_min, r_in_max)

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # Initialize Environment
        world.dim_c = 2
        num_searchers = N
        num_obstacles = Ob
        num_targets = T

        # Add Searchers
        world.searchers = [Agent() for _ in range(num_searchers)]
        for i, searcher in enumerate(world.searchers):
            searcher.searcher = True
            faction = "Searcher"
            searcher.name = f"{faction}_{i+1}"
            searcher.collide = False
            searcher.silent = True
            searcher.size = r_d
            searcher.battery = 0
            searcher.max_battery = 60000
            searcher.p_pos = init_pos[i]
            # Define max acceleration, initial position and other parameters

        # Define Agents
        world.agents += world.searchers

        # Add Landmarks (Obstacles & Targets)
        world.obstacles = [Landmark() for _ in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = "Obstacle %d" % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = r_o
        world.targets = [Landmark() for _ in range(num_targets)]
        for i, target in enumerate(world.targets):
            target.name = "Target %d" % i
            target.collide = False
            target.movable = False
            target.size = r_b
        world.landmarks += world.obstacles
        world.landmarks += world.targets
        return world
    
    # ENVIRONMENT INITIALIZATION

    def reset_world(self, world, np_random):
        # Set Obstacles
        for obstacle in world.obstacles:
            obstacle.color = np.array([0.64, 0.16, 0.16])
            obstacle.state.p_pos = np.zeros(world.dim_p) # Obstacle Position Generator
            obstacle.state.p_vel = np.zeros(world.dim_p)

        # Set Targets (#Assume one target)
        for i, target in enumerate(world.targets):
            target.color = np.array([0.1, 0.1, 0.75])
            target.state.p_pos = np.zeros(world.dim_p) # Target at origin
            target.state.p_vel = np.zeros(world.dim_p)

        # Set Searcher Locations and Targets
            init_pos = generate_positions(N, r_in_min, r_in_max)     # Searcher Position Generator
            for i, searcher in enumerate(world.searchers):
                searcher.state.p_pos = init_pos[i]
                searcher.state.p_vel = np.zeros(world.dim_p)
                searcher.target = np_random.choice(world.targets)     # Since one target, always chooses it. 


        # Define Agent Colours
        for agent in world.agents:
            agent.color = np.array([0.25, 0.75, 0.25])
            
    # AGENT INTERACTION HELPER FUNCTIONS

    # Standard distance functions

    def target_dist(self, agent):
        return np.sqrt(np.sum(np.square(agent.state.p_pos - agent.target.state.p_pos)))   # Target selector has to be defined

    # Add additional functions as needed to compute environment variables

    def power(self, agent):
        # Drone Parameters
        rho = 1.23  # Density
        s = 0.2     # Area
        C_D = 1     # Drag Coefficient
        w = 19.6    # Weight

        # Propeller Parameters
        k_t = 1.8155 * (10 ** -5)
        k_tau = 2.6212 * (10 ** -7)

        # Motor Parameters
        r_a = 0.028   # Resistance
        eta_m = 0.9   # Efficiency
        k_mt = 0.0048 
        k_e = 0.0048

        v = np.sqrt(np.sum(np.square(agent.state.p_vel)))  # Drone Velocity

        x = 0.25 * (rho ** 2) * (v ** 4) * (s ** 2) * (C_D ** 2) + (w ** 2)
        omega_s = np.sqrt(np.sqrt(x) / (4 * k_t))
        tau_p = k_tau * (omega_s ** 2)
        y = tau_p / (eta_m * k_mt)

        return y * ( (r_a * y) + (k_e * omega_s) )

    def D_optimality(self, world):
        sig = 1  # Noise SD                
        a_0 = 1

        I = np.zeros((2,2))

        for searcher in world.searchers:
            dist = self.target_dist(searcher)
            dist_vec = searcher.goal.state.p_pos - searcher.state.p_pos
            if dist != 0:
                I += np.outer(dist_vec, dist_vec) / (dist ** 4)
            else:
                I += np.zeros((2,2))
        
        I *= ((a_0 / sig) ** 2)
        return np.linalg.det(I)

    # REWARD FUNCTIONS

    def reward(self, agent, world):
        if agent.searcher:
            return self.searcher_reward(agent, world)  # Only oee class of agent, can extend to add more
        
    def searcher_reward(self, agent, world):
        # NOTE: If the I matrix provides sufficient information at each timestep such that there
        # is a gradient of value improvement to the target, the reward as defined is non-sparse.
        # However, if the gradient of det(I) is low at initial state, reward function should be 
        # changed to encourage intermediate improvement to speed up learning. 

        # Condition Parameters
        S_1 = self.D_optimality(world)
        # S_1 = np.log10(self.D_optimality(world))   # log of det(I)
        S_2 = -5
        S_3 = 1
        eta = 500

        agent.battery += (4 * self.power(agent) * world.dt)

        if self.target_dist(agent) < eta:
            safe_dist_pen = S_3 - (self.target_dist(agent) / scale)
        else:
            safe_dist_pen = 0
        
        if any(searcher.battery == searcher.max_battery for searcher in world.searchers): # Implement battery terminal condition
            return S_2 / N
            
        return (S_1 - safe_dist_pen)/ N

    # OBSERVATION SPACE
    
    def observation(self, agent, world):
        # Observation Space
        entity_pos = []
        comm = []
        other_pos = []
        other_vel = []

        # Get Exact Position of Static Landmarks (Target, Obstacles) - Modify if there is search model
        for entity in world.landmarks:  
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # Get Locations and Velocities of other Searchers and Targets - Modify depending on tracking dimension
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append((other.state.p_pos - agent.state.p_pos) / scale)
            other_vel.append(other.state.p_vel)
        if agent.searcher:
            return np.concatenate(
                [(agent.state.p_pos / scale)]
                + [(agent.state.p_vel / 1000)]
                + [((agent.target.state.p_pos - agent.state.p_pos) / scale)]
                + other_pos
            )
import numpy as np
import time
import yaml

import gymnasium as gym

from dataclasses import asdict

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, SineLane

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle

from configs.configs import EnvArgs

class MergeExitLaneHighway_Environment(AbstractEnv):
    """
    A customized multi-lane highway environment with an on-ramp and and an exit ramp.

    ## Action Space
    
        MultiAgentAction: For n-total vehicles, an 'ndarray' with shape (n, 2) representing the acceleration and steering controls.
        (n vehicles, each with acceleration and steering controls in the continuous space)
        
        Min and max values for the acceleration and steering are detailed in the parameters file.

    ## Observation Space

        MultiAgentObservation: For n-total vehicles, an 'ndarray' with shape (n, n, 6).
        (n vehicles, each observing the 6 variables of n vehicles, ordered according to proximity)

        (1) Presence
        (2) x value
        (3) y value
        (4) vx value
        (5) vy value
        (6) heading angle

    ## Rewards

        The ultimate goal is for the ego vehicle to traverse the highway SUCCESSFULLY from the on-ramp until the exit ramp.
        The goal of the RL training is to build an adversarial platoon out of the remaining vehicles that tries to spike the 
        metrics for crash risk (low TTC and high DRAC).

        The reward consists of two parts:

        (1) Overall reward
            Reward for platoon with ego success
            Negative reward for low cumulative risk
            Negative reward for platoon crash
            Negative reward for ego crash

        (2) Step-wise reward
            Incremental reward for risky actions (dynamically changing based on closeness to the exit ramp)
            incremental negative reward for non-risk actions
            Negative reward for platoon crash

    ## Starting State

        The ego vehicle always begins on the on-ramp with a starting longitudinal velocity of 25 m/s.
        The ego vehicle is controlled using IDM for low-level and MOBIL for high-level planning.
        
        The adversarial platoon is arranged such that there are two vehicle pairs per lane. Starting velocities are randomized.
        The ego vehicle is controlled using __ for low-level and MASAC-RL for high-level planning.

    ## Episode Truncation

        The episode truncates when ego crashes, when ego reaches the goal, or after 60 seconds.

    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()

        env_params = EnvArgs()

        config.update(asdict(env_params))

        return config


    def _make_road(self) -> None:

        """

        Make a road composed of a straight n-lane highway with a merge ramp and an exit ramp

        Composed of 7 longitudinal sections:
        1. Before
        2. Converging
        3. Merge ramp connection
        4. Mid
        5. Exit ramp connection
        6. Diverging
        7. After

        """

        lane_width_m = self.config["lane_width_m"]
        lanes_count = self.config["lanes_count"]
        ends_m = self.config["ends_m"] # Before, converging, merge, mid, exit, diverging, after

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # === HIGHWAY LANES ===

        # Initializing the line types for each lane (one per differing section)
        line_type = [[c, s]] + [[n, s]] * (lanes_count-2) + [[n, c]]
        line_type_merge = [[c, s]] + [[n, s]] * (lanes_count-2) + [[n, s]]
        line_type_exit = [[s, s]] + [[n, s]] * (lanes_count-2) + [[n, c]]

        net = RoadNetwork()

        y = list(range(0, int(lanes_count * lane_width_m + 1), int(lane_width_m)))

        for i in range(lanes_count):
            net.add_lane( # Before + Converging
                "a",
                "b",
                StraightLane(
                    [0, y[i]],
                    [sum(ends_m[:2]), y[i]],
                    line_types=line_type[i],
                ),
            )
            net.add_lane( # Merge ramp connection
                "b",
                "c",
                StraightLane(
                    [sum(ends_m[:2]), y[i]],
                    [sum(ends_m[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane( # Mid
                "c",
                "d",
                StraightLane(
                    [sum(ends_m[:3]), y[i]],
                    [sum(ends_m[:4]), y[i]],
                    line_types=line_type[i],
                ),
            )
            net.add_lane( # Exit ramp connection
                "d",
                "e",
                StraightLane(
                    [sum(ends_m[:4]), y[i]],
                    [sum(ends_m[:5]), y[i]],
                    line_types=line_type_exit[i],
                ),
            )
            net.add_lane( # Diverging + After
                "e",
                "f",
                StraightLane(
                    [sum(ends_m[:5]), y[i]],
                    [sum(ends_m), y[i]],
                    line_types=line_type[i],
                ),
            )


        # MERGING LANE (modeling the curve using sine wave)
        amplitude = self.config["merge_amplitude"]

        self.mergeramp_startlat = amplitude*2 + lane_width_m*lanes_count

        self.merging_jk = StraightLane( # Before
            [0, self.mergeramp_startlat],
            [ends_m[0], self.mergeramp_startlat],
            line_types=[c,c],
            forbidden=True
        )
        self.merging_kb = SineLane( # Converging
            self.merging_jk.position(ends_m[0], -amplitude),
            self.merging_jk.position(sum(ends_m[:2]), -amplitude),
            amplitude, # amplitude
            2 * np.pi / (2 * ends_m[1]), # pulsation
            np.pi / 2, # phase
            line_types=[c, c],
            forbidden=True,
        )
        self.merging_bc = StraightLane( # Merge ramp connection
            self.merging_kb.position(ends_m[1], 0),
            self.merging_kb.position(ends_m[1], 0) + [ends_m[2], 0],
            line_types=[n,c],
            forbidden=True,
        )


        net.add_lane("j", "k", self.merging_jk)
        net.add_lane("k", "b", self.merging_kb)
        net.add_lane("b", "c", self.merging_bc)

        # EXIT LANE (modeling the curve using sine wave)

        self.merging_de = StraightLane(
            [sum(ends_m[:4]), -lane_width_m],
            [sum(ends_m[:5]), -lane_width_m],
            line_types=[c,n],
            forbidden=True,
        )

        # exit_ref = StraightLane(
        #     [sum(ends_m[:4]), -lane_width_m - amplitude],
        #     [sum(ends_m[:5]), -lane_width_m - amplitude],
        # )

        self.merging_el = SineLane(
            self.merging_de.position(ends_m[4], 0) + [0, -amplitude],
            self.merging_de.position(sum(ends_m[4:6]), 0) + [0, -amplitude],
            amplitude,
            2 * np.pi / (2 * ends_m[5]),
            np.pi / 2,
            line_types=[c,c],
            forbidden=True,
        )

        self.merging_lm = StraightLane(
            self.merging_el.position(ends_m[5], 0),
            self.merging_el.position(ends_m[5], 0) + [ends_m[6], 0],
            line_types=[c, c],
            forbidden=True,
        )

        net.add_lane("d", "e", self.merging_de)
        net.add_lane("e", "l", self.merging_el)
        net.add_lane("l", "m", self.merging_lm)

        self.road = Road(
            network = net,
            np_random = self.np_random,
            record_history = self.config["show_trajectories"],
        )

    def _make_stock_vehicles(self) -> None:

        self.controlled_vehicles = []

        self.ego_lane = self.road.network.get_lane(("j", "k", 0))

        self.ego = IDMVehicle(
            self.road,
            self.ego_lane.position(0, 0),
            speed = 25,
            route = [("j", "k", 0),("k","b", 0),("b","c",0),("c","d",0),("d","e",0),("e","l",0),("l","m",0)]
        )

        self.road.vehicles.append(self.ego)

        lanes_count = self.config["lanes_count"]

        for lane_idx in range(lanes_count):
            highway_lane = self.road.network.get_lane(("a", "b", lane_idx))
            for car_idx in range(2):
                longitudinal_pos_m = 40 - (car_idx * 20)

                adv = MDPVehicle(
                    self.road,
                    highway_lane.position(longitudinal_pos_m, 0),
                    speed = np.random.uniform(20,30)
                )
                self.road.vehicles.append(adv)
                self.controlled_vehicles.append(adv)

        # print(self.controlled_vehicles)


    def _reset(self) -> None:
        self._make_road()
        self._make_stock_vehicles()

    def _reward(self, action: int) -> float:
        reward = 0.0

        ego = self.ego
        adversaries = self.controlled_vehicles

        # if ego.position[0] > sum(self.config['ends_m'][:3]):
        #     reward -= abs(ego.lane_index[2] - 4) * 25.0

        # MINIMIZE TTC
        for adv in adversaries:
            poly_ttc = PolygonTTCCalculator.compute_ttc(ego, adv)

            if 1.0 <= poly_ttc <= 4.0: # dangerously close
                reward += 4.0 / (poly_ttc + 0.1)
            elif 0.0 <= poly_ttc < 1.0: # crash soon
                reward -= 10.0

        # REACHED GOAL
        if ego.lane_index[0] == 'l' and ego.lane_index[1] == 'm':
            if not ego.crashed:
                reward += 50.0

        # CRASHES
        if ego.crashed:
            reward -= 100.0

        for adv in adversaries:
            if adv.crashed:
                reward -= 50.0

        return float(reward)

    def _is_terminated(self) -> bool:
        """Tells the engine to stop if the car crashes."""
        return self.ego.crashed

    def _is_truncated(self) -> bool:
        """Tells the engine to stop if the time limit is reached."""
        return self.time >= self.config["duration"]
    
    def step(self, split_actions):
        self.action_type.act(split_actions)

        self._simulate()

        obs = self.observation_type.observe()

        reward = self._reward(split_actions)

        terminated = bool(self.ego.crashed)

        truncated = False

        if self.ego.lane_index[0] == 'l' and self.ego.lane_index[1] == 'm':
            if not self.ego.crashed:
                truncated = True
        
        if self.steps >= self.config['duration']:
            truncated = True

        return obs, reward, terminated, truncated, {}

class Wrapper_MergeExitLaneHighway_Environment(gym.Wrapper):
    def __init__(self, env):

        """
        Wrapper takes the multi-agent observation space of the original environment, and takes the necessary
         observations for both and processes it to be arranged in one dimension.

        Observation will be 
        """

        super().__init__(env)

        self.adv_agents = self.env.unwrapped.config.get(
            'controlled_vehicles', 10
        )

        self.num_features = len(
            self.env.config.get('observation', {}).get('observation_config', {}).get('features', [1,2,3,4,5,6])
        )

    def observation(self, obs):

        """
        Intersects the raw environment output and transforms it into a 
        clean, zero-redundancy matrix of shape (n, 12).
        """

        # print(np.stack(obs).shape)
        
        self.unwrapped_env = self.env.unwrapped
        self.victim = None

        self.target_long_dist = self.unwrapped_env.config['ends_m'][4]
        self.target_lat_dist = -self.unwrapped_env.config['lane_width_m']
        
        # The victim is the vehicle on the road that is NOT in our controlled platoon list
        for vehicle in self.unwrapped_env.road.vehicles:
            if vehicle not in self.unwrapped_env.controlled_vehicles:
                victim = vehicle
                break
                
        # Instantiation
        processed_obs = np.zeros((self.adv_agents, 12), dtype=np.float32)
        
        # 3. Loop through each of your 10 platoon agents to build their unique views
        for i in range(self.adv_agents):
            ego_raw = obs[i][0]
            
            # Extract Ego absolute features (assuming standard highway-env kinematics ordering)
            ego_presence = ego_raw[0]
            ego_x        = ego_raw[1]
            ego_y        = ego_raw[2]
            ego_vx       = ego_raw[3]
            ego_vy       = ego_raw[4]
            ego_heading  = ego_raw[5]
            
            # 4. Calculate relative states if the victim is alive and active
            if victim is not None and ego_presence > 0:
                target_presence = 1.0
                # Relative calculations: Target minus Ego
                rel_long_dist   = victim.position[0] - self.unwrapped_env.controlled_vehicles[i].position[0]
                rel_lat_dist    = victim.position[1] - self.unwrapped_env.controlled_vehicles[i].position[1]
                rel_vel_long    = victim.speed - self.unwrapped_env.controlled_vehicles[i].speed
                rel_vel_lat     = 0.0 - ego_vy  # Assuming victim has negligible lateral velocity
                rel_heading     = victim.heading - ego_heading
            else:
                # Fallback values if the victim crashes out or the agent is inactive
                target_presence = 0.0
                rel_long_dist   = 0.0
                rel_lat_dist    = 0.0
                rel_vel_long    = 0.0
                rel_vel_lat     = 0.0
                rel_heading     = 0.0

            processed_obs[i] = [
                ego_presence,
                self.target_long_dist - ego_x,
                self.target_lat_dist - ego_y,
                ego_vx,
                ego_vy,
                ego_heading,
                target_presence,
                rel_long_dist,
                rel_lat_dist,
                rel_vel_long,
                rel_vel_lat,
                rel_heading
            ]
            
        return processed_obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # print(np.stack(obs).shape)
        # print(self.env.unwrapped.config['observation']['type'])
        return self.observation(obs), info
    
    def step(self, action):
        
        """
        Expects action to be a NumPy matrix of shape (adv_agents, 2).
        Converts it directly to the environment's tuple-of-tuples format without splitting.
        """
        
        formatted_actions = tuple(tuple(action[i]) for i in range(self.adv_agents))
        
        # Step the underlying simulation engine
        _obs, reward, terminated, truncated, info = self.env.step(formatted_actions)

        # Collapse individual vehicle rewards into your global training scalar
        if isinstance(reward, (list, tuple, np.ndarray)):
            collective_reward = float(np.mean(reward))
        elif isinstance(reward, dict):
            collective_reward = float(np.mean(list(reward.values())))
        else:
            collective_reward = float(reward)
        
        # Process the raw dict output straight into your clean 10x12 state matrix
        return self.observation(_obs), collective_reward, terminated, truncated, info
    

class PolygonTTCCalculator:
    """
    A high-performance NumPy utility class to compute the precise Time-to-Collision (TTC)
    between two oriented rectangular vehicle bounding boxes.
    """
    
    @staticmethod
    def _line(p0, p1):
        a = p0[1] - p1[1]
        b = p1[0] - p0[0]
        c = p0[0] * p1[1] - p1[0] * p0[1]
        return a, b, c

    @staticmethod
    def _intersect(line0, line1):
        a0, b0, c0 = line0
        a1, b1, c1 = line1
        D = a0 * b1 - a1 * b0
        if abs(D) < 1e-5:
            return np.array([np.nan, np.nan])
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        return np.array([x, y])

    @staticmethod
    def _ison(line_start, line_end, point):
        if np.isnan(point[0]):
            return False
        crossproduct = (point[1] - line_start[1]) * (line_end[0] - line_start[0]) - (point[0] - line_start[0]) * (line_end[1] - line_start[1])
        if abs(crossproduct) > 1e-5:
            return False
        dotproduct = (point[0] - line_start[0]) * (line_end[0] - line_start[0]) + (point[1] - line_start[1]) * (line_end[1] - line_start[1])
        squaredlength = (line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2
        return (dotproduct >= 0) and (dotproduct <= squaredlength)

    @classmethod
    def get_bounding_box_corners(cls, x, y, heading, length, width):
        """Calculates the 4 absolute corner points of a vehicle."""
        h_vec = np.array([np.cos(heading), np.sin(heading)])
        perp_h_vec = np.array([-h_vec[1], h_vec[0]])
        
        point_up = np.array([x, y]) + h_vec * (length / 2.0)
        point_down = np.array([x, y]) - h_vec * (length / 2.0)
        
        return [
            point_up + perp_h_vec * (width / 2.0),
            point_up - perp_h_vec * (width / 2.0),
            point_down + perp_h_vec * (width / 2.0),
            point_down - perp_h_vec * (width / 2.0)
        ]

    @classmethod
    def _pairwise_ttc(cls, params_i, params_j):
        """Computes directional ray-cast TTC from Pack I to Pack J."""
        corners_i = cls.get_bounding_box_corners(params_i[0], params_i[1], params_i[4], params_i[5], params_i[6])
        corners_j = cls.get_bounding_box_corners(params_j[0], params_j[1], params_j[4], params_j[5], params_j[6])
        
        v_i = np.array([params_i[2], params_i[3]])
        v_j = np.array([params_j[2], params_j[3]])
        direct_v = v_i - v_j
        
        rel_speed = np.linalg.norm(direct_v)
        if rel_speed < 1e-5:
            return np.inf

        min_dist_ist = np.inf
        valid_collision_course = False
        edges_j = [(corners_j[0], corners_j[1]), (corners_j[2], corners_j[3]), 
                   (corners_j[0], corners_j[2]), (corners_j[1], corners_j[3])]

        for p_start in corners_i:
            p_end = p_start + direct_v
            ray_line = cls._line(p_start, p_end)
            
            for edge_start, edge_end in edges_j:
                edge_line = cls._line(edge_start, edge_end)
                ist = cls._intersect(ray_line, edge_line)
                
                if cls._ison(edge_start, edge_end, ist):
                    leaving_check = direct_v[0] * (ist[0] - p_start[0]) + direct_v[1] * (ist[1] - p_start[1])
                    if leaving_check >= 0:
                        valid_collision_course = True
                        dist_ist = np.linalg.norm(ist - p_start)
                        if dist_ist < min_dist_ist:
                            min_dist_ist = dist_ist

        if not valid_collision_course or min_dist_ist == np.inf:
            return np.inf
            
        return min_dist_ist / rel_speed

    @classmethod
    def compute_ttc(cls, veh_i, veh_j) -> float:
        """
        Public API: Call this to get the symmetrical polygon TTC between two highway-env vehicles.
        """
        # Format: (x, y, vx, vy, heading, length, width)
        # Safely handle environments that use velocity vector arrays vs raw scalar speeds
        v_i = getattr(veh_i, 'velocity', np.array([veh_i.speed * np.cos(veh_i.heading), veh_i.speed * np.sin(veh_i.heading)]))
        v_j = getattr(veh_j, 'velocity', np.array([veh_j.speed * np.cos(veh_j.heading), veh_j.speed * np.sin(veh_j.heading)]))

        params_i = (veh_i.position[0], veh_i.position[1], v_i[0], v_i[1], veh_i.heading, veh_i.LENGTH, veh_i.WIDTH)
        params_j = (veh_j.position[0], veh_j.position[1], v_j[0], v_j[1], veh_j.heading, veh_j.LENGTH, veh_j.WIDTH)
        
        return min(cls._pairwise_ttc(params_i, params_j), cls._pairwise_ttc(params_j, params_i))

# env = MergeExitLaneHighway_Environment()
# env.render_mode = "human" # Tells Gymnasium to prepare a visual window
# env.reset()

# # Run a loop to step the physics engine forward in time
# for _ in range(200):
#     # Action '1' is the IDLE action (tells the car to just cruise forward)
#     # If your gym version is older, you might only get 4 return values instead of 5
#     obs, reward, done, truncated, info = env.step(1)

#     env.render()

#     time.sleep(0.2) # <--- Add this to slow down the frame rate

#     # If the car crashes or finishes the route, reset the map
#     if done or truncated:
#         env.reset()
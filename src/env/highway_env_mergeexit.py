import numpy as np
import gymnasium as gym

from dataclasses import asdict

from highway_env.envs.common.abstract import AbstractEnv, Vehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, SineLane

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle

from src.env.risk_calculators import PolygonTTCCalculator

from configs.configs import EnvArgs, RLArgs

class MergeExitLaneHighway_Environment(AbstractEnv):
    """
    A customized multi-lane highway environment with an on-ramp and and an exit ramp.

    ## Action Space
    
        MultiAgentAction: For n-total vehicles, an 'ndarray' with shape (n, 2) representing the acceleration and steering controls for each.
        
        Min and max values for acceleration and steering imposed and detailed in the parameters file.

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
        Each is controlled using MDP for low-level and MASAC-RL for high-level planning.

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
        1. Before (j-k)
        2. Converging (k-b)
        3. Merge ramp connection (b-c)
        4. Mid (c-d)
        5. Exit ramp connection (d-e)
        6. Diverging (e-l)
        7. After (l-m)

        The road network is constructed as follows:

                                         /--(l)---(m)
        (a)---------(b)---(c)---(d)---(e)---------(f)
        (j)---(k)--/

        Each of the segments is a straight lane (or set of lanes in the case of a through f), except for k-b and e-l which are sine lanes to model the curves of the merge and exit ramps.

        To adhere to highway environment library's indexing scheme, the lanes are added to the road network from top to bottom.

        """

        # Configuring road geometry
        lane_width_m = self.config["lane_width_m"]
        lanes_count = self.config["lanes_count"]
        ends_m = self.config["ends_m"] # Before, converging, merge, mid, exit, diverging, after

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # Arrays of line types for accurate drawing of multi-lane highway with merge and exit ramps

        line_type = [[c, s]] + [[n, s]] * (lanes_count-2) + [[n, c]]
        line_type_merge = [[c, s]] + [[n, s]] * (lanes_count-2) + [[n, s]]
        line_type_exit = [[s, s]] + [[n, s]] * (lanes_count-2) + [[n, c]]

        # Constructing the road network

        net = RoadNetwork()

        amplitude = self.config["merge_amplitude"]

        # ==== EXIT LANE (e-l-m) ====

        self.merging_de = StraightLane(
            [sum(ends_m[:4]), -lane_width_m],
            [sum(ends_m[:5]), -lane_width_m],
            line_types=[c,n],
            forbidden=True,
        )

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

        # ==== STRAIGHT LANES (a-b-c-d-e-f) ====

        # Array containing y-coordinates for each lane in highway
        y = list(range(0, int(lanes_count * lane_width_m + 1), int(lane_width_m)))

        self.straight_lane_nodes = ["a", "b", "c", "d", "e", "f"]

        for i in range(lanes_count): # lane index

            # Defining the different straight lane segments
            
            setattr(self, f"straight_ab_{i}", StraightLane( # Before converging
                [0, y[i]],
                [sum(ends_m[:2]), y[i]],
                line_types=line_type[i],
            ))

            setattr(self, f"straight_bc_{i}", StraightLane( # Merge ramp connection
                [sum(ends_m[:2]), y[i]],
                [sum(ends_m[:3]), y[i]],
                line_types=line_type_merge[i],
            ))
            setattr(self, f"straight_cd_{i}", StraightLane( # Mid section
                [sum(ends_m[:3]), y[i]],
                [sum(ends_m[:4]), y[i]],
                line_types=line_type[i],
            ))
            setattr(self, f"straight_de_{i}", StraightLane( # Exit ramp connection
                [sum(ends_m[:4]), y[i]],
                [sum(ends_m[:5]), y[i]],
                line_types=line_type_exit[i],
            ))
            setattr(self, f"straight_ef_{i}", StraightLane( # Diverging + After
                [sum(ends_m[:5]), y[i]],
                [sum(ends_m), y[i]],
                line_types=line_type[i],
            ))
            
            # Adding all the straight segments to the road network

            for j in range(len(self.straight_lane_nodes) - 1): # segment indexing
                net.add_lane(
                    self.straight_lane_nodes[j],
                    self.straight_lane_nodes[j + 1],
                    getattr(self, f"straight_{self.straight_lane_nodes[j]}{self.straight_lane_nodes[j + 1]}_{i}")
                )


        # ==== MERGING LANE (j-k-b-c) ====

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

        # Building the road object with the constructed network and random number generator

        self.road = Road(
            network = net,
            np_random = self.np_random,
            record_history = self.config["show_trajectories"],
        )

    def _get_current_lateral_boundaries(self, vehicle: Vehicle) -> None:
        
        """
        Returns the boundaries of the road for a given vehicle.
        The boundaries are defined as the left and right edges of the road.
        
        For use mainly in determining whether given vehicle is currently out of bounds (off the road) and should be crashed.
        
        """

        lane_width_m = self.config["lane_width_m"]

        current_segment = vehicle.lane_index # tuple of (start_node, end_node, lane_index)
        current_position = vehicle.position[0], vehicle.position[1]

        # Grabbing the lane indices for the outermost lanes of the vehicle at its current point
        
        if current_segment[0] in self.straight_lane_nodes and current_segment[1] in self.straight_lane_nodes:
            min_index = 0
            max_index = len(self.road.network.all_side_lanes(current_segment)) - 1
        else:
            min_index = 0
            max_index = 0

        # Calculating absolute position of lane boundaries

        left_boundary = self.road.network.get_lane((current_segment[0], current_segment[1], min_index)).position(longitudinal=current_position[0], lateral=0) + np.array([0, lane_width_m / 2])
        right_boundary = self.road.network.get_lane((current_segment[0], current_segment[1], max_index)).position(longitudinal=current_position[0], lateral=0) - np.array([0, lane_width_m / 2])
        
        return left_boundary, right_boundary
    
    def _is_out_of_bounds(self, vehicle: Vehicle) -> bool:

        """
        Checks if a vehicle is out of the road boundaries.
        A vehicle is considered out of bounds if it is outside the lateral boundaries of the road.
        
        """

        left_boundary, right_boundary = self._get_current_lateral_boundaries(vehicle)
        vehicle_position = vehicle.position[1] # only counting lateral position for out-of-bounds check
        
        # Assuming 0 heading, crash if vehicle borders touch or cross road boundaries
        return not (right_boundary[1] >= vehicle_position >= left_boundary[1])
    
    def _make_vehicles(self) -> None:

        self.controlled_vehicles = []

        # LANE CHANGER: Spawns on the on-ramp and is controlled by IDM + MOBIL. The goal is to reach the exit ramp successfully.

        self.ego_lane = self.road.network.get_lane(("j", "k", 0))

        self.ego = IDMVehicle(
            self.road,
            self.ego_lane.position(0, 0),
            speed = 25,
            route = [("j", "k", 0),("k","b", 0),("b","c",0),("c","d",0),("d","e",0),("e","l",0),("l","m",0)]
        )

        self.road.vehicles.append(self.ego)

        # ADVERSARIAL VEHICLES: 2 per lane, controlled by MDP + MASAC-RL. The goal is to maximize risk between ego and adversaries, while minimizing risk between adversaries.
        # Pairs each lane, one situated at 20m in and one at 40m in, with randomized starting velocities between 20 and 30 m/s.

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

    def _reset(self) -> None:

        # Remake the road and all the vehicles for clean slate

        self._make_road()
        self._make_vehicles()

    def _is_terminated(self) -> bool:
        """Tells the engine to stop if the car crashes."""
        return self.ego.crashed

    def _is_truncated(self) -> bool:
        """Tells the engine to stop if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _reward(self, action: int) -> float:
        
        # Actions already reverted to original highway-env action space

        """
        Reward is an array of n elements, one element pertaining to one adversary.
        Indexing is based on order in controlled vehicles list, based on order of spawn in _make_vehicles().

        Collective rewards

        Adversaries must:
        (1) Reduce TTC with ego
        (2) Increase TTC with other adversaries to avoid crashing into each other
        (3) Avoid crashing into ego or other adversaries (higher penalty for ego crash)
        (4) Occupy space surrounding the ego vehicle to make it difficult for ego to switch lanes or exit the highway until last second
        (5) Perform a sudden change in behavior to allow ego to exit successfully, if possible
        (6) Bunch up like a gate (especially at the end) to create a platoon that is difficult for ego to navigate through


        The reward function was designed with the following questions in mind:

        (1) Can the adversarial platoon exhbit emely risky behavior while still avoiding crashing into each other and the ego vehicle?
        (2) Can the adversarial platoon exhibit a sudden change in behavior (from fighting during cruising to giving way to ego) to allow ego to reach the exit ramp successfully?
        (3) Can the adversarial platoon bring out the vulnerabilities of the ego vehicle's IDM + MOBIL control system, and exploit them to maximize risk?
        
        """

        # ====== SETUP =======

        # Setting the reward arguments

        reward_args = self.config["reward"]

        # Setting important environment-related variables
        
        ego = self.ego
        adversaries = self.controlled_vehicles
        dist_to_exit = sum(self.config["ends_m"][:4]) - ego.position[0]

        # Initializing the rewards

        indiv_rewards = [0.0] * len(adversaries)
        team_reward = 0.0

        # Splitting bullying into two phases: (1) blocking the ego from reaching the exit ramp, and (2) allowing the ego to reach the exit ramp successfully

        is_release_phase = dist_to_exit < reward_args["release_distance"]

        # ====== LOCAL REWARDS: Rewarding each adversary =======

        for i, adv in enumerate(adversaries):

            adv_reward = 0.0

            # Drive safe!
            if adv.crashed and not self.config["adv_crash_penalization"][i]:
                adv_reward -= 55.0 # Penalty for bad driving leading to crash

            # Don't endanger other team mates!
            for j, other_adv in enumerate(adversaries):
                if i == j: continue
                adv_adv_ttc = PolygonTTCCalculator.compute_ttc(adv, other_adv)

                if 0.0 <= adv_adv_ttc < 1.0: # REALLY high risk of fratricide
                    adv_reward -= 15.0
                elif 1.0 <= adv_adv_ttc <= 4.0: # Need to back off!
                    adv_reward -= 8.0 / adv_adv_ttc # [2, 8]
                elif adv_adv_ttc > 4.0: # Okay, but don't stray too far!
                    adv_reward -= adv_adv_ttc / 1.5 # [1, inf]
            
            # Bully the ego!
            adv_ego_ttc = PolygonTTCCalculator.compute_ttc(adv, ego)

            if not is_release_phase: # Still trying to block on the highway
                if 0.0 <= adv_ego_ttc < 1.0: # Okay uhh, too much
                    adv_reward -= 30.0
                elif 1.0 <= adv_ego_ttc <= 4.0: # Cool, try to keep it like this
                    adv_reward += 4.0 / adv_ego_ttc # [1, 4]
                elif adv_ego_ttc > 4.0: # Too safe, get closer to ego!
                    adv_reward -= adv_ego_ttc / 4.0 # [1, inf]
            else: # Release phase, let ego go!
                if 0.0 <= adv_ego_ttc <= 4.0: # OkaaaYYY you really gotta back off now 
                    adv_reward -= 4.0 / adv_ego_ttc # [1, 4]

            # Consolidate rewards

            indiv_rewards[i] += adv_reward

        # ====== GLOBAL REWARDS: Rewarding team performance =======
        
        # Try sandwiching the ego

        longitudinal_occupancy_longitudinal_corridor = reward_args["longitudinal_occupancy_longitudinal_corridor"]
        lateral_occupancy_longitudinal_corridor = reward_args["lateral_occupancy_longitudinal_corridor"]
        lane_keeping_corridor = reward_args["lane_keeping_corridor"]

        if not is_release_phase:
            zones_occupied = {"front": 0, "back": 0, "left": 0, "right": 0}

            for adv in adversaries:
                dx = adv.position[0] - ego.position[0]
                dy = adv.position[1] - ego.position[1]

                if 0 < dx < longitudinal_occupancy_longitudinal_corridor and abs(dy) < lane_keeping_corridor: # 25m in front of ego and within lane width
                    zones_occupied["front"] = 1
                elif -longitudinal_occupancy_longitudinal_corridor < dx < 0 and abs(dy) < lane_keeping_corridor: # 25m behind ego and within lane width
                    zones_occupied["back"] = 1
                elif abs(dx) < lateral_occupancy_longitudinal_corridor and abs(dy) > lane_keeping_corridor:
                    if dy > 0:
                        zones_occupied["right"] = 1
                    elif dy < 0:
                        zones_occupied["left"] = 1
            
            occupied_count = sum(zones_occupied.values())
            team_reward += occupied_count ** 2 * 2.5

        # Ego reached goal!!
        if ego.lane_index[0] == 'l' and ego.lane_index[1] == 'm':
            if not ego.crashed:
                team_reward += 100.0

        # Ego has crashed!!
        if ego.crashed:
            team_reward -= 100.0

        for i in range(len(adversaries)):
            indiv_rewards[i] += team_reward # Penalty for crashing into ego

        return indiv_rewards
    
    def step(self, split_actions):
        
        # Actions shape has been reverted to highway env original space (tuple of tuples, one for each vehicle, each containing acceleration and steering)

        self.action_type.act(split_actions)

        self._simulate()

        # Crash if driving out of bounds (off the road)
        for vehicle in self.road.vehicles:
            if self._is_out_of_bounds(vehicle):
                
                vehicle.crashed = True

                # Penalize adversary for crashing
                
                if vehicle in self.controlled_vehicles:
                    self.config["adv_crash_penalization"][self.controlled_vehicles.index(vehicle)] = True

        # Foolproof crashing of ego if it drives off the road (shouldn't happen with IDM + MOBIL, but just in case)
        if self._is_out_of_bounds(self.ego):
            self.ego.crashed = True

        # Grab new observations, rewards, and termination/truncation flags

        obs = self.observation_type.observe()

        reward = self._reward(split_actions)

        terminated = bool(self.ego.crashed)

        truncated = False

        if (self.ego.lane_index[0] == 'l' and self.ego.lane_index[1] == 'm' and not self.ego.crashed) or self.steps >= self.config['duration']:
            truncated = True

        return obs, reward, terminated, truncated, {}

class Wrapper_MergeExitLaneHighway_Environment(gym.Wrapper):
    def __init__(self, env):

        """
        Wrapper takes the multi-agent observation space of the original environment, and takes the necessary
         observations for both and processes it to be arranged in one dimension.

        Observation will be adversarial vehicle (adv)-centric; every other vehicle's observation values will be modified
        to be in relation to the main adv.

        The derived values are detailed in the chart below. Presence values will be binary, while the other 5 features are 
        normalized to a range of 0 to 1.

        - Dist. from exit: Fraction of total distance from init. point to goal
        - Dist. from adv: Clipped to range of 0 to 20m and normalized to 0 to 1.
        - 2D TTC: Clipped to range of 0 to 10s and normalized to 0 to 1.


                   |      adv (self)     |        ego         |        adv 1       | ..... |    adv n-1         |
        ---------------------------------------------------------------------------------------------------------
         Presence  |        self         |       self         |         self       | ..... |      self          |
         x         |  dist. from exit    |  dist. from adv    |   dist. from adv   | ..... |  dist. from adv    |
         y         |  dist. from exit    |  dist. from adv    |   dist. from adv   | ..... |  dist. from adv    |
         speed     |        self         | mag. of vec. diff. | mag. of vec. diff. | ..... | mag. of vec. diff. |
         "heading" |        self         | angle of vec. diff | angle of vec. diff | ..... | angle of vec. diff |
         2D TTC    |                     |   with adv         |   with adv         | ..... |    with adv        |

         
        This, when flattened to a one-dimensional array, serves as the input of the actor NN.


        """

        super().__init__(env)

        self.configs = self.env.unwrapped.config
        self.RLargs = RLArgs()

        self.adv_agents = self.configs.get(
            'controlled_vehicles', 10
        )

    def observation(self, obs):
        """
        Builds the MASAC feature vector directly from the physics engine objects.
        Ignores the default 'obs' matrix to prevent floating-point mismatch and sorting bugs.
        Vector shape: [Self (5), Victim (6), Other Advs sorted by distance (n-1 * 6)]

        """

        def _take_magnitude(vx, vy):
            return np.sqrt(vx**2 + vy**2)
        
        env = self.env.unwrapped
        
        # Grab target distances

        target_long_dist = sum(env.config['ends_m'][:4])
        target_lat_dist = -env.config['lane_width_m']
        
        # Identify the victim vehicle (actual object)
        
        victim = None
        for vehicle in env.road.vehicles:
            if vehicle not in env.controlled_vehicles:
                victim = vehicle
                break

        # Initialize the processed observation array

        processed_obs = np.zeros((self.adv_agents, self.RLargs.obs_dim), dtype=np.float32)

        # Build all (6n + 5) observations for each of adversaries

        for i in range(self.adv_agents):
            
            # Defense against despawned agents (if a platoon car is deleted from the engine)
            if i >= len(env.controlled_vehicles):
                continue # Leaves this agent's row as pure zeros
                
            self_veh = env.controlled_vehicles[i]

            # Features to be built modularly, first for self, then for victim (lane changer), then for other adversaries, and finally concatenated into one row of the observation matrix
            
            # ===== SELF OBSERVATION =====
            self_x, self_y = self_veh.position
            self_vx, self_vy = self_veh.velocity
            
            current_adv_obs = np.array([
                1.0, # Presence
                (target_long_dist - self_x) / target_long_dist,
                (self_y - target_lat_dist) / (self.configs.get('lane_width_m', 4) * self.configs.get('num_lanes', 5) - target_lat_dist),
                _take_magnitude(self_vx, self_vy) / self.configs.get('speed_limit', 20),
                self_veh.heading / (2 * np.pi) + 0.5
            ], dtype=np.float32)

            # ===== VICTIM OBSERVATION =====
            victim_obs = np.zeros(6, dtype=np.float32)
            if victim is not None:
                v_x, v_y = victim.position
                v_vx, v_vy = victim.velocity
                
                ttc = PolygonTTCCalculator.compute_ttc(self_veh, victim)
                
                victim_obs = np.array([
                    1.0, # Victim presence
                    (v_x - self_x) / self.configs.get('rel_dist_normalizer', 20.0),
                    (v_y - self_y) / self.configs.get('rel_dist_normalizer', 20.0),
                    _take_magnitude(v_vx - self_vx, v_vy - self_vy) / self.configs.get('speed_limit', 20),
                    np.arctan2(v_vy - self_vy, v_vx - self_vx) / (2 * np.pi) + 0.5,
                    np.clip(ttc / self.configs.get('ttc_normalizer', 10.0), 0.0, 1.0)
                ], dtype=np.float32)

            # ===== OTHER ADV OBSERVATION =====
            other_advs_list = []
            
            for j, other_veh in enumerate(env.controlled_vehicles):
                if i == j: 
                    continue # Skip self
                    
                o_x, o_y = other_veh.position
                o_vx, o_vy = other_veh.velocity
                
                ttc = PolygonTTCCalculator.compute_ttc(self_veh, other_veh)
                distance = np.hypot(o_x - self_x, o_y - self_y)
                
                feat = np.array([
                    1.0, # Presence
                    (o_x - self_x) / self.configs.get('rel_dist_normalizer', 20.0),
                    (o_y - self_y) / self.configs.get('rel_dist_normalizer', 20.0),
                    _take_magnitude(o_vx - self_vx, o_vy - self_vy) / self.configs.get('speed_limit', 20),
                    np.arctan2(o_vy - self_vy, o_vx - self_vx) / (2 * np.pi) + 0.5,
                    np.clip(ttc / self.configs.get('ttc_normalizer', 10.0), 0.0, 1.0)
                ], dtype=np.float32)
                
                other_advs_list.append((distance, feat))
                
            # Sort other adversaries by distance so the network gets consistent structure
            other_advs_list.sort(key=lambda x: x[0])
            
            # Flatten the sorted features, padding with zeros if any platoon cars despawned
            other_adv_obs = np.zeros((self.adv_agents - 1) * 6, dtype=np.float32)
            for idx, (_, feat) in enumerate(other_advs_list):
                if idx < (self.adv_agents - 1):
                    other_adv_obs[idx * 6 : (idx + 1) * 6] = feat

            # ===== CONCATENATE =====
            processed_obs[i] = np.concatenate([
                current_adv_obs,
                victim_obs,
                other_adv_obs
            ])
            
        return processed_obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Flatten the observation matrix
        return self.observation(obs), info
    
    def step(self, action):
        
        """
        Coming from the RL agent, action is a 2D NumPy array of shape (adv_agents, action_dim). Each row corresponds to an adversarial vehicle's action.
        The wrapper formats into a highway-env's default tuple of tuples, where each inner tuple corresponds to an individual vehicle's action.
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
        
        # Turn next observation into shape RL can read (flatten) and return everything
        return self.observation(_obs), reward, terminated, truncated, info
    



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
import numpy as np
import gymnasium as gym

from dataclasses import asdict

from highway_env.envs.common.abstract import AbstractEnv, Vehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, SineLane

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

from src.env.risk_calculators import PolygonTTCCalculator, THWCalculator
from src.env.reward_functions import (
    AdversarialCrashPenalty,
    DistanceToEgoRewardFunction,
    LaneKeepingRewardFunction,
    RewardTTCAdvAdvFunction, 
    RewardTTCEgoAdvFunction, 
    SandwichingRewardFunction,
    SimpleRewardFunction
)



from configs.configs import EnvArgs, RLArgs

MAX_TTC_SECONDS = 12.0
MIN_TTC_SECONDS = 1e-3
VEHICLE_WIDTH = Vehicle.WIDTH
MAX_SPEED = Vehicle.MAX_SPEED
MIN_SPEED = Vehicle.MIN_SPEED


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

        left_boundary = self.road.network.get_lane((current_segment[0], current_segment[1], min_index)).position(longitudinal=current_position[0], lateral=0) - np.array([0, lane_width_m / 2])
        right_boundary = self.road.network.get_lane((current_segment[0], current_segment[1], max_index)).position(longitudinal=current_position[0], lateral=0) + np.array([0, lane_width_m / 2])
        
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

        num_adv = self.config["controlled_vehicles"]

        self.controlled_vehicles = []

        # LANE CHANGER: Spawns on the on-ramp and is controlled by IDM + MOBIL. The goal is to reach the exit ramp successfully.

        self.ego_lane = self.road.network.get_lane(("j", "k", 0))

        self.ego = IDMVehicle(
            self.road,
            self.ego_lane.position(0, 0),
            speed = 25,
            route = [("j", "k", 0),("k","b", 0),("b","c",0),("c","d",0),("d","e",0),("e","l",0),("l","m",0)]
        )

        self.observer_vehicle = self.ego # The vehicle that the environment observes for reward calculation

        self.road.vehicles.append(self.ego)

        # ADVERSARIAL VEHICLES: Controlled by MDP + MASAC-RL. The goal is to maximize risk between ego and adversaries, while minimizing risk between adversaries.
    
        lanes_count = self.config["lanes_count"]
    
        for i in range(num_adv):
            highway_lane = self.road.network.get_lane(("a", "b", np.random.randint(0, lanes_count)))
            longitudinal_pos_m = 40 * i
            adv = Vehicle(
                self.road,
                highway_lane.position(longitudinal_pos_m, 0),
                speed = np.random.uniform(25,36)
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

        # Setting important environment-related variables
        
        ego = self.ego
        adversaries = self.controlled_vehicles

        # Initializing the rewards

        indiv_rewards = np.zeros(len(adversaries), dtype=np.float32)

        # Splitting bullying into two phases: (1) blocking the ego from reaching the exit ramp, and (2) allowing the ego to reach the exit ramp successfully

        adv_adv_reward_calculator = RewardTTCAdvAdvFunction()
        adv_ego_reward_calculator = RewardTTCEgoAdvFunction()
        sandwiching_reward_calculator = SandwichingRewardFunction()
        simple_reward_calculator = SimpleRewardFunction()
        lane_keeping_reward_calculator = LaneKeepingRewardFunction()
        dist_to_ego_reward_calculator = DistanceToEgoRewardFunction()
        adv_crash_penalty_calculator = AdversarialCrashPenalty()

        adv_ego_reward_calculator.check_phase(ego.position[0])

        # ====== LOCAL REWARDS: Rewarding each adversary =======

        for i, adv in enumerate(adversaries):

            adv_reward = 0.0

            # Calculate distance to ego and apply distance-based reward
            dist_to_ego = np.linalg.norm(adv.position - ego.position)
            adv_reward += dist_to_ego_reward_calculator.compute_reward(dist_to_ego)

            # Dont't make the ego go below the speed limit (20 m/s) by driving too close to it
            if not adv_ego_reward_calculator.is_release_phase:
                if ego.velocity[0] < 20.0 and dist_to_ego < 10.0:  # If ego is below speed limit and adv is too close
                    adv_reward += simple_reward_calculator.get_reward("adv_ego_speed_penalty")

            # Don't reverse on the highway!
            if adv.velocity[0] < 0:
                adv_reward += simple_reward_calculator.get_reward("adv_reverse_penalty")

            # Drive safe!
            if self.config["adv_crash_penalization"][i]:
                step_since_crash = self.config["adv_step_since_crash_counter"][i]
                adv_reward += adv_crash_penalty_calculator.compute_reward(step_since_crash)

            # If adversary driving too close to boundary

            left_boundary, right_boundary = self._get_current_lateral_boundaries(adv)
            adv_reward += lane_keeping_reward_calculator.compute_reward(adv.position[1], left_boundary[1], right_boundary[1])

            # Don't endanger other team mates!
            for j, other_adv in enumerate(adversaries):
                if i == j: continue
                adv_adv_ttc = PolygonTTCCalculator.compute_ttc(adv, other_adv)
                adv_reward += adv_adv_reward_calculator.compute_reward(adv_adv_ttc)
            
            # Bully the ego!
            adv_ego_ttc = PolygonTTCCalculator.compute_ttc(adv, ego)
            adv_ego_thw = THWCalculator.compute_thw(adv, ego)

            adv_ego_reward_calculator.check_phase(ego.position[0])
            adv_reward += adv_ego_reward_calculator.compute_reward(adv_ego_ttc, adv_ego_thw)

            # Consolidate rewards

            indiv_rewards[i] += adv_reward

        # ====== PLATOON REWARDS: Rewarding team performance =======
        
        # Try sandwiching the ego

        sandwiching_reward_calculator.check_phase(ego.position[0])

        indiv_rewards += sandwiching_reward_calculator.compute_reward(adversaries, ego) 

        # Ego reached goal!!
        if ego.lane_index[0] == 'l' and ego.lane_index[1] == 'm' and not ego.crashed:
            indiv_rewards += np.array(simple_reward_calculator.get_reward("ego_reach_exit_reward"), dtype=np.float32)

        # Ego has crashed!!
        if ego.crashed:
            indiv_rewards += np.array(simple_reward_calculator.get_reward("ego_crash_penalty"), dtype=np.float32)

        return indiv_rewards
    
    def step(self, split_actions):
        
        # Actions shape has been reverted to highway env original space (tuple of tuples, one for each vehicle, each containing acceleration and steering)

        self.action_type.act(split_actions)

        self._simulate()

        # Crash if driving out of bounds (off the road)
        for vehicle in self.road.vehicles:
            if self._is_out_of_bounds(vehicle) and vehicle in self.controlled_vehicles:
                vehicle.crashed = True

            if vehicle.crashed and vehicle in self.controlled_vehicles:
                vehicle_idx = self.controlled_vehicles.index(vehicle)
                self.config["adv_crash_penalization"][vehicle_idx] = True
                self.config["adv_step_since_crash_counter"][vehicle_idx] += 1

            if vehicle in self.controlled_vehicles:

                if PolygonTTCCalculator.compute_ttc(vehicle, self.ego) <= 4.0:  # Example condition
                    vehicle.color = (128, 0, 128)  # Orange color for both vehicles
                    self.ego.color = (128, 0, 128)  # Orange color for both vehicles
                else:
                    vehicle.color = None  # Reset to default color
                    self.ego.color = None  # Reset to default color
            

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


        NEW IDEA: 
        (1) Self observation:
            - activity (0 or 1)
            - dist. from left boundary (fraction of total width)
            - dist. from right boundary (fraction of total width)
            - long. dist. to ego (fraction of total distance to exit)
            - lat. dist. to ego (fraction of total width)
            - long. speed (fraction of speed limit)
            - lat. speed (fraction of speed limit)
        (2) Victim (ego LC) observation:
            - long. dist. to exit (fraction of total distance to exit)
            - lat. dist. to exit (fraction of total width)
            - rel. long. speed to self (fraction of speed limit)
            - rel. lat. speed to self (fraction of speed limit)
            - inverse TTC
        (3) Other adversaries observation:
            - activity (0 or 1)
            - long. dist. to self (fraction of total distance to exit)
            - lat. dist. to self (fraction of total width)
            - rel. long. speed to self (fraction of speed limit)
            - rel. lat. speed to self (fraction of speed limit)
            - inverse TTC

        """
        
        env = self.env.unwrapped
        
        # Grab target distances

        target_long_dist = sum(env.config['ends_m'][:4])
        target_lat_dist = -env.config['lane_width_m']

        # Grab highway geometry

        lane_width_m = env.config['lane_width_m']
        lanes_count = env.config['lanes_count']

        highway_safe_buffer = lane_width_m - VEHICLE_WIDTH # Adjusting for vehicle width to avoid clipping into lane boundaries
        lanes_total_width = lane_width_m * lanes_count
        
        
        # Identify the victim vehicle (actual object)
        
        victim = None
        for vehicle in env.road.vehicles:
            if vehicle not in env.controlled_vehicles:
                victim = vehicle
                break

        # Initialize the processed observation array

        processed_obs = np.zeros((self.adv_agents, self.RLargs.obs_dim), dtype=np.float32)

        # Build all each observation space of each adversary

        for i in range(self.adv_agents):
            
            # Defense against despawned agents (if a platoon car is deleted from the engine)
            if i >= len(env.controlled_vehicles):
                continue # Leaves this agent's row as pure zeros
                
            self_veh = env.controlled_vehicles[i]

            # Features to be built modularly, first for self, then for victim (lane changer), then for other adversaries, and finally concatenated into one row of the observation matrix
            
            
            self_x, self_y = self_veh.position
            self_vx, self_vy = self_veh.velocity

            if victim is not None:
                victim_x, victim_y = victim.position
                victim_vx, victim_vy = victim.velocity
            else:
                victim_x, victim_y = 0.0, 0.0
                victim_vx, victim_vy = 0.0, 0.0
            
            # ===== SELF OBSERVATION =====

            if self_veh.crashed:
                continue # Skip crashed adversaries, leaving their observation row as pure zeros
            else:
                current_adv_obs = np.array([
                    1.0,
                    (self_y - highway_safe_buffer/2) / (lanes_total_width + highway_safe_buffer), # dist to left boundary
                    (lanes_total_width - (self_y + highway_safe_buffer/2)) / (lanes_total_width + highway_safe_buffer), # dist to right boundary
                    (victim_x - self_x) / target_long_dist, # long. dist to victim
                    (victim_y - self_y) / (lanes_total_width + highway_safe_buffer - target_lat_dist), # lat. dist to victim
                    self_vx / ((MAX_SPEED - MIN_SPEED)/2), # long. speed
                    self_vy / ((MAX_SPEED - MIN_SPEED)/(2*np.sqrt(2))) # lat. speed
                ], dtype=np.float32)

            # ===== VICTIM OBSERVATION =====
            victim_obs = np.zeros(5, dtype=np.float32)
            
            victim_obs = np.array([
                (target_long_dist - victim_x) / target_long_dist,
                (target_lat_dist - victim_y) / (lanes_total_width - target_lat_dist),
                (victim_vx - self_vx) / (MAX_SPEED - MIN_SPEED), # rel. long. speed to victim
                (victim_vy - self_vy) / ((MAX_SPEED - MIN_SPEED)/(2*np.sqrt(2))),
                1.0 / (PolygonTTCCalculator.compute_ttc(self_veh, victim) + 1.0)
            ], dtype=np.float32)

            # ===== OTHER ADV OBSERVATION =====
            other_advs_list = []
            
            for j, other_veh in enumerate(env.controlled_vehicles):
                if i == j: 
                    continue # Skip self
                    
                o_x, o_y = other_veh.position
                o_vx, o_vy = other_veh.velocity
                
                distance = np.hypot(o_x - self_x, o_y - self_y)
                
                if other_veh.crashed:
                    feat = np.zeros(6, dtype=np.float32)
                else:
                    feat = np.array([
                        1.0,
                        (o_x - self_x) / target_long_dist,
                        (o_y - self_y) / (lanes_total_width + highway_safe_buffer - target_lat_dist),
                        (o_vx - self_vx) / ((MAX_SPEED - MIN_SPEED)/2),
                        (o_vy - self_vy) / ((MAX_SPEED - MIN_SPEED)/(2*np.sqrt(2))),
                        1.0 / (PolygonTTCCalculator.compute_ttc(self_veh, other_veh) + 1.0)
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
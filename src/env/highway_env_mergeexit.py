import numpy as np
import time
import yaml

from pathlib import Path

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, SineLane

from highway_env.vehicle.behavior import IDMVehicle

class MergeExitLaneHighway_Environment(AbstractEnv):
    """
    A customized environment with an on-ramp and and exit ramp.

    The ego vehicle starts driving from the merge ramp and traverses the highway to exit through the
     exit ramp. The ego vehicle will be controlled not by the agent of the environment, rather by a
      separate white-box MPC-based controller. The agent of this environment will be the adversarial platoon of vehicles,
        trying to spike the risk metrics (to crash) of this ego vehicle.
    
        The platoon is to be composed of 2n vehicles for a highway environment of n lanes. Each vehicle has the following
        inputs

    ## Action Space
    The agent takes a 

    ## Rewards

    The reward consists of two parts:

    The goal of the environment is to have the ego lane-changer (LC) go from the on-ramp to the exit ramp.
    A final reward of +10 is given to the agent 

    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()

        current_dir = Path(__file__).resolve().parent
        params_path = current_dir.parent.parent / "configs" / "params_main.yaml"

        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        config.update(params)

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

    def find_borders(self, target_long_pos: float):
        
        lanes_count = self.config['env_params']['lanes_count']
        lane_width = self.config['env_params']['lane_width_m']

        sections = self.config['env_params']['ends_m']
        # ex. [150, 80, 80, 300, 80, 80, 150]

        target_sec_idx = 0
        total = target_long_pos
        for sec_idx in range(len(sections)):
            total -= sections[sec_idx]
            if total <= 0:
                break
            target_sec_idx += 1
        

        match target_sec_idx:
            case 0:
                target_lat_upper = 
                target_lat_lower = 
            case 1:
                target_lat_upper = 
                target_lat_lower =
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            
        if target_sec_idx <= 2:

        elif target_sec_idx == 3:
            target_lat_upper = 0
            target_lat_lower = lane_width * lanes_count
        elif target_sec_idx >= 4:



        target_lat_upper = 
        return target_lat_upper, target_lat_lower

    def _make_vehicles(self) -> None:
        self.controlled_vehicles = []

        # Spawning SUT
        ego_lane = self.road.network.get_lane(("j", "k", 0))
        ego = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(0, 0),
            speed = 25
        )

        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        # Spawning adversarial cruisers

        lanes_count = self.config["lanes_count"]

        for lane_idx in range(lanes_count):
            highway_lane = self.road.network.get_lane(("a", "b", lane_idx))
            for car_idx in range(2):
                longitudinal_pos_m = 40 - (car_idx * 20)

                vehicle = self.action_type.vehicle_class(
                    self.road,
                    highway_lane.position(longitudinal_pos_m, 0),
                    speed = 25
                )
                self.road.vehicles.append(vehicle)

        self.vehicle = self.controlled_vehicles[0]

    def _make_stock_vehicles(self) -> None:

        self.controlled_vehicles = []

        ego_lane = self.road.network.get_lane(("j", "k", 0))

        ego = IDMVehicle(
            self.road,
            ego_lane.position(0, 0),
            speed = 25,
            route = [("j", "k", 0),("k","b", 0),("b","c",4),("c","d",4),("d","e",4),("e","l",0),("l","m",0)]
        )

        # ego = self.action_type.vehicle_class(
        #     self.road,
        #     ego_lane.position(0,0),
        #     speed = 25
        # )

        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)


        for _ in range(10):
            vehicle = IDMVehicle.create_random(
                self.road,
                speed = np.random.uniform(20,30),
                spacing = 0.5,
            )
            self.road.vehicles.append(vehicle)

        # ego.plan_route_to("l")

        # for lane_idx in range(5):
        #     highway_lane = self.road.network.get_lane(("a", "b", lane_idx))
        #     for car_idx in range(2):
        #         longitudinal_pos_m = 20 - (car_idx * 20)

        #         vehicle = self.action_type.vehicle_class(
        #             self.road,
        #             highway_lane.position(longitudinal_pos_m, 0),
        #             speed = 0
        #         )
        #         self.road.vehicles.append(vehicle)

        self.vehicle = self.controlled_vehicles[0]


    def _reset(self) -> None:
        self._make_road()
        self._make_stock_vehicles()

    def _reward(self, action: int) -> float:
        """Dummy reward function for MPC testing."""
        return 0.0

    def _is_terminated(self) -> bool:
        """Tells the engine to stop if the car crashes."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """Tells the engine to stop if the time limit is reached."""
        return self.time >= self.config["duration"]




env = MergeExitLaneHighway_Environment()
env.render_mode = "human" # Tells Gymnasium to prepare a visual window
env.reset()

# Run a loop to step the physics engine forward in time
for _ in range(200):
    # Action '1' is the IDLE action (tells the car to just cruise forward)
    # If your gym version is older, you might only get 4 return values instead of 5
    obs, reward, done, truncated, info = env.step(1)

    env.render()

    time.sleep(0.05) # <--- Add this to slow down the frame rate

    # If the car crashes or finishes the route, reset the map
    if done or truncated:
        env.reset()
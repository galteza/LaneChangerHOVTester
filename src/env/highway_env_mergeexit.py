import numpy as np
import time

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, SineLane

class MergeExitLaneHighway_Environment(AbstractEnv):
    """
    A customized environment with an on-ramp and and exit ramp.

    The ego-vehicle will be driving on the merge and will be exiting through to the exit ramp by traversing the highway.
    The ego vehicle will be using the created SUT lane changer MPC created.

    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # === STARTER CONFIGS ===

            "observation": {
                "type" : "Kinematics"
            },
            "action" : {
                "type" : "DiscreteMetaAction",
            },
            "duration" : 40,
            "initial_spacing" : 2,
            "simulation_frequency" : 15,
            "policy_frequency" : 1,

            # === ENVIRONMENT ===

            "lanes_count" : 5,
            "lane_width_m": 4.0,
            "ends_m": [150, 80, 80, 300, 80, 80, 150], # Establishing lengths of each section
            "merge_amplitude": 3.25,

            # === CAMERA SETTINGS ===

            "scaling": 3,                # Default is ~5.5. Lower number = further zoomed out!
            "screen_width": 1200,          # Makes the Pygame window much wider (Default is 600)
            "screen_height": 400,          # Makes the Pygame window taller
            "centering_position": [0.2, 0.5], # Pushes the car to the left 20% of the screen so you can see more of the road ahead
            })
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

        # === HIGHWAY LANES

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


        net.add_lane(
            "d", "e",
            StraightLane(
                [sum(ends_m[:4]), -lane_width_m],
                [sum(ends_m[:5]), -lane_width_m],
                line_types=[c,n],
                forbidden=True,
            )
        )

        # MERGING LANE (modeling the curve using sine wave)
        amplitude = self.config["merge_amplitude"]

        merging_jk = StraightLane( # Before
            [0, amplitude*2 + lane_width_m*(lanes_count)],
            [ends_m[0], amplitude*2 + lane_width_m*(lanes_count)],
            line_types=[c,c],
            forbidden=True
        )
        merging_kb = SineLane( # Converging
            merging_jk.position(ends_m[0], -amplitude),
            merging_jk.position(sum(ends_m[:2]), -amplitude),
            amplitude, # amplitude
            2 * np.pi / (2 * ends_m[1]), # pulsation
            np.pi / 2, # phase
            line_types=[c, c],
            forbidden=True,
        )
        merging_bc = StraightLane( # Merge ramp connection
            merging_kb.position(ends_m[1], 0),
            merging_kb.position(ends_m[1], 0) + [ends_m[2], 0],
            line_types=[n,c],
            forbidden=True,
        )


        net.add_lane("j", "k", merging_jk)
        net.add_lane("k", "b", merging_kb)
        net.add_lane("b", "c", merging_bc)

        # EXIT LANE (modeling the curve using sine wave)
        exit_ref = StraightLane(
            [sum(ends_m[:4]), -lane_width_m - amplitude],
            [sum(ends_m[:5]), -lane_width_m - amplitude],
        )

        merging_el = SineLane(
            exit_ref.position(ends_m[4], 0),
            exit_ref.position(sum(ends_m[4:6]), 0),
            amplitude,
            2 * np.pi / (2 * ends_m[5]),
            np.pi / 2,
            line_types=[c,c],
            forbidden=True,
        )

        merging_lm = StraightLane(
            merging_el.position(ends_m[5], 0),
            merging_el.position(ends_m[5], 0) + [ends_m[6], 0],
            line_types=[c, c],
            forbidden=True,
        )

        net.add_lane("e", "l", merging_el)
        net.add_lane("l", "m", merging_lm)

        self.road = Road(
            network = net,
            np_random = self.np_random,
            record_history = self.config["show_trajectories"],
        )

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


    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

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

    # This is the magic command that physically draws the Pygame window!
    env.render()

    time.sleep(0.05) # <--- Add this to slow down the frame rate

    # If the car crashes or finishes the route, reset the map
    if done or truncated:
        env.reset()
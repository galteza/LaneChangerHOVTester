import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

# ===== ENVIRONMENT CONFIGURATION CLASSES ====== #

@dataclass
class EnvActionConfigArgs:
    type: str = "ContinuousAction"
    longitudinal: bool = True
    lateral: bool = True
    acceleration_range: tuple[float, float] = field(default_factory=lambda: (-4.0, 1.0)) # in m/s^2
    steering_range: tuple[float, float] = field(default_factory=lambda: (-0.174, 0.174)) # approx. 10 degrees in rad
    # can add speed range too! it's a tuple

@dataclass
class EnvActionArgs:
    type: str = "MultiAgentAction"
    action_config: EnvActionConfigArgs = field(default_factory=EnvActionConfigArgs)

@dataclass
class EnvObsConfigArgs:
    type: str = "Kinematics"
    normalize: bool = False
    absolute: bool = True
    vehicles_count: int = 3 # 1 System Under Test (SUT) + 10 Adversaries
    features: List[str] = field(
        default_factory=lambda: ["presence", "x", "y", "vx", "vy", "heading"]
    )

@dataclass
class Env_ObsArgs:
    type: str = "MultiAgentObservation"
    observation_config: EnvObsConfigArgs = field(default_factory=EnvObsConfigArgs)

@dataclass
class EnvRewardArgs:

    # Blocking and release parameters

    release_distance: float = 20.0

    # TTC-based reward shaping parameters (Adversary-to-Adversary and Ego-to-Adversary)

    advadv_baseline_N: int = 5.7
    advadv_peak_P: float = 5.85
    advadv_rise_slope_k1: float = 0.8
    advadv_decay_slope_k2: float = 0.6
    advadv_rise_shift_a: float = 3.4
    advadv_decay_shift_b: float = 9.5
    
    egoadv_blocking_baseline_N: float = 5.7
    egoadv_blocking_peak_P: float = 9.5
    egoadv_blocking_rise_slope_k1: float = 2.4
    egoadv_blocking_decay_slope_k2: float = 1.3
    egoadv_blocking_rise_shift_a: float = 1.4
    egoadv_blocking_decay_shift_b: float = 6.0

    egoadv_release_baseline_N: float = 5.7
    egoadv_release_peak_P: float = 9.5
    egoadv_release_rise_slope_k1: float = 2.4
    egoadv_release_decay_slope_k2: float = 1.3
    egoadv_release_rise_shift_a: float = 3.6
    egoadv_release_decay_shift_b: float = 6.0

    # Distance to ego reward function

    dist_base1_c1: float = 3.1
    dist_base2_c2: float = 0
    dist_base3_c3: float = -3.2
    dist_down1_a: float = 23
    dist_down2_b: float = 49.4
    dist_slope1_k1: float = 0.7
    dist_slope2_k2: float = 0.2


    # Sandwiching and occupancy corridor parameters

    longitudinal_occupancy_longitudinal_corridor: float = 25.0 # meters
    lateral_occupancy_longitudinal_corridor: float = 10.0  # meters
    lane_keeping_corridor: float = 2.0 # meters

    ellipse_base_a: float = 20.0      # Base longitudinal radius (meters)
    ellipse_b: float = 3.5            # Lateral radius (slightly less than a lane width)
    speed_k: float = 0.5              # How much the ellipse stretches with ego speed

    base_proximity_reward: float = 0.18
    sandwich_bonus: float = 0.5

    # Lane keeping reward parameters

    max_lane_penalty: float = 0.4
    boundary_hit_penalty: float = -0.5

    # Simple addition reward parameters

    adv_crash_penalty: float = -60.0
    ego_crash_penalty: float = -110.0
    ego_reach_exit_reward: float = 80.0
    adv_reverse_penalty: float = -10.0
    adv_ego_speed_penalty: float = -10.0


    """
    # Adversary-to-Adversary TTC Penalties

    adv_adv_ttc_close_penalty: float = -50.0

    adv_adv_ttc_near_m: float = -60.0
    adv_adv_ttc_near_b: float = 10.0

    adv_adv_ttc_far_m: float = -2.0
    adv_adv_ttc_far_b: float = 24.0

    # Adversary-to-Ego TTC Penalties

    adv_ego_ttc_close_penalty: float = -60.0

    adv_ego_ttc_near_a: float = -20.8
    adv_ego_ttc_near_h: float = 2.8
    adv_ego_ttc_near_k: float = 100.0

    adv_ego_ttc_far_m: float = -6.25
    adv_ego_ttc_far_b: float = 15.0

    adv_release_phase_m: float = 12.5
    adv_release_phase_b: float = -70.0
    """

@dataclass
class EnvArgs:
    
    """Highway-env geometry, multi-agent infrastructure, and generation setup."""

    action: EnvActionArgs = field(default_factory=EnvActionArgs)

    observation: Env_ObsArgs = field(default_factory=Env_ObsArgs)

    reward: EnvRewardArgs = field(default_factory=EnvRewardArgs)

    env_id: str = "merge_exit_highway"
    render_mode: Optional[str] = None
    lanes_count: int = 2
    lane_width_m: int = 4
    # Establishing lengths of each physical section
    ends_m: List[int] = field(default_factory=lambda: [150, 80, 80, 300, 80, 80, 150])
    merge_amplitude: float = 3.25

    # Observation modification parameters
    speed_limit: float = 20.0 # lifted from highway-env default
    rel_dist_normalizer: float = 20.0
    ttc_normalizer: float = 10.0

    # Vehicle configuration
    vehicles_count: int = 20          # Total ambient cars
    controlled_vehicles: int = 0 # defined post init
    adv_crash_penalization: List[bool] = field(default_factory=lambda: [False]*10)
    adv_rewards: List[float] = field(default_factory=lambda: [0.0]*10)
    vehicle_density: float = 0.0
    initial_lane_id: int = 0
    initial_spacing: int = 2

    # Camera settings
    scaling: int = 3
    screen_width: int = 1200
    screen_height: int = 400
    centering_position: List[float] = field(default_factory=lambda: [0.2, 0.5])
    offscreen_rendering: bool = False

    # Simulation configuration
    duration: int = 10000             # Seconds per episode
    simulation_frequency: int = 10
    policy_frequency: int = 10        # How often MPC is called

    def __post_init__(self):
        self.controlled_vehicles = self.observation.observation_config.vehicles_count - 1


# ===== RL CONFIGURATION CLASSES ====== #

@dataclass
class RLArgs:
    env: EnvArgs = field(default_factory=EnvArgs)

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    device: str = "cuda" if torch.cuda.is_available() and cuda else "cpu"
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    
    # Environment & MASAC specific args
    env_id: str = "merge_exit_highway"
    num_agents: int = 0 # to be defined post init
    obs_dim: int = 0 # to be defined post init
    action_dim: int = 2  # Throttle, Steering
    
    total_timesteps: int = 1000000
    buffer_size: int = int(5e5) # int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5e2
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    checkpoints_num: int = 100
    logging_frequency: int = 100

    def __post_init__(self):
        self.num_agents = self.env.controlled_vehicles
        self.obs_dim = 7 + 5 + (self.num_agents - 1) * 6 # 7 for main adv, 5 for ego, 6 for each other adv


@dataclass
class VehicleSafetyMargins:
    """Boundary parameters to prevent vehicle overlap calculations."""
    longitudinal_margin: float = 5.0
    lateral_margin: float = 2.0


@dataclass
class VehicleModelParams:
    """Physical constraints and performance bounds (unnormalized, Volvo Standards)."""
    wheelbase_L: float = 2.5                 # in meters
    max_steer_rad: float = 0.174             # approx. 10 degrees in rad
    min_steer_rad: float = -0.174
    max_long_accel_ms2: float = 1.0          # in m/s^2
    min_long_accel_ms2: float = -4.0         # in m/s^2
    max_long_vel_ms: int = 30

    max_yaw_rad: float = 0.087               # approx. 5 degrees in rad
    min_yaw_rad: float = -0.087

    target_heading_angle: int = 0
    
    # Comfort and jerk penalties
    max_steering_rate_rads: float = 0.296    # in rad/s (for ride comfort)
    min_steering_rate_rads: float = -0.296   # if using rate-of-change inputs
    
    # Nested safety parameters
    safety_margins: VehicleSafetyMargins = field(default_factory=VehicleSafetyMargins)


@dataclass
class SUTLaneChangerMPCParams:
    """Model Predictive Control horizon parameters, state tracking weights, and utility rewards."""
    horizon_N: int = 20
    dt: float = 0.1

    # Longitudinal Optimization Tracking
    Q_s: float = 10.0
    Q_v: float = 5.0
    R_a: float = 1.0                         # Higher for higher penalty in longitudinal jerkiness

    # Lateral Optimization Tracking
    Q_d: float = 20.0
    Q_heading: float = 50.0
    R_steer: float = 100.0                   # Higher for higher penalty in steering jerkiness
    
    # Utility Function Components
    utilweight_avetravtimeperlane: int = 1
    utilweight_avetimegapdensityperlane: int = 1
    utilweight_remainingtravtime: int = 1
    utilweight_urgency: int = 1

    nonzerodivparam_avetravtimeperlane: int = 2 # gamma
    scalingparam_avetimegapdensityperlane: int = 2 # alpha > 1
    urgencyfactor_urgency: float = 0.0075 # kappa
    final_utilweight_factor: float = 0.1 # zeta

    min_time_gap: int = 1
    safety_dist_buffer: int = 1

    persistence_quota_N: int = 5
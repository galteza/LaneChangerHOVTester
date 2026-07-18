import matplotlib.pyplot as plt
import numpy as np

from configs.configs import RLArgs

class RewardTTCFunction:
    """
    A reward function that penalizes vehicles based on their Time-to-Collision (TTC) with other vehicles.

    This TTC reward function will be by default a difference of sigmoids model.
    """
    
    def __init__(self):
        
        self.args = RLArgs()
        self.adv_reward = 0.0

        self.baseline_N = 0
        self.peak_P = 0
        self.rise_slope_k1 = 0
        self.decay_slope_k2 = 0
        self.rise_shift_a = 0
        self.decay_shift_b = 0

    def compute_reward(self) -> float:

        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def take_data_points(self):
        
        ttc_values = np.linspace(0.1, 30, int(30/0.1))  # From 0.1 to 30 seconds in increments of 0.1
        rewards = [self.compute_reward(ttc) for ttc in ttc_values]
        
        return ttc_values, rewards


class RewardTTCAdvAdvFunction(RewardTTCFunction):
    """
    A specialized reward function that penalizes the ego vehicle based on its Time-to-Collision (TTC) with adversary vehicles.
    """

    def __init__(self):
        super().__init__()

        self.adv_adv_ttc_close_penalty = self.args.env.reward.adv_adv_ttc_close_penalty
        self.adv_adv_ttc_near_m = self.args.env.reward.adv_adv_ttc_near_m
        self.adv_adv_ttc_near_b = self.args.env.reward.adv_adv_ttc_near_b
        self.adv_adv_ttc_far_m = self.args.env.reward.adv_adv_ttc_far_m
        self.adv_adv_ttc_far_b = self.args.env.reward.adv_adv_ttc_far_b

        self.baseline_N = 5.7
        self.peak_P = 5.85
        self.rise_slope_k1 = 0.8
        self.decay_slope_k2 = 0.6
        self.rise_shift_a = 3.4
        self.decay_shift_b = 9.5

    
    def compute_reward(self, ttc) -> float:

        self.adv_reward = -self.baseline_N + \
            (self.peak_P + self.baseline_N) / (1 + np.exp(-self.rise_slope_k1 * (ttc - self.rise_shift_a))) - \
            (self.peak_P) / (1 + np.exp(-self.decay_slope_k2 * (ttc - self.decay_shift_b)))


        # if 0.0 <= ttc < 1.0: # REALLY high risk of fratricide
        #     self.adv_reward = self.adv_adv_ttc_close_penalty # [-50]
        # elif 1.0 <= ttc <= 4.0: # Need to back off!
        #     self.adv_reward = self.adv_adv_ttc_near_m * (1/ttc) ** 1/2 + self.adv_adv_ttc_near_b # [-50, -20]
        # elif ttc > 4.0: # Okay, but don't stray too far!
        #     self.adv_reward = self.adv_adv_ttc_far_m * (1/ttc) + self.adv_adv_ttc_far_b # [-20, -60]


        return self.adv_reward
    
class RewardTTCEgoAdvFunction(RewardTTCFunction):
    """
    A specialized reward function that penalizes the ego vehicle based on its Time-to-Collision (TTC) with adversary vehicles.
    """
    
    def __init__(self):
        super().__init__()

        self.adv_ego_ttc_close_penalty = self.args.env.reward.adv_ego_ttc_close_penalty
        self.adv_ego_ttc_near_a = self.args.env.reward.adv_ego_ttc_near_a
        self.adv_ego_ttc_near_h = self.args.env.reward.adv_ego_ttc_near_h
        self.adv_ego_ttc_near_k = self.args.env.reward.adv_ego_ttc_near_k
        self.adv_ego_ttc_far_m = self.args.env.reward.adv_ego_ttc_far_m
        self.adv_ego_ttc_far_b = self.args.env.reward.adv_ego_ttc_far_b

        self.adv_release_phase_m = self.args.env.reward.adv_release_phase_m
        self.adv_release_phase_b = self.args.env.reward.adv_release_phase_b

        self.phase = "BLOCKING"  # BLOCKING or RELEASE

    def compute_reward(self, ttc) -> float:

        if self.phase == "BLOCKING":
            self.baseline_N = 5.7
            self.peak_P = 9.5
            self.rise_slope_k1 = 2.4
            self.decay_slope_k2 = 1.3
            self.rise_shift_a = 1.4
            self.decay_shift_b = 6
        elif self.phase == "RELEASE":
            self.baseline_N = 5.7
            self.peak_P = 9.5
            self.rise_slope_k1 = 2.4
            self.decay_slope_k2 = 1.3
            self.rise_shift_a = 3.6
            self.decay_shift_b = 6

        self.adv_reward = -self.baseline_N + \
            (self.peak_P + self.baseline_N) / (1 + np.exp(-self.rise_slope_k1 * (ttc - self.rise_shift_a))) - \
            (self.peak_P) / (1 + np.exp(-self.decay_slope_k2 * (ttc - self.decay_shift_b)))



        # if self.phase != "RELEASE": # Still trying to block on the highway
        #     if 0.0 <= ttc < 1.0: # Okay uhh, too much
        #         self.adv_reward = self.adv_ego_ttc_close_penalty # [-60]
        #     elif 1.0 <= ttc <= 4.0: # Cool, try to keep it like this
        #         self.adv_reward = self.adv_ego_ttc_near_a * (ttc - self.adv_ego_ttc_near_h)**2 + self.adv_ego_ttc_near_k # [-50, 20, -10]
        #     elif ttc > 4.0: # Too safe, get closer to ego!
        #         self.adv_reward = self.adv_ego_ttc_far_m * (1/ttc) + self.adv_ego_ttc_far_b  # [-10, -60]
        # else: # Release phase, let ego go!
        #     if 0.0 <= ttc <= 4.0: # OkaaaYYY you really gotta back off now 
        #         self.adv_reward = self.adv_release_phase_m * ttc + self.adv_release_phase_b # [-70, -20]
        
        return self.adv_reward
    
    def set_phase(self, phase: str):
        if phase not in ["BLOCKING", "RELEASE"]:
            raise ValueError("Phase must be either 'BLOCKING' or 'RELEASE'.")
        self.phase = phase

class SandwichingFunction():
    """
    A reward function that encourages the adversary vehicles to sandwich the ego vehicle between them, promoting a more challenging environment for the ego vehicle.
    """

    def __init__(self):
        self.args = RLArgs()
        self.sandwiching_reward = 0.0

        self.longitudinal_occupancy_longitudinal_corridor = self.args.env.reward.longitudinal_occupancy_longitudinal_corridor
        self.lateral_occupancy_longitudinal_corridor = self.args.env.reward.lateral_occupancy_longitudinal_corridor
        self.lane_keeping_corridor = self.args.env.reward.lane_keeping_corridor

        self.ellipse_base_a = 20.0  # Base longitudinal radius (meters)
        self.ellipse_b = 3.5        # Lateral radius (slightly less than a lane width)
        self.speed_k = 0.5          # How much the ellipse stretches with ego speed
        
        # Calculate dynamic longitudinal radius based on ego speed
        ellipse_a = ellipse_base_a + speed_k * max(0.0, ego.velocity[0])

    def compute_reward(self, zones_occupied) -> float:
        # Example implementation, replace with actual logic
        self.sandwiching_reward = 0.0
        if zones_occupied.get("front") and zones_occupied.get("back"):
            self.sandwiching_reward = 1.0
        return self.sandwiching_reward


        if not is_release_phase:
            has_front_blocker = False
            has_back_blocker = False
            has_left_blocker = False
            has_right_blocker = False
            
            ellipse_reward_pool = 0.0

            for adv in adversaries:
                dx = adv.position[0] - ego.position[0]
                dy = adv.position[1] - ego.position[1]

                # Base Dense Reward: The Risk Ellipse
                d_ell = (dx / ellipse_a) ** 2 + (dy / ellipse_b) ** 2
                if d_ell <= 1.0:
                    # Provides a max of +0.5 base reward per vehicle if perfectly centered
                    ellipse_reward_pool += 0.18 * np.exp(1.0 - d_ell)

                # Track Multiplier Zones using your exact boundaries
                
                # Check Longitudinal Blocking (Front and Back within the same lane)
                if abs(dy) < lane_keeping_corridor:
                    if 0 < dx < longitudinal_occupancy_longitudinal_corridor:
                        has_front_blocker = True
                    elif -longitudinal_occupancy_longitudinal_corridor < dx < 0:
                        has_back_blocker = True
                
                # Check Parallel Cruising (Left and Right within the side corridor)
                elif abs(dx) < lateral_occupancy_longitudinal_corridor and abs(dy) >= lane_keeping_corridor:
                    if dy > 0: # Assuming positive dy is the right lane
                        has_right_blocker = True
                    elif dy < 0: # Assuming negative dy is the left lane
                        has_left_blocker = True

            # 3. Apply the Multipliers
            multiplier = 1.0
            
            # Classic longitudinal sandwich bonus
            if has_front_blocker and has_back_blocker:
                multiplier += 1.0  # Adds 1x to the multiplier (e.g., base * 2)
                
            # Parallel cruising lateral sandwich bonus
            if has_left_blocker and has_right_blocker:
                multiplier += 1.0  # Adds 1x to the multiplier (e.g., base * 3 if both happen)

            # Add the final stacked reward to the team
            team_reward += ellipse_reward_pool * multiplier
    
    def take_data_points(self):
        
        ttc_values = np.linspace(0.1, 30, int(30/0.1))  # From 0.1 to 30 seconds in increments of 0.1
        rewards = [self.compute_reward(ttc) for ttc in ttc_values]
        
        return ttc_values, rewards
import matplotlib.pyplot as plt
import numpy as np

from configs.configs import RLArgs

from highway_env.vehicle.kinematics import Vehicle


class RewardFunction:
    """
    Base class for all reward functions.
    """
    
    def __init__(self):
        self.args = RLArgs()
        self.release_distance = self.args.env.reward.release_distance
        self.dist_to_exit = sum(self.args.env.ends_m[:4])

        self.is_release_phase = False  # Default to blocking phase

    def check_phase(self, longitudinal_pos: float):
        """
        Determines whether the adversary is in the blocking phase or the release phase based on its position.
        """

        self.dist_to_exit = sum(self.args.env.ends_m[:4]) - longitudinal_pos  # Update distance to exit based on current position

        if self.dist_to_exit > self.release_distance:  # Assuming the release phase starts after the release distance
            self.is_release_phase = False
        else:
            self.is_release_phase = True


# ===== TTC REWARD FUNCTIONS ====== #

class RewardTTCFunction(RewardFunction):
    """
    A reward function that penalizes vehicles based on their Time-to-Collision (TTC) with other vehicles.

    This TTC reward function will be by default a difference of sigmoids model.
    """
    
    def __init__(self):
        super().__init__()
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

        # self.adv_adv_ttc_close_penalty = self.args.env.reward.adv_adv_ttc_close_penalty
        # self.adv_adv_ttc_near_m = self.args.env.reward.adv_adv_ttc_near_m
        # self.adv_adv_ttc_near_b = self.args.env.reward.adv_adv_ttc_near_b
        # self.adv_adv_ttc_far_m = self.args.env.reward.adv_adv_ttc_far_m
        # self.adv_adv_ttc_far_b = self.args.env.reward.adv_adv_ttc_far_b

        self.baseline_N = self.args.env.reward.advadv_baseline_N
        self.peak_P = self.args.env.reward.advadv_peak_P
        self.rise_slope_k1 = self.args.env.reward.advadv_rise_slope_k1
        self.decay_slope_k2 = self.args.env.reward.advadv_decay_slope_k2
        self.rise_shift_a = self.args.env.reward.advadv_rise_shift_a
        self.decay_shift_b = self.args.env.reward.advadv_decay_shift_b

    
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

        if not self.is_release_phase:
            self.baseline_N = self.args.env.reward.egoadv_blocking_baseline_N
            self.peak_P = self.args.env.reward.egoadv_blocking_peak_P
            self.rise_slope_k1 = self.args.env.reward.egoadv_blocking_rise_slope_k1
            self.decay_slope_k2 = self.args.env.reward.egoadv_blocking_decay_slope_k2
            self.rise_shift_a = self.args.env.reward.egoadv_blocking_rise_shift_a
            self.decay_shift_b = self.args.env.reward.egoadv_blocking_decay_shift_b
        elif self.is_release_phase:
            self.baseline_N = self.args.env.reward.egoadv_release_baseline_N
            self.peak_P = self.args.env.reward.egoadv_release_peak_P
            self.rise_slope_k1 = self.args.env.reward.egoadv_release_rise_slope_k1
            self.decay_slope_k2 = self.args.env.reward.egoadv_release_decay_slope_k2
            self.rise_shift_a = self.args.env.reward.egoadv_release_rise_shift_a
            self.decay_shift_b = self.args.env.reward.egoadv_release_decay_shift_b


        # self.adv_ego_ttc_close_penalty = self.args.env.reward.adv_ego_ttc_close_penalty
        # self.adv_ego_ttc_near_a = self.args.env.reward.adv_ego_ttc_near_a
        # self.adv_ego_ttc_near_h = self.args.env.reward.adv_ego_ttc_near_h
        # self.adv_ego_ttc_near_k = self.args.env.reward.adv_ego_ttc_near_k
        # self.adv_ego_ttc_far_m = self.args.env.reward.adv_ego_ttc_far_m
        # self.adv_ego_ttc_far_b = self.args.env.reward.adv_ego_ttc_far_b

        # self.adv_release_phase_m = self.args.env.reward.adv_release_phase_m
        # self.adv_release_phase_b = self.args.env.reward.adv_release_phase_b

    def compute_reward(self, ttc) -> float:

        
        self.adv_reward = -self.baseline_N + \
            (self.peak_P + self.baseline_N) / (1 + np.exp(-self.rise_slope_k1 * (ttc - self.rise_shift_a))) - \
            (self.peak_P) / (1 + np.exp(-self.decay_slope_k2 * (ttc - self.decay_shift_b)))



        # if not self.is_release_phase: # Still trying to block on the highway
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


class DistanceToEgoRewardFunction(RewardFunction):
    """
    A reward function that penalizes adversary vehicles based on their distance to the ego vehicle.
    The farther the adversary is from the ego vehicle, the higher the penalty!
    We want the adversaries to approach the ego as much as possible.

    Using a double sigmoid step down function.

    """

    def __init__(self):
        super().__init__()
        self.adv_reward = 0.0

        self.base1_c1 = self.args.env.reward.dist_base1_c1
        self.base2_c2 = self.args.env.reward.dist_base2_c2
        self.base3_c3 = self.args.env.reward.dist_base3_c3
        self.down1_a = self.args.env.reward.dist_down1_a
        self.down2_b = self.args.env.reward.dist_down2_b
        self.slope1_k1 = self.args.env.reward.dist_slope1_k1
        self.slope2_k2 = self.args.env.reward.dist_slope2_k2

    def compute_reward(self, distance_to_ego: float) -> float:

        if distance_to_ego < 0:
            raise ValueError("Distance to ego must be non-negative.")

        self.adv_reward = self.base3_c3 + \
            (self.base1_c1 - self.base2_c2) / (1 + np.exp(self.slope1_k1 * (distance_to_ego - self.down1_a))) + \
            (self.base2_c2 - self.base3_c3) / (1 + np.exp(self.slope2_k2 * (distance_to_ego - self.down2_b)))

        return self.adv_reward
    
    def take_data_points(self):
        """
        Generates a 1D array of distances to ego and their corresponding rewards for plotting.
        """
        distances = np.linspace(0, 50, 500)  # From 0 to 50 meters in increments of 0.1
        rewards = [self.compute_reward(d) for d in distances]
        
        return distances, rewards


# ====== SANDWICHING REWARD FUNCTION ====== #



class SandwichingRewardFunction(RewardFunction):
    """
    A reward function that encourages the adversary vehicles to sandwich the ego vehicle between them, 
    promoting a more challenging environment for the ego vehicle.

    Ellipse is made around the ego vehicle and adversaries are rewarded for being inside the ellipse and 
    depending on their proximity to the ego.

    Extra rewards for involved vehicles in a sandwich maneuver (front-back or left-right).

    """

    def __init__(self):
        super().__init__()

        self.long_corridor = self.args.env.reward.longitudinal_occupancy_longitudinal_corridor
        self.lat_corridor = self.args.env.reward.lateral_occupancy_longitudinal_corridor
        self.lane_keeping_corridor = self.args.env.reward.lane_keeping_corridor

        self.ellipse_base_a = self.args.env.reward.ellipse_base_a      
        self.ellipse_b = self.args.env.reward.ellipse_b            
        self.speed_k = self.args.env.reward.speed_k              
        
        self.base_proximity_reward = self.args.env.reward.base_proximity_reward
        self.sandwich_bonus = self.args.env.reward.sandwich_bonus

    def compute_reward(self, adversaries, ego) -> np.ndarray:
        """
        Calculates the individual sandwiching and proximity rewards for all adversaries.
        Returns an array of floats corresponding to each adversary's reward.
        """
        # Initialize an array of zeros for all adversaries
        indiv_rewards = np.zeros(len(adversaries), dtype=np.float32)

        if self.is_release_phase:
            return indiv_rewards # No sandwiching rewards during the release phase

        # Calculate dynamic longitudinal radius based on CURRENT ego speed
        ellipse_a = self.ellipse_base_a + self.speed_k * max(0.0, ego.velocity[0])

        # Track which specific agents (by index) are in which zone
        front_blockers = []
        back_blockers = []
        left_blockers = []
        right_blockers = []
        
        vehicles_in_ellipse = 0

        for i, adv in enumerate(adversaries):
            dx = adv.position[0] - ego.position[0]
            dy = adv.position[1] - ego.position[1]

            # 1. Baseline Proximity Reward (Dynamic Risk Ellipse)
            d_ell = (dx / ellipse_a) ** 2 + (dy / self.ellipse_b) ** 2
            if d_ell <= 1.0:
                indiv_rewards[i] += self.base_proximity_reward * np.exp(1.0 - d_ell)
                vehicles_in_ellipse += 1

            # 2. Zone Categorization for Sandwiching
            if abs(dy) < self.lane_keeping_corridor:
                if 0 < dx < self.long_corridor:
                    front_blockers.append(i)
                elif -self.long_corridor < dx < 0:
                    back_blockers.append(i)
                    
            elif abs(dx) < self.lat_corridor and abs(dy) >= self.lane_keeping_corridor:
                if dy > 0: # Assuming positive dy is the right lane
                    right_blockers.append(i)
                elif dy < 0: # Assuming negative dy is the left lane
                    left_blockers.append(i)

        # 3. Exponential Swarm Bonus (Team Reward broadcasted to all individuals)
        if vehicles_in_ellipse > 0:
            swarm_bonus = 0.05 * ((2 ** vehicles_in_ellipse) - 1)
            indiv_rewards += swarm_bonus 

        # 4. Targeted Sandwich Bonuses (Individual Reward)
        is_long_sandwich = len(front_blockers) > 0 and len(back_blockers) > 0
        is_lat_sandwich = len(left_blockers) > 0 and len(right_blockers) > 0

        # Only reward the specific agents participating in the maneuvers
        if is_long_sandwich:
            for idx in front_blockers + back_blockers:
                indiv_rewards[idx] += self.sandwich_bonus
                
        if is_lat_sandwich:
            for idx in left_blockers + right_blockers:
                indiv_rewards[idx] += self.sandwich_bonus

        return indiv_rewards

    def take_data_points(self, current_ego_speed: float = 25.0):
        """
        Generates a 2D grid of X and Y offsets to plot a spatial heatmap 
        of the risk ellipse and proximity rewards.
        """
        ellipse_a = self.ellipse_base_a + self.speed_k * max(0.0, current_ego_speed)
        
        # Create a grid around the ego vehicle (e.g., +/- 40m long, +/- 10m lat)
        x_values = np.linspace(-40, 40, 100)
        y_values = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_values, y_values)
        
        # Calculate the base proximity reward for every point on the grid
        D_ell = (X / ellipse_a) ** 2 + (Y / self.ellipse_b) ** 2
        
        # Apply the exponential reward only where D_ell <= 1.0, otherwise 0
        Rewards = np.where(D_ell <= 1.0, self.base_proximity_reward * np.exp(1.0 - D_ell), 0.0)
        
        return X, Y, Rewards
    
class SimpleRewardFunction(RewardFunction):
    """
    A simple reward function that returns a constant reward from the RLArgs configuration.
    Useful for static penalties (crashes, boundaries) and baseline comparisons.
    """
    
    def __init__(self):
        super().__init__()

    def get_reward(self, name: str) -> float:
        """
        Returns a constant reward scalar based on the variable name in EnvRewardArgs.
        Defaults to 0.0 if the variable name does not exist.
        """
        # getattr dynamically fetches the attribute from the dataclass object
        return getattr(self.args.env.reward, name, 0.0)


class LaneKeepingRewardFunction(RewardFunction):
    """
    Applies a continuous global cosine wave penalty across the entire highway 
    to encourage centering (0, 4, 8) and penalize lane straddling (2, 6, 10).
    """

    def __init__(self):
        super().__init__()
        self.vehicle_width = Vehicle.WIDTH
        self.lane_width_m = self.args.env.lane_width_m
        
        self.max_lane_penalty = self.args.env.reward.max_lane_penalty 
        self.boundary_hit_penalty = self.args.env.reward.boundary_hit_penalty  # Negative value for hitting boundaries

    def compute_reward(self, vehicle_y: float, left_boundary_y: float, right_boundary_y: float) -> float:
        reward = 0.0
        
        # 1. The Global Continuous Wave (Dense Penalty)
        # Perfectly handles 0, 4, 8 as peaks and 2, 6, 10 as troughs naturally
        amplitude = self.max_lane_penalty / 2.0
        cosine_penalty = amplitude * np.cos((2 * np.pi / self.lane_width_m) * vehicle_y) - amplitude
        
        reward += cosine_penalty

        # 2. The Sparse Physical Boundary Hit (from your snippet)
        # This acts as the physical "rumble strip" / guardrail penalty
        if left_boundary_y + (self.vehicle_width / 2) > vehicle_y or \
           right_boundary_y - (self.vehicle_width / 2) < vehicle_y:
            
            reward += self.boundary_hit_penalty

        return reward

    def take_data_points(self):
        """
        Plots the wave across multiple lanes to prove the continuity.
        """
        # Plot from -2 (left edge) to 10 (edge of 3rd lane)
        y_values = np.linspace(-2.0, 10.0, 500) 
        
        amplitude = self.max_lane_penalty / 2.0
        rewards = [amplitude * np.cos((2 * np.pi / self.lane_width_m) * y) - amplitude for y in y_values]
        
        return y_values, rewards



# ===== VISUALIZATION CLASS ====== #


    
class FunctionVisualizer:
    """
    A utility class to visualize the continuous RL reward functions.
    """
    def __init__(self, 
                 reward_ttc_ego_adv_function: RewardTTCEgoAdvFunction = None, 
                 reward_ttc_adv_adv_function: RewardTTCAdvAdvFunction = None, 
                 sandwiching_reward_function: SandwichingRewardFunction = None,
                 lane_keeping_reward_function: LaneKeepingRewardFunction = None):
        
        self.reward_ttc_ego_adv_function = reward_ttc_ego_adv_function
        self.reward_ttc_adv_adv_function = reward_ttc_adv_adv_function
        self.sandwiching_reward_function = sandwiching_reward_function
        self.lane_keeping_reward_function = lane_keeping_reward_function

    def plot_ttc_functions(self):
        """
        Plots the 1D line graphs for the Time-to-Collision (TTC) reward functions.
        """
        plt.figure(figsize=(10, 6))
        
        has_plots = False

        if self.reward_ttc_adv_adv_function:
            ttc_vals, rewards = self.reward_ttc_adv_adv_function.take_data_points()
            plt.plot(ttc_vals, rewards, label='Adv-Adv Reward vs TTC', color='orange', linewidth=2)
            has_plots = True
            
        if self.reward_ttc_ego_adv_function:
            ttc_vals, rewards = self.reward_ttc_ego_adv_function.take_data_points()
            plt.plot(ttc_vals, rewards, label='Ego-Adv Reward vs TTC', color='blue', linewidth=2)
            has_plots = True

        if not has_plots:
            print("No TTC reward functions provided to visualize.")
            return

        plt.title('Reward Function based on Time-to-Collision (TTC)')
        plt.xlabel('Time-to-Collision (seconds)')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    def plot_sandwiching_ellipse(self, ego_speed: float = 25.0):
        """
        Plots the 2D spatial heatmap for the dynamic risk ellipse around the ego vehicle.
        """
        if not self.sandwiching_reward_function:
            print("No SandwichingRewardFunction provided to visualize.")
            return

        # Grab the 2D grid data from the sandwiching function
        X, Y, Rewards = self.sandwiching_reward_function.take_data_points(current_ego_speed=ego_speed)
        
        plt.figure(figsize=(12, 6))
        
        # Plot the spatial heatmap using a contour plot
        # 'inferno' or 'viridis' are great colormaps for this
        cp = plt.contourf(X, Y, Rewards, levels=50, cmap='inferno')
        plt.colorbar(cp, label='Base Proximity Reward')
        
        # Mark the Ego Vehicle at the center of the grid
        plt.plot(0, 0, 'w*', markersize=15, markeredgecolor='black', label='Ego Vehicle (0,0)')
        
        # Formatting
        plt.title(f'Dynamic Risk Ellipse Heatmap (Ego Speed = {ego_speed} m/s)')
        plt.xlabel('Longitudinal Distance from Ego (meters)')
        plt.ylabel('Lateral Distance from Ego (meters)')
        plt.grid(color='white', linestyle='--', alpha=0.2)
        plt.legend()
        
        # Set aspect ratio to equal so the ellipse doesn't stretch artificially
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_lane_keeping_function(self):
        """
        Plots the 1D cosine wave for the continuous lane-keeping penalty.
        """
        if not self.lane_keeping_reward_function:
            print("No LaneKeepingRewardFunction provided to visualize.")
            return
            
        y_vals, rewards = self.lane_keeping_reward_function.take_data_points()
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_vals, rewards, label='Continuous Lane Keeping Penalty', color='green', linewidth=2)
        
        # Add visual markers for the lane centers to prove the math aligns
        # Assuming standard lane centers at 0, 4, 8
        for center in [0, 4, 8]:
            label = 'Lane Center' if center == 0 else ""
            plt.axvline(x=center, color='gray', linestyle='--', alpha=0.6, label=label)
            
        plt.title('Lane Keeping Penalty (Cosine Wave)')
        plt.xlabel('Lateral Position (meters)')
        plt.ylabel('Penalty')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Deduplicate legend labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        
        plt.show()

    def plot_all(self):
        """
        Convenience method to plot all available reward functions.
        """
        self.plot_ttc_functions()
        self.plot_sandwiching_ellipse()
        self.plot_lane_keeping_function()
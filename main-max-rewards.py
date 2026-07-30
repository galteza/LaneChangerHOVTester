import numpy as np
from scipy.optimize import differential_evolution

# Import your custom calculators and reward functions
from src.env.risk_calculators import PolygonTTCCalculator, THWCalculator
from src.env.reward_functions import (
    LaneKeepingRewardFunction,
    SpeedMatchingRewardFunction,
    RewardTTCEgoAdvFunction,
    DistanceToEgoRewardFunction,
    RewardTHWAdvEgoFunction,
    RewardTTCAdvAdvFunction,
    SandwichingRewardFunction
)

# Mock Vehicle class
class MockVehicle:
    """Mocks the highway-env Vehicle class for the calculators."""
    def __init__(self, x, y, vx, vy=0.0, heading=0.0):
        self.position = np.array([x, y], dtype=np.float32)
        self.velocity = np.array([vx, vy], dtype=np.float32)
        self.speed = np.linalg.norm(self.velocity)
        self.heading = heading
        self.LENGTH = 5.0
        self.WIDTH = 2.0
        self.crashed = False

# Reward classes
# rule 1 adv crash --> starts already at negative reward, makes everything smaller
lane_reward = LaneKeepingRewardFunction() # rule 2 lane keeping
# rule 3 no reversing --> starts at negative reward, makes everything smaller
adv_adv_ttc_reward = RewardTTCAdvAdvFunction() # rule 4 no adv-adv crash
dist_reward = DistanceToEgoRewardFunction() # rule 5a distance to ego
ttc_ego_reward = RewardTTCEgoAdvFunction() # rule 5b TTC to ego
thw_reward = RewardTHWAdvEgoFunction() # rule 5b THW to ego
speed_reward = SpeedMatchingRewardFunction() # rule 5c speed matching
sandwich_reward = SandwichingRewardFunction() # rule 5d sandwiching and swarm
# rule 5e ego success and ego fail --> reward at success, penalty at crash


# Force the phase to BLOCKING for consistency

dist_reward.set_phase("BLOCKING")
ttc_ego_reward.set_phase("BLOCKING")
sandwich_reward.phase = "BLOCKING" # Ensure sandwiching doesn't return 0 from RELEASE phase

# ==========================================
# 3. REWARD CALCULATION PIPELINE
# ==========================================
def calculate_platoon_reward(adv1, adv2, ego):
    """Mirrors the exact logic from MergeExitLaneHighway_Environment._reward()"""
    adversaries = [adv1, adv2]
    indiv_rewards = np.zeros(len(adversaries), dtype=np.float32)
    adv_ttc_rewards = []
    
    # Track individual breakdowns for printing later
    breakdowns = [{'lane': 0, 'adv_adv_ttc': 0, 'dist': 0, 'ttc': 0, 'thw': 0, 'speed': 0, 'sandwich': 0} for _ in range(2)]

    for i, adv in enumerate(adversaries):
        adv_reward = 0.0

        # RULE 2: Lane Keeping
        # Assuming 4m lanes[cite: 5], calculate boundaries dynamically based on y position
        lane_idx = int(adv.position[1] // 4)
        left_boundary = lane_idx * 4.0
        right_boundary = left_boundary + 4.0
        r_lane = lane_reward.compute_reward(adv.position[1], left_boundary, right_boundary)
        adv_reward += r_lane
        breakdowns[i]['lane'] = r_lane

        # RULE 4: Adv-Adv TTC
        other_adv = adversaries[1 - i] # Get the other adversary
        ttc_adv_adv = PolygonTTCCalculator.compute_ttc(adv, other_adv)
        r_adv_adv = adv_adv_ttc_reward.compute_reward(ttc_adv_adv)
        adv_reward += r_adv_adv
        breakdowns[i]['adv_adv_ttc'] = r_adv_adv

        # RULE 5a: Distance to Ego
        dist_to_ego = np.linalg.norm(adv.position - ego.position)
        r_dist = dist_reward.compute_reward(dist_to_ego)
        adv_reward += r_dist
        breakdowns[i]['dist'] = r_dist

        # RULE 5b: TTC and THW to Ego
        ttc_ego = PolygonTTCCalculator.compute_ttc(adv, ego)
        thw_ego = THWCalculator.compute_thw(adv, ego)
        
        r_ttc = ttc_ego_reward.compute_reward(ttc_ego)
        adv_ttc_rewards.append(r_ttc) # Saved specifically for sandwich multiplier
        adv_reward += r_ttc
        breakdowns[i]['ttc'] = r_ttc
        
        r_thw = thw_reward.compute_reward(thw_ego)
        adv_reward += r_thw
        breakdowns[i]['thw'] = r_thw

        # RULE 5c: Speed Matching (Applied if dx < 20m)
        if abs(ego.position[0] - adv.position[0]) < 20.0:
            r_speed = speed_reward.compute_reward(abs(adv.velocity[0] - ego.velocity[0]))
            adv_reward += r_speed
            breakdowns[i]['speed'] = r_speed

        indiv_rewards[i] += adv_reward

    # RULE 5d: Sandwiching & Swarm Bonus
    # The multiplier `adv_ttc_rewards[i] * indiv_rewards[i]` is handled inside this function
    r_sandwich_array = sandwich_reward.compute_reward(adversaries, ego, adv_ttc_rewards)
    
    indiv_rewards += r_sandwich_array
    
    for i in range(2):
        breakdowns[i]['sandwich'] = r_sandwich_array[i]

    return np.sum(indiv_rewards), indiv_rewards, breakdowns

# ==========================================
# 4. OPTIMIZER OBJECTIVE FUNCTION
# ==========================================
def objective_function(x):
    """
    x[0:3] = Adv 1 (x, y, vx)
    x[3:6] = Adv 2 (x, y, vx)
    """
    ego = MockVehicle(0.0, 4.0, 25.0) # Ego locked at (0, 4) going 25 m/s
    adv1 = MockVehicle(x[0], x[1], x[2])
    adv2 = MockVehicle(x[3], x[4], x[5])
    
    total_reward, _, _ = calculate_platoon_reward(adv1, adv2, ego)
    
    # Differential Evolution MINIMIZES by default, so we return negative
    return -total_reward

# ==========================================
# 5. EXECUTION AND BREAKDOWN
# ==========================================
if __name__ == "__main__":
    
    # BOUNDS: Format -> (x_pos, y_pos, vx_speed)
    # x_pos: -30m (behind) to +30m (ahead)
    # y_pos: 0m (right edge) to 4m (left edge of 1-lane highway)
    # vx_speed: 10 m/s to 36 m/s
    bounds = [
        (-30.0, 30.0), (0.0, 4.0), (10.0, 36.0), # Adv 1 limits
        (-30.0, 30.0), (0.0, 4.0), (10.0, 36.0)  # Adv 2 limits
    ]

    print("Running Multi-Agent Differential Evolution...")
    print("This may take 15-30 seconds depending on CPU power...")
    
    result = differential_evolution(objective_function, bounds, strategy='best1bin', popsize=15, tol=0.01)

    if result.success:
        optimal_x = result.x
        
        # Reconstruct the optimal state
        ego = MockVehicle(0.0, 2.0, 25.0)
        adv1 = MockVehicle(optimal_x[0], optimal_x[1], optimal_x[2])
        adv2 = MockVehicle(optimal_x[3], optimal_x[4], optimal_x[5])
        
        # Run one final time to extract the breakdown dictionaries
        total_sum, indiv_sums, breakdowns = calculate_platoon_reward(adv1, adv2, ego)
        
        print("\n" + "="*50)
        print("🎯 OPTIMAL KINEMATIC STATE FOUND")
        print("="*50)
        print(f"Ego Vehicle: Fixed at X: {ego.position[0]:.2f}m, Y: {ego.position[1]:.2f}m, Speed: {ego.velocity[0]:.2f}m/s")
        print(f"Adversary 1: Optimal at X: {adv1.position[0]:.2f}m, Y: {adv1.position[1]:.2f}m, Speed: {adv1.velocity[0]:.2f}m/s")
        print(f"Adversary 2: Optimal at X: {adv2.position[0]:.2f}m, Y: {adv2.position[1]:.2f}m, Speed: {adv2.velocity[0]:.2f}m/s")
        
        print("\n" + "="*50)
        print("📊 REWARD BREAKDOWN PER AGENT")
        print("="*50)
        
        for i in range(2):
            print(f"\n--- ADVERSARY {i+1} ---")
            print(f"Lane Keeping:         {breakdowns[i]['lane']:>8.3f}")
            print(f"Adv-Adv TTC:          {breakdowns[i]['adv_adv_ttc']:>8.3f}")
            print(f"Distance to Ego:      {breakdowns[i]['dist']:>8.3f}")
            print(f"TTC to Ego:           {breakdowns[i]['ttc']:>8.3f}")
            print(f"THW to Ego:           {breakdowns[i]['thw']:>8.3f}")
            print(f"Speed Matching:       {breakdowns[i]['speed']:>8.3f}")
            print(f"Sandwich/Swarm Bonus: {breakdowns[i]['sandwich']:>8.3f}")
            print(f"SUBTOTAL:             {indiv_sums[i]:>8.3f}")
            
        print("\n" + "="*50)
        print(f"🏆 ABSOLUTE MAXIMUM THEORETICAL REWARD: {total_sum:.3f}")
        print("="*50)
    else:
        print("Optimization failed:", result.message)
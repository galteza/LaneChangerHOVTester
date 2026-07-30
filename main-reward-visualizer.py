from src.env.reward_functions import (
    RewardTTCAdvAdvFunction, 
    RewardTTCEgoAdvFunction, 
    RewardTHWAdvEgoFunction,
    SandwichingRewardFunction,
    LaneKeepingRewardFunction,
    AdversarialCrashPenalty,
    SpeedMatchingRewardFunction,
    DistanceToEgoRewardFunction,

    FunctionVisualizer,
)

import os
import matplotlib.pyplot as plt
from math import pi

def generate_spider_plot(output_dir="thesis_figures/radar_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. DEFINE THE REWARD METRICS (The spokes of the web)
    # These should match the individual reward components your wrapper tracks
    categories = [
        'Ego-Adv TTC Risk', 
        'Adv-Adv Safety', 
        'Sandwich Bonus', 
        'Lane Keeping', 
        'Distance to Ego', 
        'Speed Matching'
    ]
    N = len(categories)

    # 2. MOCK CLUSTER DATA (Replace this with your actual cluster averages!)
    # Values should ideally be normalized between 0 and 1 (or 0 to 100) for clean comparison
    data = {
        'Cluster 0 (Brake-Check)': [0.9, 0.8, 0.1, 0.7, 0.9, 0.4],
        'Cluster 1 (Tailgate)':    [0.7, 0.9, 0.1, 0.8, 0.8, 0.9],
        'Cluster 2 (Sandwich)':    [0.95, 0.6, 1.0, 0.3, 0.95, 0.8]
    }

    # 3. MATHEMATICAL SETUP FOR POLAR PLOT
    # Calculate the angle for each spoke (divide the circle into N equal slices)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop so the shape connects back to the start

    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Set the first spoke to the top (12 o'clock)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw the category labels around the web
    plt.xticks(angles[:-1], categories)

    # Set the radial limits (e.g., 0 to 1 for normalized data)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "MAX"], color="grey", size=12)
    plt.ylim(0, 1.0)

    # 4. PLOT EACH CLUSTER'S TACTICAL PROFILE
    colors = ['#1f77b4', '#ff7f0e', '#d62728'] # Blue, Orange, Red
    
    for i, (cluster_name, values) in enumerate(data.items()):
        # Close the loop for the data values
        values_closed = values + values[:1]
        
        # Plot the outline
        ax.plot(angles, values_closed, color=colors[i], linestyle='solid', label=cluster_name)
        
        # Fill the shape with high transparency so overlapping shapes remain visible
        ax.fill(angles, values_closed, color=colors[i], alpha=0.15)

    # 5. FORMATTING & EXPORT
    plt.title('Multi-Objective Reward Distribution by Swarm Tactic', y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "tactical_spider_plot.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎥 Spider plot saved to: {filepath}")

if __name__ == "__main__":
    # Create instances of the reward functions
    adv_adv_reward_function = RewardTTCAdvAdvFunction()
    ego_adv_reward_function = RewardTTCEgoAdvFunction()
    sandwich_reward_function = SandwichingRewardFunction()
    lane_keeping_reward_function = LaneKeepingRewardFunction()
    adversarial_crash_penalty = AdversarialCrashPenalty()
    speed_matching_reward_function = SpeedMatchingRewardFunction()
    thw_adv_ego_reward_function = RewardTHWAdvEgoFunction()  # New THW-based reward function
    distance_to_ego_reward_function = DistanceToEgoRewardFunction()  # New distance-to-ego reward function

    # Pass them into your new visualizer
    visualizer = FunctionVisualizer(
        reward_ttc_adv_adv_function=adv_adv_reward_function,
        reward_ttc_ego_adv_function=ego_adv_reward_function,
        sandwiching_reward_function=sandwich_reward_function,
        lane_keeping_reward_function=lane_keeping_reward_function,
        adversarial_crash_penalty=adversarial_crash_penalty,
        speed_matching_reward_function=speed_matching_reward_function,
        reward_thw_adv_ego_function=thw_adv_ego_reward_function,
        distance_to_ego_reward_function=distance_to_ego_reward_function,
    )

    # Plot everything
    visualizer.plot_all()

    generate_spider_plot(output_dir="thesis_figures/radar_plots")
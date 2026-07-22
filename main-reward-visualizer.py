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
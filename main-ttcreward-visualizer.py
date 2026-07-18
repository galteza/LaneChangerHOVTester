from src.env.reward_functions import (
    RewardTTCAdvAdvFunction, 
    RewardTTCEgoAdvFunction, 
    SandwichingRewardFunction,
    LaneKeepingRewardFunction,
    FunctionVisualizer
)

if __name__ == "__main__":
    # Create instances of the reward functions
    adv_adv_reward_function = RewardTTCAdvAdvFunction()
    ego_adv_reward_function = RewardTTCEgoAdvFunction()
    sandwich_reward_function = SandwichingRewardFunction()
    lane_keeping_reward_function = LaneKeepingRewardFunction()

    # Pass them into your new visualizer
    visualizer = FunctionVisualizer(
        reward_ttc_adv_adv_function=adv_adv_reward_function,
        reward_ttc_ego_adv_function=ego_adv_reward_function,
        sandwiching_reward_function=sandwich_reward_function,
        lane_keeping_reward_function=lane_keeping_reward_function
    )

    # Plot everything
    visualizer.plot_all()
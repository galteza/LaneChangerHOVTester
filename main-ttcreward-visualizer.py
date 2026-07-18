from matplotlib import pyplot as plt
from src.env.reward_functions import RewardTTCAdvAdvFunction, RewardTTCEgoAdvFunction

if __name__ == "__main__":
    # Create instances of the reward functions
    adv_adv_reward_function = RewardTTCAdvAdvFunction()
    ego_adv_reward_function = RewardTTCEgoAdvFunction()

    # Graph the reward functions
    ttc_values_adv_adv, rewards_adv_adv = adv_adv_reward_function.take_data_points()
    ttc_values_ego_adv, rewards_ego_adv = ego_adv_reward_function.take_data_points()

    plt.figure(figsize=(10, 6))
    plt.plot(ttc_values_adv_adv, rewards_adv_adv, label='Adv-Adv Reward vs TTC')
    plt.plot(ttc_values_ego_adv, rewards_ego_adv, label='Ego-Adv Reward vs TTC')
    plt.title('Reward Function based on Time-to-Collision (TTC)')
    plt.xlabel('Time-to-Collision (seconds)')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()
    plt.show()
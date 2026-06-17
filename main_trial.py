import gymnasium as gym
import time

from stable_baselines3 import SAC
import numpy as np 

from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, FlattenWrapper

# Create highway environment

base_env = MergeExitLaneHighway_Environment(render_mode="human")

env = FlattenWrapper(base_env) # takes the base_env, clones it to make a version with a flattened observation space

# Import SAC from SB3

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_platoon")

del model

model = SAC.load("sac_platoon")

obs, info = env.reset()
done, truncated = False, False

# Main 

while True:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, truncated, info = env.step(action)

    env.render()

    time.sleep(0.2)
    
    if done or truncated:
        obs, info = env.reset()


# env = MergeExitLaneHighway_Environment()
# env.render_mode = "human" # Tells Gymnasium to prepare a visual window
# env.reset()

# # Run a loop to step the physics engine forward in time
# for _ in range(200):
#     # Action '1' is the IDLE action (tells the car to just cruise forward)
#     # If your gym version is older, you might only get 4 return values instead of 5
#     obs, reward, done, truncated, info = env.step(1)

#     env.render()

#     time.sleep(0.2) # <--- Add this to slow down the frame rate

#     # If the car crashes or finishes the route, reset the map
#     if done or truncated:
#         env.reset()
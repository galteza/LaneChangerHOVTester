import gymnasium as gym
import highway_env
import yaml
import time

from highway_env.vehicle.behavior import IDMVehicle

class Highway_Environment:
    def __init__(self, env_params):
        self.env_params = env_params
        self.env_id = env_params['env_id']
        self.lanes_count = env_params['lanes_count']
        self.vehicles_count = env_params['vehicles_count']
        self.controlled_vehicles = env_params['controlled_vehicles']
        self.duration = env_params['duration']
        self.simulation_frequency = env_params['simulation_frequency']
        self.policy_frequency = env_params['policy_frequency']

        self.action_params = env_params['action']
        self.observation_params = env_params['observation']

        self.env = None

        self._setup_env()

    def _setup_env(self):
        self.env = gym.make(
            self.env_id, 
            render_mode = "human",
            config = self.env_params
            )

        return self.env
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def _make_stock_vehicles(self) -> None:

        
        self.controlled_vehicles = []

        ego = IDMVehicle(
            self.road,
            speed = 25,
        )

        self.controlled_vehicles.append(ego)

        self.vehicle = self.controlled_vehicles[0]

with open("../../configs/simenv_params.yaml", "r") as f:
    simenv_params = yaml.safe_load(f)

env = Highway_Environment(simenv_params['env_params'])
env.render_mode = "human" # Tells Gymnasium to prepare a visual window
env.reset()

# Run a loop to step the physics engine forward in time
for _ in range(200):
    # Action '1' is the IDLE action (tells the car to just cruise forward)
    # If your gym version is older, you might only get 4 return values instead of 5
    obs, reward, done, truncated, info = env.step((1,))

    # This is the magic command that physically draws the Pygame window!
    env.render()

    time.sleep(0.05) # <--- Add this to slow down the frame rate

    # If the car crashes or finishes the route, reset the map
    if done or truncated:
        env.reset()
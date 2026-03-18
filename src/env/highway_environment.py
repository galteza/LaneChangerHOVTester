import gymnasium as gym
import highway_env

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
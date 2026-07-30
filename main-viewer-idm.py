import gymnasium as gym
import time
from highway_env.vehicle.behavior import IDMVehicle
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, Wrapper_MergeExitLaneHighway_Environment

def inject_idm_only(env_unwrapped):
    # 1. Clear whatever vehicles the environment spawned by default
    env_unwrapped.road.vehicles = []

    # 2. Get the starting lane based on your provided route
    ego_lane = env_unwrapped.road.network.get_lane(("j", "k", 0))

    # 3. Inject your specific IDM vehicle
    ego = IDMVehicle(
        env_unwrapped.road,
        ego_lane.position(0, 0),
        speed=25,
        route=[("j", "k", 0), ("k", "b", 0), ("b", "c", 0), ("c", "d", 0), ("d", "e", 0), ("e", "l", 0), ("l", "m", 0)]
    )
    env_unwrapped.road.vehicles.append(ego)
    
    return ego

if __name__ == "__main__":
    # Setup Environment
    base_env = MergeExitLaneHighway_Environment(render_mode="human")
    env = Wrapper_MergeExitLaneHighway_Environment(base_env)

    obs, info = env.reset()
    
    # Override the environment with our IDM vehicle
    ego = inject_idm_only(base_env.unwrapped)

    # Point the camera at the new IDM vehicle
    if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
        base_env.unwrapped.viewer.observer_vehicle = ego

    print("Running IDM-Only Environment... press Ctrl+C to stop.")
    try:
        while True:
            # We don't have an RL agent, so we pass a dummy action.
            # IDMVehicles step themselves based on their internal longitudinal/lateral models.
            dummy_action = env.action_space.sample() 
            
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            env.render()
            
            time.sleep(0.02)
            
            if terminated or truncated:
                print("Episode finished. Resetting environment...")
                obs, info = env.reset()
                
                # Re-apply the override after every reset
                ego = inject_idm_only(base_env.unwrapped)

                if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
                    base_env.unwrapped.viewer.observer_vehicle = ego
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        env.close()
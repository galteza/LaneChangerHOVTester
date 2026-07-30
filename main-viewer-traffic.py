import gymnasium as gym
import time
from highway_env.vehicle.behavior import IDMVehicle
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, Wrapper_MergeExitLaneHighway_Environment
from src.env.risk_calculators import PolygonTTCCalculator

def inject_idm_and_traffic(env_unwrapped):
    # Clear default vehicles
    env_unwrapped.road.vehicles = []

    # Inject your specific IDM ego vehicle
    ego_lane = env_unwrapped.road.network.get_lane(("j", "k", 0))
    ego = IDMVehicle(
        env_unwrapped.road,
        ego_lane.position(0, 0),
        speed=25,
        route=[("j", "k", 0), ("k", "b", 0), ("b", "c", 0), ("c", "d", 0), ("d", "e", 0), ("e", "l", 0), ("l", "m", 0)]
    )
    env_unwrapped.road.vehicles.append(ego)
    
    # CRITICAL: Point the environment's ego reference to our new vehicle 
    # so the native truncation and reward logic tracks the right car.
    env_unwrapped.ego = ego

    # Inject exactly 2 random background IDM vehicles
    for _ in range(2):
        random_car = IDMVehicle.create_random(env_unwrapped.road, speed=20)
        env_unwrapped.road.vehicles.append(random_car)
    
    return ego

if __name__ == "__main__":
    # Setup Environment
    base_env = MergeExitLaneHighway_Environment(render_mode="human")
    env = Wrapper_MergeExitLaneHighway_Environment(base_env)

    obs, info = env.reset()
    
    # Override the environment with our IDM vehicle + 2 cars
    ego = inject_idm_and_traffic(base_env.unwrapped)

    if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
        base_env.unwrapped.viewer.observer_vehicle = ego

    print("Running IDM + 2 Random Cars with TTC colors... press Ctrl+C to stop.")
    try:
        while True:
            # Dummy action to step the environment forward
            dummy_action = env.action_space.sample() 
            
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
            # --- DYNAMIC COLOR LOGIC ---
            # Check TTC between ego and all other vehicles on the road
            for vehicle in base_env.unwrapped.road.vehicles:
                if vehicle is not ego:
                    ttc = PolygonTTCCalculator.compute_ttc(vehicle, ego)
                    if ttc <= 4.0:
                        vehicle.color = (128, 0, 128)  # Purple color for danger
                        ego.color = (128, 0, 128)
                    else:
                        vehicle.color = None  # Reset to default
                        ego.color = None
            # ---------------------------

            # Manually check for goal truncation just in case the wrapper misses it
            if ego.lane_index[0] == 'l' and ego.lane_index[1] == 'm':
                print("Goal reached!")
                truncated = True

            env.render()
            time.sleep(0.02)
            
            if terminated or truncated:
                print("Episode finished. Resetting environment...")
                obs, info = env.reset()
                
                # Re-apply the override after every reset
                ego = inject_idm_and_traffic(base_env.unwrapped)

                if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
                    base_env.unwrapped.viewer.observer_vehicle = ego
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        env.close()
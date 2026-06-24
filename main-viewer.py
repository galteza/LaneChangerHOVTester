import gymnasium as gym
import torch
import time
import os

from src.agents.platoon_masac import MASACRL
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, Wrapper_MergeExitLaneHighway_Environment
from configs.configs import RLArgs

# IMPORTANT: You need to import or recreate the 'args' object you used during training
# so that MASACRL initializes with the correct dimensions (obs_dim, action_dim, etc.)
# from src.config import get_args  <-- Adjust this to match your codebase

if __name__ == "__main__":
    # 1. Setup Environment
    base_env = MergeExitLaneHighway_Environment(render_mode="human")
    env = Wrapper_MergeExitLaneHighway_Environment(base_env)

    # 2. Setup Device (Render on CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize Model Architecture
    print("Initializing MASACRL architecture...")
    args = RLArgs() # Uncomment and set this up so your class gets its parameters
    model = MASACRL(args, device) 

    # 4. Define Paths to your specific .pth files
    run_dir = "/home/gabrielalteza/LaneChangerHOVTester/runs/merge_exit_highway__configs__1__1781686594"
    actor_path = os.path.join(run_dir, "actor_final.pth")
    # You only strictly need the actor for rendering, but you can load critics if you want to log Q-values!
    
    # 5. Load the trained weights into your custom actor
    print(f"Loading trained weights from {actor_path}...")
    model.actor.load_state_dict(torch.load(actor_path, map_location=device))
    model.actor.eval() # Set network to evaluation mode (turns off dropout/batchnorm updates if any)

    obs, info = env.reset()

    # env.unwrapped.vehicle = env.unwrapped.road.vehicles[0]

    if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
        # Manually point the viewer's target to the victim vehicle object directly
        # assuming you defined 'victim' somewhere accessible, or from road vehicles:
        base_env.unwrapped.viewer.observer_vehicle = base_env.unwrapped.road.vehicles[0]

    print("Running model through environment... press Ctrl+C to stop.")
    try:
        while True:
            # 6. Use your custom get_action method with the deterministic flag we added!
            # This ensures the swarm executes the master strategy without exploration noise.
            action = model.get_action(obs) 
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # Adjust sleep time if the simulation renders too fast or too slow
            time.sleep(0.02)
            
            if terminated or truncated:
                print("Episode finished. Resetting environment...")
                obs, info = env.reset()

                if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
                    base_env.unwrapped.viewer.observer_vehicle = base_env.unwrapped.road.vehicles[0]
                # env.unwrapped.vehicle = env.unwrapped.road.vehicles[0]
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        env.close()
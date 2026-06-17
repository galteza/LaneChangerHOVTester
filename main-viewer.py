import gymnasium as gym
from src.agents.platoon_masac import MASACRL
import time

from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, Wrapper_MergeExitLaneHighway_Environment

if __name__ == "__main__":
    base_env = MergeExitLaneHighway_Environment(render_mode="human")
    env = Wrapper_MergeExitLaneHighway_Environment(base_env)

    model_path = "runs/merge_exit_highway__configs__1__1781588400/best_model.zip" 

    print("Loading trained MASACRL model...")
    model = MASACRL.load(model_path, env=env)
    obs, info = env.reset()
    done = False

    print("Running model through environment... press Ctrl+C to stop.")
    while True:
        action, _states = model.predict(obs, deterministic=True) # takes best-learned actions, not exploration actions
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)
        if terminated or truncated:
            obs, info = env.reset()
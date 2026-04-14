import numpy as np
import time

from matplotlib import pyplot as plt
import yaml

from src.agents.decoupled_lanechanger_mpc import DecoupledLaneChanger_MPC
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment

# === 0. PREPARING HELPER FUNCTIONS

def lateral_radar_sweep(env, target_longitudinal_s, current_lateral_position):
    """
    Sweeping the road network at given longitudinal position and returning the boundaries
    of the block that the car is currently on.
    """

    active_lanes = []
    all_lanes = env.unwrapped.road.network.lanes_list()

    for lane in all_lanes:
        target_longitudinal_s_converted, _ = lane.local_coordinates(np.array([target_longitudinal_s, 0])) # Automatically converts to local coordinates for that lane
        error_buffer_m = 1
        if -error_buffer_m + 0 <= target_longitudinal_s_converted <= lane.length + error_buffer_m:
            _, lane_center_y = lane.position(target_longitudinal_s_converted, 0)
            half_width = lane.width / 2.0
            
            active_lanes.append({
                'left': lane_center_y - half_width,
                'right': lane_center_y + half_width
            })

    if not active_lanes:
        return current_lateral_position - 2, current_lateral_position + 2
    
    # Sort lanes

    active_lanes = sorted(active_lanes, key=lambda x: x['left'])

    # Find lane in closest match to the one car is on

    closest_lane_idx = 0
    min_dist = float('inf')

    for i, lane in enumerate(active_lanes):
        lane_center = (lane['left'] + lane['right']) / 2
        dist = abs(current_lateral_position - lane_center)
        if dist < min_dist:
            min_dist = dist
            closest_lane_idx = 1

    # Expand boundaries from car's current lane

    d_min = active_lanes[closest_lane_idx]['left']
    d_max = active_lanes[closest_lane_idx]['right']

    gap_tolerance = 0.5

    for i in range(closest_lane_idx-1, -1, -1):
        if abs(active_lanes[i]['right'] - d_min) <= gap_tolerance:
            d_min = active_lanes[i]['right']
        else:
            break
    
    for i in range(closest_lane_idx+1, len(active_lanes)):
        if abs(active_lanes[i]['left'] - d_max) <= gap_tolerance:
            d_max = active_lanes[i]['right']
        else:
            break
    
    return d_min, d_max

# === 0. PREPARING CONFIGS & ARRAYS FOR GRAPH DISPLAY ===

with open("configs/simenv_params.yaml", "r") as f:
    simenv_params = yaml.safe_load(f)

history_time = []
history_actual_d = []
history_actual_s = []

t = 0.0


# === 1. INITIALIZE CUSTOM ENVIRONMENT ===
highway_environment = MergeExitLaneHighway_Environment()
highway_environment.configure(simenv_params['env_params'])
highway_environment.render_mode = "human"

sutlanechanger_mpc = DecoupledLaneChanger_MPC(
    simenv_params['env_params'],
    simenv_params['vehiclemodel_params'],
    simenv_params['sutlanechanger_mpc_params'],
)

obs, info = highway_environment.reset()

# === 2. RUN THE SIMULATION / EPISODES ===

done = False
truncated = False

try:
    while not (done or truncated):

        multi_agent_actions = []
        
        for i, agent_obs_matrix in enumerate(obs):
            agent_self_state = agent_obs_matrix[0]
            current_agent_s = agent_self_state[1]       # longitudinal distance
            current_agent_d = agent_self_state[2]       # lateral distance
            current_agent_vx = agent_self_state[3]      # longitudinal velocity
            current_agent_vy = agent_self_state[4]      # lateral velocity
            current_agent_psi = agent_self_state[5]     # heading angle

            current_agent_v = current_agent_vx * np.cos(current_agent_psi) + current_agent_vy * np.sin(current_agent_psi)
            
            obstacle_matrix = agent_obs_matrix[1:]

            # CAR 0: SUT LANE CHANGER (Heading to exit)
            if i == 0:
                x0_extracted_structure = sutlanechanger_mpc.mpc.x0

                x0_extracted_structure['s'] = current_agent_s
                x0_extracted_structure['d'] = current_agent_d
                x0_extracted_structure['psi'] = current_agent_psi
                x0_extracted_structure['v'] = current_agent_v

                sutlanechanger_mpc.update_targets_and_obstacles_in_sut_mpc(
                    target_d=-4.0,
                    target_v=simenv_params['sutlanechanger_mpc_params']['target_v'],
                    observation_matrix=obstacle_matrix
                )

                # CALCULATE NEXT OPTIMAL STEP USING MPC

                u0 = sutlanechanger_mpc.make_step(x0_extracted_structure)
                u0_extracted_structure = sutlanechanger_mpc.mpc.u0

                # EXTRACTING STEPS FOR RENDERING
                steering_rad = float(u0_extracted_structure['steering'])
                accel_ms2 = float(u0_extracted_structure['accel'])

                # RECORDING DISTANCE FOR LATER PLOTTING

                history_actual_d.append(current_agent_d)
                history_actual_s.append(current_agent_s)

            # ADVERSARIAL PLATOON OF CRUISERS (Staying on highway)

            else:
                x0_extracted_structure = sutlanechanger_mpc.mpc.x0 # change this later
                
                x0_extracted_structure['s'] = current_agent_s
                x0_extracted_structure['d'] = current_agent_d
                x0_extracted_structure['psi'] = current_agent_psi
                x0_extracted_structure['v'] = current_agent_v

                sutlanechanger_mpc.update_targets_and_obstacles_in_sut_mpc( # change this later too
                    target_d = current_agent_d, # Tell them to stay in whatever lane they spawned in!
                    target_v = 25.0,          # Standard highway cruising speed
                    observation_matrix = obstacle_matrix
                )

                u0 = sutlanechanger_mpc.make_step(x0_extracted_structure) # change this one as well
                u0_extracted_structure = sutlanechanger_mpc.mpc.u0 # change this one last

                steering_rad = float(u0_extracted_structure['steering'])
                accel_ms2 = float(u0_extracted_structure['accel'])


            normalized_accel = accel_ms2 / simenv_params['vehiclemodel_params']['max_long_accel_ms2']
            normalized_steer = steering_rad / simenv_params['vehiclemodel_params']['max_steer_rad']

            agent_action = np.array([normalized_accel, normalized_steer], dtype=np.float32)
            multi_agent_actions.append(agent_action)
        
        obs, reward, done, truncated, info = highway_environment.step(tuple(multi_agent_actions)) # converting into tuple

        highway_environment.render()
        #time.sleep(0.05)

        history_time.append(t)
        t += 0.1

except KeyboardInterrupt:
    print("\n[INFO] You pressed Ctrl+C! Stopping the simulation and generating the graph...")


# === 3. PLOTTING RESULTS ===

# plt.title('Highway Environment')
# plt.imshow(highway_environment.render())

plt.figure(figsize=(10, 5))
plt.plot(history_time, history_actual_d, 'b-', label='Lateral Distance (d)')
plt.title('Lateral Distance vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Lateral Distance (m)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_time, history_actual_s, 'r-', label='Longitudinal Distance (s)')
plt.title('Longitudinal Distance vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Longitudinal Distance (m)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history_actual_s, history_actual_d, 'g-', label='Trajectory')
plt.title('Trajectory')
plt.xlabel('Longitudinal Distance (m)')
plt.ylabel('Lateral Distance (m)')
plt.legend()
plt.grid(True)
plt.show()

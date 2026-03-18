import numpy as np
from matplotlib import pyplot as plt
import yaml
from src.agents.sutlanechanger_mpc import SUTLaneChanger_MPC
from src.env.highway_environment import Highway_Environment

with open("configs/simenv_params.yaml", "r") as f:
    simenv_params = yaml.safe_load(f)

highway_environment = Highway_Environment(simenv_params['env_params'])
sutlanechanger_mpc = SUTLaneChanger_MPC(simenv_params['env_params'], simenv_params['vehiclemodel_params'], simenv_params['sutlanechanger_mpc_params'])

obs, info = highway_environment.reset()

# === INITIALIZE MATRIX FOR GRAPH DISPLAY
history_time = []
history_actual_d = []
history_actual_s = []

t = 0.0

# === CALCULATION FOR ONE EPISODE

done = False
try:
    while not done or truncated:
        # GATHER CURRENT DATA ON LANE CHANGER SUT'S OBSERVATIONS
        sut_obs = obs[0]

        # DEFINE CURRENT STATE OF LANE CHANGER SUT
        sut_self_state = sut_obs[0]

        current_sut_s = sut_self_state[1]       # longitudinal position
        current_sut_d = sut_self_state[2]       # lateral distance
        current_sut_vx = sut_self_state[3]      # longitudinal velocity
        current_sut_vy = sut_self_state[4]      # lateral velocity
        current_sut_psi = sut_self_state[5]     # heading angle

        current_sut_v = current_sut_vx * np.cos(current_sut_psi) + current_sut_vy * np.sin(current_sut_psi)

        # x0 = np.array([current_sut_s, current_sut_d, current_sut_psi, current_sut_v]).reshape(-1, 1)
        
        x0_extracted_structure = sutlanechanger_mpc.mpc.x0

        x0_extracted_structure['s'] = current_sut_s
        x0_extracted_structure['d'] = current_sut_d
        x0_extracted_structure['psi'] = current_sut_psi
        x0_extracted_structure['v'] = current_sut_v

        # DEFINE CURRENT STATE OF SURROUNDING (OBSTACLE) CRUISERS
        obstacle_matrix = sut_obs[1:]
        
        current_target_d = simenv_params['sutlanechanger_mpc_params']['target_d']

        sutlanechanger_mpc.update_targets_and_obstacles_in_sut_mpc(
            target_d=(simenv_params['env_params']['lanes_count']-1)*4.0,
            target_v=simenv_params['sutlanechanger_mpc_params']['target_v'],
            observation_matrix=obstacle_matrix
        )

        # CALCULATE NEXT OPTIMAL STEP USING MPC

        u0 = sutlanechanger_mpc.make_step(x0_extracted_structure)

        u0_extracted_structure = sutlanechanger_mpc.mpc.u0

        # EXTRACTING STEPS FOR RENDERING
        steering_rad = float(u0_extracted_structure['steering'])
        accel_ms2 = float(u0_extracted_structure['accel'])

        normalized_accel = accel_ms2 / simenv_params['vehiclemodel_params']['max_long_accel_ms2']
        normalized_steer = steering_rad / simenv_params['vehiclemodel_params']['max_steer_rad']

        sut_action = np.array([normalized_accel, normalized_steer], dtype=np.float32)
        dummy_action = np.array([0.0, 0.0], dtype=np.float32)

        multi_agent_actions = tuple([sut_action] + [dummy_action] * (simenv_params['env_params']['controlled_vehicles'] - 1))
        
        obs, reward, done, truncated, info = highway_environment.step(multi_agent_actions)

        highway_environment.render()

        # RECORDING DISTANCE FOR LATER PLOTTING

        history_time.append(t)
        history_actual_d.append(current_sut_d)
        history_actual_s.append(current_sut_s)

        t += 0.1

        # PYGAME VIEWER

        # viewer = highway_environment.unwrapped.viewer

        # if viewer is not None:
        #     longitudinal_margin = simenv_params['vehiclemodel_params']['safety_margin']['longitudinal_margin']
        #     lateral_margin = simenv_params['vehiclemodel_params']['safety_margin']['lateral_margin']

        #     sut_x 

except KeyboardInterrupt:
    print("\n[INFO] You pressed Ctrl+C! Stopping the simulation and generating the graph...")


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

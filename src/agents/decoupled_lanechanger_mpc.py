import numpy as np
import do_mpc
import casadi as ca

class DecoupledLaneChanger_MPC:
    def __init__(self, env_params, vehiclemodel_params, sutlanechanger_mpc_params):
        self.env_params = env_params
        
        # Initialize both MPCs (This will compile the CasADi backend immediately)
        self.long_mpc, self.long_model, self.long_tvp = self.create_longitudinal_mpc()
        self.lat_mpc, self.lat_model, self.lat_tvp = self.create_lateral_mpc()
        
        self.prediction_horizon = 20
        self.dt = 0.1

    def make_step(self, obs_matrix, target_gap_s, target_gap_v, target_lane_d):
        """
        Executes one step of the decoupled control architecture.
        Returns an array: [optimal_acceleration, optimal_steering]
        """
        # Extract Ego state from Observation Matrix
        ego_state = obs_matrix[0]
        current_s = ego_state[1] 
        current_d = ego_state[2]
        current_v = ego_state[3]
        current_psi = ego_state[5]
        
        # --- 1. SOLVE LONGITUDINAL (The Gas Pedal) ---
        for k in range(self.prediction_horizon):
            self.long_tvp['_tvp', k, 's_ref'] = target_gap_s + (target_gap_v * k * self.dt)
            self.long_tvp['_tvp', k, 'v_ref'] = target_gap_v
            
            # TODO: Replace with dynamic lag/lead car positions from your RL Environment
            self.long_tvp['_tvp', k, 's_min'] = 0.0 
            self.long_tvp['_tvp', k, 's_max'] = 1000.0 

        # We do not call set_tvp_fun here! The lambda attached during setup reads the updated self.long_tvp automatically.
        long_x0 = np.array([current_s, current_v]).reshape(-1, 1)
        u_long = self.long_mpc.make_step(long_x0)
        optimal_accel = float(u_long[0])

        # --- 2. THE BRIDGE (Passing Data) ---
        # Extract the exact predicted speed array from the longitudinal solver
        predicted_s_array = self.long_mpc.data['_x', 's'][-1]
        predicted_v_array = self.long_mpc.data['_x', 'v'][-1]

        # --- 3. SOLVE LATERAL (The Steering Wheel) ---
        # Dynamic lane change length (Approx 4 seconds to complete)
        lane_change_length = max(current_v * 4.0, 20.0) # 20m absolute minimum failsafe
        
        for k in range(self.prediction_horizon):
            future_s = float(predicted_s_array[k])
            future_v = float(predicted_v_array[k])
            
            # Generate the Ghost Car trajectory point for this future step
            ref_d, ref_psi = self.generate_ghost_car_target(
                current_s=future_s, 
                start_s=current_s, 
                end_s=current_s + lane_change_length, 
                start_d=current_d, 
                end_d=target_lane_d
            )
            
            # Load the data into the lateral template
            self.lat_tvp['_tvp', k, 'v_ext'] = future_v
            self.lat_tvp['_tvp', k, 'd_ref'] = ref_d
            self.lat_tvp['_tvp', k, 'psi_ref'] = ref_psi
            
            # TODO: Plug in your dynamic get_dynamic_road_tunnel() radar sweep here!
            self.lat_tvp['_tvp', k, 'd_min'] = -6.0 
            self.lat_tvp['_tvp', k, 'd_max'] = 18.0 

        lat_x0 = np.array([current_d, current_psi]).reshape(-1, 1)
        u_lat = self.lat_mpc.make_step(lat_x0)
        optimal_steer = float(u_lat[0])

        # --- 4. RETURN COUPLED ACTION ---
        return np.array([optimal_accel, optimal_steer], dtype=np.float32)

    def create_lateral_mpc(self):
        model = do_mpc.model.Model('continuous')
        
        # States & Inputs
        d = model.set_variable(var_type='_x', var_name='d')
        psi = model.set_variable(var_type='_x', var_name='psi')
        steer = model.set_variable(var_type='_u', var_name='steer')

        # Time-Varying Parameters
        v_ext = model.set_variable(var_type='_tvp', var_name='v_ext') 
        d_ref = model.set_variable(var_type='_tvp', var_name='d_ref')
        psi_ref = model.set_variable(var_type='_tvp', var_name='psi_ref')
        d_min = model.set_variable(var_type='_tvp', var_name='d_min')
        d_max = model.set_variable(var_type='_tvp', var_name='d_max')

        # Kinematics
        wheelbase = 2.5
        model.set_rhs('d', v_ext * ca.sin(psi))
        model.set_rhs('psi', (v_ext / wheelbase) * ca.tan(steer))
        model.setup()

        # Controller Setup
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {'n_horizon': 20, 't_step': 0.1, 'store_full_solution': True}
        mpc.set_param(**setup_mpc)

        # Cost Function
        lterm = 20.0 * (d - d_ref)**2 + 50.0 * (psi - psi_ref)**2
        mpc.set_objective(mterm=lterm, lterm=lterm)
        mpc.set_rterm(steer=100.0) 

        # Bounds & Constraints
        max_steer = 0.523 # ~30 degrees
        max_yaw = 0.087   # ~5 degrees
        mpc.bounds['lower', '_u', 'steer'] = -max_steer
        mpc.bounds['upper', '_u', 'steer'] = max_steer
        mpc.bounds['lower', '_x', 'psi'] = -max_yaw
        mpc.bounds['upper', '_x', 'psi'] = max_yaw

        mpc.set_nl_cons('stay_right_of_left_line', d_min - d, ub=0.0)
        mpc.set_nl_cons('stay_left_of_right_line', d - d_max, ub=0.0)

        # TVP Attachment BEFORE Setup
        tvp_template = mpc.get_tvp_template()
        mpc.set_tvp_fun(lambda t_now: tvp_template)
        mpc.setup()
        
        return mpc, model, tvp_template
    
    def create_longitudinal_mpc(self):
        model = do_mpc.model.Model('continuous')
        
        # States & Inputs
        s = model.set_variable(var_type='_x', var_name='s')
        v = model.set_variable(var_type='_x', var_name='v')
        a = model.set_variable(var_type='_u', var_name='a')

        # Time-Varying Parameters
        s_ref = model.set_variable(var_type='_tvp', var_name='s_ref')
        v_ref = model.set_variable(var_type='_tvp', var_name='v_ref')
        s_min = model.set_variable(var_type='_tvp', var_name='s_min')
        s_max = model.set_variable(var_type='_tvp', var_name='s_max')

        # Kinematics
        model.set_rhs('s', v)
        model.set_rhs('v', a)
        model.setup()

        # Controller Setup
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {'n_horizon': 20, 't_step': 0.1, 'store_full_solution': True}
        mpc.set_param(**setup_mpc)

        # Cost Function
        lterm = 10.0 * (s - s_ref)**2 + 5.0 * (v - v_ref)**2
        mpc.set_objective(mterm=lterm, lterm=lterm)
        mpc.set_rterm(a=10.0) 

        # Bounds & Constraints
        mpc.bounds['lower', '_u', 'a'] = -5.0 
        mpc.bounds['upper', '_u', 'a'] = 3.0  
        mpc.bounds['lower', '_x', 'v'] = 0.0  

        mpc.set_nl_cons('stay_behind_lead', s - s_max, ub=0.0) 
        mpc.set_nl_cons('stay_ahead_of_lag', s_min - s, ub=0.0)

        # TVP Attachment BEFORE Setup
        tvp_template = mpc.get_tvp_template()
        mpc.set_tvp_fun(lambda t_now: tvp_template)
        mpc.setup()
        
        return mpc, model, tvp_template
    
    def generate_ghost_car_target(self, current_s, start_s, end_s, start_d, end_d):
        """
        Generates a perfectly smooth S-curve using a Cosine function.
        Returns the target lateral position (d) and target heading (psi).
        """
        if current_s <= start_s:
            return start_d, 0.0 
            
        if current_s >= end_s:
            return end_d, 0.0 

        # Calculate progress from 0.0 to 1.0
        progress = (current_s - start_s) / (end_s - start_s)
        
        # Smooth interpolation for Lateral Position (d)
        target_d = start_d + (end_d - start_d) * 0.5 * (1 - np.cos(np.pi * progress))
        
        # Derivative of the curve to find Heading Angle (psi)
        slope = (end_d - start_d) * 0.5 * np.pi * np.sin(np.pi * progress) / (end_s - start_s)
        target_psi = np.arctan(slope)
        
        return target_d, target_psi
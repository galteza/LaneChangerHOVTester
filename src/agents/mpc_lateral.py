import do_mpc
import casadi as ca

class MPC_Lateral():
    def __init__(self, vehiclemodel_params, sutlanechanger_mpc_params):
        self.model = None
        self.mpc = None
        
        self.wheelbase_L = vehiclemodel_params['wheelbase_L']
    
        self.horizon_N = sutlanechanger_mpc_params['horizon_N']
        self.dt = sutlanechanger_mpc_params['dt']
        self.Q_d = sutlanechanger_mpc_params['Q_d']
        self.Q_heading = sutlanechanger_mpc_params['Q_heading']
        self.R_steer = sutlanechanger_mpc_params['R_steer']
        self.max_steer_rad = sutlanechanger_mpc_params['max_steer_rad']
        self.min_steer_rad = sutlanechanger_mpc_params['min_steer_rad']
        self.max_yaw_rad = sutlanechanger_mpc_params['max_yaw_rad']
        self.min_yaw_rad = sutlanechanger_mpc_params['min_yaw_rad']

        self._setup_mpc()

    def _setup_mpc(self):
        self.model = do_mpc.model.Model('continuous')

        ### SETUP MODEL

        # === 1. States & Inputs ===
        d = self.model.set_variable(var_type='_x', var_name='d')
        heading = self.model.set_variable(var_type='_x', var_name='psi')

        steer = self.model.set_variable(var_type='_u', var_name='steer')

        # === 2. Time-Varying Parameters ===
        v_ext = self.model.set_variable(var_type='_tvp', var_name='v_ext') # needs to know the velocity with both components (calculated outside of MPC_Lateral) 
        
        d_ref = self.model.set_variable(var_type='_tvp', var_name='d_ref') # target lateral position
        heading_ref = self.model.set_variable(var_type='_tvp', var_name='heading_ref') # target heading angle

        d_min = self.model.set_variable(var_type='_tvp', var_name='d_min') # lower bound of lateral position (according to lane formation)
        d_max = self.model.set_variable(var_type='_tvp', var_name='d_max')   # upper bound of lateral position (according to lane formation)

        # === 3. Equations of Motion ===
        self.model.set_rhs('d', v_ext * ca.sin(heading_ref)) # takes the lateral velocity and calculates next d
        self.model.set_rhs('psi', (v_ext / self.wheelbase_L) * ca.tan(steer))
        self.model.setup()

        ### SETUP CONTROLLER

        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': self.horizon_N, 
            't_step': self.dt, 
            'store_full_solution': True
        }
        self.mpc.set_param(**setup_mpc)

        # === 4. Cost Function (Track the Ghost Car) ===
        # Keep the bumper on the reference line and the nose pointed correctly
        lterm = self.Q_d * (d - d_ref)**2 + self.Q_heading * (heading - heading_ref)**2
        self.mpc.set_objective(mterm=lterm, lterm=lterm)
        self.mpc.set_rterm(steer=self.R_steer)

        # === 5. Constraints  ===
        
        # wheel turn angle, which will be "scaled" and added to heading 
        self.mpc.bounds['lower', '_u', 'steer'] = self.min_steer_rad
        self.mpc.bounds['upper', '_u', 'steer'] = self.max_steer_rad

        # heading shouldn't be too much
        self.mpc.bounds['lower', '_x', 'heading'] = self.min_yaw_rad
        self.mpc.bounds['upper', '_x', 'heading'] = self.max_yaw_rad

        # Non-linear constraints for the dynamic radar funnel
        self.mpc.set_nl_cons('stay_right_of_left_line', d_min - d, ub=0.0)
        self.mpc.set_nl_cons('stay_left_of_right_line', d - d_max, ub=0.0)

        self.mpc.setup()

    def get_tvp_template(self):
        tvp_template = self.mpc.get_tvp_template()
        return tvp_template


    def make_step(self, current_state):
        return self.mpc.make_step(current_state)


    def update_targets_and_obstacles_in_sut_mpc(self, target_d, target_v, observation_matrix):
        tvp_template = self.mpc.get_tvp_template()

        for k in range(self.horizon_N + 1):

            # target distance and velocity do not change (ASSUMED)
            tvp_template['_tvp', k, 'target_d'] = target_d
            tvp_template['_tvp', k, 'target_v'] = target_v

            for i in range(self.observed_vehicles_count):
                # gives no presence (0) to the excess observed vehicles count  
                presence = observation_matrix[i][0] if len(observation_matrix) > i else 0

                if presence == 1:
                    observed_x_current = observation_matrix[i][1]
                    observed_y_current = observation_matrix[i][2]
                    observed_vx_current = observation_matrix[i][3]
                    observed_vy_current = observation_matrix[i][4]

                    # POSITION PREDICTION FOR EACH CAR

                    predicted_x = observed_x_current + observed_vx_current * (k * self.dt)
                    predicted_y = observed_y_current + observed_vy_current * (k * self.dt)

                    tvp_template['_tvp', k, f'obs_{i}_x'] = predicted_x
                    tvp_template['_tvp', k, f'obs_{i}_y'] = predicted_y

                else:
                    # set safety ellipse really far back if car is of no presence
                    tvp_template['_tvp', k, f'obs_{i}_x'] = -1000.0
                    tvp_template['_tvp', k, f'obs_{i}_y'] = -1000.0

        self.current_tvp = tvp_template
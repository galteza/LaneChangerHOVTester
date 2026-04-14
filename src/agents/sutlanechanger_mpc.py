import do_mpc
import casadi as ca
from env.highway_env_mergeexit import MergeExitLaneHighway_Environment

class SUTLaneChanger_MPC:
    def __init__(self, env_params, vehiclemodel_params, sutlanechanger_mpc_params):
        self.observed_vehicles_count = env_params['observation']['observation_config']['vehicles_count']

        self.wheelbase_L = vehiclemodel_params['wheelbase_L']
        self.max_steer_rad = vehiclemodel_params['max_steer_rad']
        self.max_long_accel_ms2 = vehiclemodel_params['max_long_accel_ms2']
        self.min_long_accel_ms2 = vehiclemodel_params['min_long_accel_ms2']

        self.longitudinal_margin = vehiclemodel_params['safety_margins']['longitudinal_margin']
        self.lateral_margin = vehiclemodel_params['safety_margins']['lateral_margin']

        self.horizon_N = sutlanechanger_mpc_params['horizon_N']
        self.dt = sutlanechanger_mpc_params['dt']
        self.target_v = sutlanechanger_mpc_params['target_v']
        self.target_d = (env_params['lanes_count'] - 1) * 4.0 # sutlanechanger_mpc_params['target_d']
        self.target_s = ()
        self.target_heading = 

        self.Q_lateral = sutlanechanger_mpc_params['Q_lateral']
        self.Q_velocity = sutlanechanger_mpc_params['Q_velocity']
        self.Q_heading = sutlanechanger_mpc_params['Q_heading']
        self.R_steering = sutlanechanger_mpc_params['R_steering']
        self.R_accel = sutlanechanger_mpc_params['R_accel']

        self.current_tvp = None
        self.model = None
        self.mpc = None

        self._setup_mpc()

    def _setup_mpc(self):
        # === 1. REGISTER THE TYPE OF MODEL FOR MPC ===
        model_type = 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # === 2. REGISTER STATE-SPACE REPRESENTATION/EOM USING KBM + TIME-VARYING PARAMS ===

            # STATES (x)
        s = self.model.set_variable(var_type='_x', var_name='s')         # distance along road
        d = self.model.set_variable(var_type='_x', var_name='d')         # lateral deviation
        psi = self.model.set_variable(var_type='_x', var_name='psi')     # heading angle
        v = self.model.set_variable(var_type='_x', var_name='v')         # longitudinal velocity

            # INPUT (u)
        steering = self.model.set_variable(var_type='_u', var_name='steering')       # steering angle
        accel = self.model.set_variable(var_type='_u', var_name='accel')             # longitudinal acceleration


        # === 3. REGISTER RIGHT-HAND SIDE ===

        self.model.set_rhs('s', v * ca.cos(psi))
        self.model.set_rhs('d', v * ca.sin(psi))
        self.model.set_rhs('psi', (v / self.wheelbase_L) * ca.tan(steering))
        self.model.set_rhs('v', accel)


        # === 4. REGISTER TIME-VARYING PARAMETERS (TVP) ===

        self.model.set_variable(var_type='_tvp', var_name='target_d')
        self.model.set_variable(var_type='_tvp', var_name='target_v')
        self.model.set_variable(var_type='_tvp', var_name='target_psi')
        self.model.set_variable(var_type='_tvp', var_name='d_min')
        self.model.set_variable(var_type='_tvp', var_name='d_max')

        for i in range(self.observed_vehicles_count):
            self.model.set_variable(var_type='_tvp', var_name=f'obs_{i}_x')
            self.model.set_variable(var_type='_tvp', var_name=f'obs_{i}_y')

        # === 5. DEPLOY MODEL ===

        self.model.setup()

        # === 6. REGISTER CONTROLLER ===

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.horizon_N,
            't_step': self.dt,
            'store_full_solution': True,
        }

        self.mpc.set_param(**setup_mpc)

            # ESTABLISH TIME VARYING PARAMETERS (SURROUNDING CRUISERS)
        tvp_template = self.mpc.get_tvp_template()

        tvp_template['_tvp', :, 'target_d'] = self.target_d
        tvp_template['_tvp', :, 'target_v'] = self.target_v

        for i in range(self.observed_vehicles_count):
            tvp_template['_tvp', :, f'obs_{i}_x'] = -1000.0
            tvp_template['_tvp', :, f'obs_{i}_y'] = -1000.0

        self.current_tvp = tvp_template

        def tvp_fun(t_now):
            return self.current_tvp
        
        self.mpc.set_tvp_fun(tvp_fun)
        # do_mpc: set_tvp_fun(<insert python function>)
        
        # === 6. ESTABLISH COST FUNCTION

        lterm = (self.Q_lateral * (d - self.model.tvp['target_d'])**2 
                 + self.Q_velocity * (v - self.model.tvp['target_v'])**2 
                 + self.Q_psi * (psi - self.model.tvp['target_psi'])
                ) # Lagrange/term with Q matrix
        
        mterm = self.Q_lateral * (d - self.model.tvp['target_d'])**2 # Mayer term 
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            steering = self.R_steering,         # Keep steering smooth
            accel = self.R_accel                # Keep acceleration smooth
        )

        self.mpc.bounds['lower', '_u', 'steering'] = -self.max_steer_rad
        self.mpc.bounds['upper', '_u', 'steering'] = self.max_steer_rad
        self.mpc.bounds['lower', '_u', 'accel'] = self.min_long_accel_ms2
        self.mpc.bounds['upper', '_u', 'accel'] = self.max_long_accel_ms2
        self.mpc.bounds['lower', '_x', 'v'] = 0.0

        # === 7. OBSERVATION OF OTHER VEHICLES

        for i in range(self.observed_vehicles_count):
            observed_x = self.model.tvp[f'obs_{i}_x']
            observed_y = self.model.tvp[f'obs_{i}_y']

            # equation for ellipse
            collision_expression = 1.0 - ((s - observed_x) / self.longitudinal_margin)**2 - ((d - observed_y) / self.lateral_margin)**2

            self.mpc.set_nl_cons(f'obstacle_{i}_avoidance', collision_expression, ub=0.0)

        # === 8. REGISTER MPC & RETURN MODEL

        self.mpc.setup()
        
        return self.model
    
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
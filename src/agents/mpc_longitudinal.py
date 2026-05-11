import do_mpc
import casadi as ca

class MPC_Longitudinal:
    def __init__(self, env_params, vehiclemodel_params, sutlanechanger_mpc_params):
        self.max_long_vel_ms = vehiclemodel_params['max_long_vel_ms']
        self.max_long_accel_ms2 = vehiclemodel_params['max_long_accel_ms2']
        self.min_long_accel_ms2 = vehiclemodel_params['min_long_accel_ms2']

        self.Q_s = sutlanechanger_mpc_params['Q_s']
        self.Q_v = sutlanechanger_mpc_params['Q_v']
        self.R_a = sutlanechanger_mpc_params['R_a']
        
        self.horizon_N = sutlanechanger_mpc_params['horizon_N']
        self.dt = sutlanechanger_mpc_params['dt']

        self.model = None
        self.mpc = None

        self._setup_mpc()

    def _setup_mpc(self):
    
        self.model = do_mpc.model.Model('continuous')
        
        # --- 1. States & Inputs ---
        s = self.model.set_variable(var_type='_x', var_name='s')
        v = self.model.set_variable(var_type='_x', var_name='v')

        a = self.model.set_variable(var_type='_u', var_name='a')

        # --- 2. Time-Varying Parameters (The Ghost Car & Traffic) ---
        s_ref = model.set_variable(var_type='_tvp', var_name='s_ref')
        v_ref = model.set_variable(var_type='_tvp', var_name='v_ref') # try to keep this reference velocity
        s_min = model.set_variable(var_type='_tvp', var_name='s_min') # for the minimum of the corridor
        s_max = model.set_variable(var_type='_tvp', var_name='s_max') # for the maximum of the corridor

        # --- 3. Equations of Motion ---
        self.model.set_rhs('s', v)
        self.model.set_rhs('v', a)
        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.horizon_N, 
            't_step': self.dt, 
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)

        # --- 4. Cost Function (Track the Ghost Car) ---
        # Penalize deviation from the reference trajectory and jerk (changing acceleration)
        lterm = self.Q_s * (s - s_ref)**2 + self.Q_v * (v - v_ref)**2
        self.mpc.set_objective(mterm=lterm, lterm=lterm)
        self.mpc.set_rterm(a=self.R_a) # Smooth gas/brake pedal

        # --- 5. Constraints (Safety Corridors & Physics) ---
        self.mpc.bounds['lower', '_u', 'a'] = self.min_long_accel_ms2 # Max braking
        self.mpc.bounds['upper', '_u', 'a'] = self.max_long_accel_ms2 # Max acceleration
        self.mpc.bounds['lower', '_x', 'v'] = 0.0  # No reversing on the highway!
        self.mpc.bounds['upper', '_x', 'v'] = self.max_long_vel_ms

        # Non-linear constraints to prevent rear-ending cars
        self.mpc.set_nl_cons('stay_behind_lead', s - s_max, ub=0.0) 
        self.mpc.set_nl_cons('stay_ahead_of_lag', s_min - s, ub=0.0)

        self.mpc.setup()

def get_tvp_template(self):
    tvp_template = self.mpc.get_tvp_template()
    return tvp_template

def make_step(self, current_state):
    return self.mpc.make_step(current_state)
import do_mpc
import casadi as ca

def create_longitudinal_mpc():
    model = do_mpc.model.Model('continuous')

    # --- 1. States & Inputs ---
    s = model.set_variable(var_type='_x', var_name='s')
    v = model.set_variable(var_type='_x', var_name='v')
    a = model.set_variable(var_type='_u', var_name='a')

    # --- 2. Time-Varying Parameters (The Ghost Car & Traffic) ---
    s_ref = model.set_variable(var_type='_tvp', var_name='s_ref')
    v_ref = model.set_variable(var_type='_tvp', var_name='v_ref')
    s_min = model.set_variable(var_type='_tvp', var_name='s_min')
    s_max = model.set_variable(var_type='_tvp', var_name='s_max')

    # --- 3. Equations of Motion ---
    model.set_rhs('s', v)
    model.set_rhs('v', a)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {'n_horizon': 20, 't_step': 0.1, 'store_full_solution': True}
    mpc.set_param(**setup_mpc)

    # --- 4. Cost Function (Track the Ghost Car) ---
    # Penalize deviation from the reference trajectory and jerk (changing acceleration)
    lterm = 10.0 * (s - s_ref)**2 + 5.0 * (v - v_ref)**2
    mpc.set_objective(mterm=lterm, lterm=lterm)
    mpc.set_rterm(a=10.0) # Smooth gas/brake pedal

    # --- 5. Constraints (Safety Corridors & Physics) ---
    mpc.bounds['lower', '_u', 'a'] = -5.0 # Max braking
    mpc.bounds['upper', '_u', 'a'] = 3.0  # Max acceleration
    mpc.bounds['lower', '_x', 'v'] = 0.0  # No reversing on the highway!

    # Non-linear constraints to prevent rear-ending cars
    mpc.set_nl_cons('stay_behind_lead', s - s_max, ub=0.0) 
    mpc.set_nl_cons('stay_ahead_of_lag', s_min - s, ub=0.0)

    mpc.setup()
    return mpc, model

def get_tvp_template(self):
    tvp_template = self.mpc.get_tvp_template()
    return tvp_template


def make_step(self, current_state):
    return self.mpc.make_step(current_state)
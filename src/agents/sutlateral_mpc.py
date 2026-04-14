import do_mpc
import casadi as ca

def create_lateral_mpc():
    model = do_mpc.model.Model('continuous')

    # --- 1. States & Inputs ---
    d = model.set_variable(var_type='_x', var_name='d')
    psi = model.set_variable(var_type='_x', var_name='psi')
    steer = model.set_variable(var_type='_u', var_name='steer')

    # --- 2. Time-Varying Parameters (The Bridge & The Ghost Car) ---
    # This is how the gas pedal tells the steering wheel how fast it's going!
    v_ext = model.set_variable(var_type='_tvp', var_name='v_ext') 
    
    d_ref = model.set_variable(var_type='_tvp', var_name='d_ref')
    psi_ref = model.set_variable(var_type='_tvp', var_name='psi_ref')
    d_min = model.set_variable(var_type='_tvp', var_name='d_min')
    d_max = model.set_variable(var_type='_tvp', var_name='d_max')

    # --- 3. Equations of Motion ---
    wheelbase = 2.5
    model.set_rhs('d', v_ext * ca.sin(psi))
    model.set_rhs('psi', (v_ext / wheelbase) * ca.tan(steer))
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {'n_horizon': 20, 't_step': 0.1, 'store_full_solution': True}
    mpc.set_param(**setup_mpc)

    # --- 4. Cost Function (Track the Ghost Car) ---
    # Keep the bumper on the reference line and the nose pointed correctly
    lterm = 20.0 * (d - d_ref)**2 + 50.0 * (psi - psi_ref)**2
    mpc.set_objective(mterm=lterm, lterm=lterm)
    mpc.set_rterm(steer=100.0) # HEAVY penalty to stop the crab-walk

    # --- 5. Constraints (Road Boundaries & Physics) ---
    max_steer = 0.523 # roughly 30 degrees in radians
    max_yaw = 0.087   # roughly 5 degrees in radians
    
    mpc.bounds['lower', '_u', 'steer'] = -max_steer
    mpc.bounds['upper', '_u', 'steer'] = max_steer
    mpc.bounds['lower', '_x', 'psi'] = -max_yaw
    mpc.bounds['upper', '_x', 'psi'] = max_yaw

    # Non-linear constraints for the dynamic radar funnel
    mpc.set_nl_cons('stay_right_of_left_line', d_min - d, ub=0.0)
    mpc.set_nl_cons('stay_left_of_right_line', d - d_max, ub=0.0)

    mpc.setup()
    return mpc, model

def get_tvp_template(self):
    tvp_template = self.mpc.get_tvp_template()
    return tvp_template


def make_step(self, current_state):
    return self.mpc.make_step(current_state)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# Import your actual risk calculators and reward functions
from src.env.risk_calculators import PolygonTTCCalculator
from src.env.reward_functions import (
    RewardTTCEgoAdvFunction, 
    RewardTTCAdvAdvFunction,
    SandwichingRewardFunction,
    LaneKeepingRewardFunction,
    DistanceToEgoRewardFunction,
    SpeedMatchingRewardFunction
)

# ==========================================
# 1. Vehicle State Container
# ==========================================
class MockVehicle:
    def __init__(self, x, y, speed, heading):
        self.position = np.array([float(x), float(y)])
        self.speed = float(speed)
        self.heading = np.deg2rad(heading)
        self.LENGTH = 5.0
        self.WIDTH = 2.0
        self.velocity = np.array([0.0, 0.0])
        self.crashed = False
        self.update_velocity()

    def update_velocity(self):
        self.velocity = np.array([self.speed * np.cos(self.heading), self.speed * np.sin(self.heading)])

    def step(self, dt):
        self.position += self.velocity * dt

# A1, A2 = Adversaries, Ego = Ego Vehicle
veh_A1 = MockVehicle(10, 10, 20, 0)
veh_A2 = MockVehicle(10, 2, 20, 0)
veh_Ego = MockVehicle(50, 6, 25, 0)

# ==========================================
# 2. Initialize Reward Calculators
# ==========================================
adv_ego_reward_calculator = RewardTTCEgoAdvFunction()
adv_adv_reward_calculator = RewardTTCAdvAdvFunction()
sandwiching_reward_calculator = SandwichingRewardFunction()
lane_keeping_reward_calculator = LaneKeepingRewardFunction()
dist_to_ego_reward_calculator = DistanceToEgoRewardFunction()
speed_matching_reward_calculator = SpeedMatchingRewardFunction()

# Initialize phase for calculations (Assuming starting in blocking phase)
adv_ego_reward_calculator.check_phase(veh_Ego.position[0])
sandwiching_reward_calculator.check_phase(veh_Ego.position[0])

# ==========================================
# 3. Main Plot Setup
# ==========================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(bottom=0.35, right=0.75)

ax.set_xlim(-10, 100)
ax.set_ylim(-10, 60)
ax.set_aspect('equal')
ax.grid(color='#333333', linestyle=':', linewidth=0.5)
ax.set_title("Multi-Agent Reward Landscape Playground", color='white', pad=15, fontsize=16)

# --- DRAW HIGHWAY LANES ---
lane_width = 4.0
num_lanes = 4
for i in range(num_lanes + 1):
    y_pos = i * lane_width
    if i == 0 or i == num_lanes:
        # Outer road boundaries
        ax.axhline(y_pos, color='white', linewidth=2, linestyle='-')
    else:
        # Inner lane dividers
        ax.axhline(y_pos, color='gray', linewidth=1, linestyle='--')

# Colors: A1=Cyan, A2=Orange, Ego=Magenta
C_A1 = '#00FFFF'
C_A2 = '#FFA500'
C_EGO = '#FF00FF'

poly_A1 = patches.Polygon([[0,0]], closed=True, fc=C_A1, alpha=0.6)
poly_A2 = patches.Polygon([[0,0]], closed=True, fc=C_A2, alpha=0.6)
poly_Ego = patches.Polygon([[0,0]], closed=True, fc=C_EGO, alpha=0.6)
ax.add_patch(poly_A1)
ax.add_patch(poly_A2)
ax.add_patch(poly_Ego)

arrow_A1 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=C_A1, width=0.004)
arrow_A2 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=C_A2, width=0.004)
arrow_Ego = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=C_EGO, width=0.004)

tail_A1_plot, = ax.plot([], [], 'o', color=C_A1, markersize=8, alpha=0.8)
tail_A2_plot, = ax.plot([], [], 'o', color=C_A2, markersize=8, alpha=0.8)
tail_Ego_plot, = ax.plot([], [], 'o', color=C_EGO, markersize=8, alpha=0.8)

tip_A1_plot, = ax.plot([], [], 'D', color=C_A1, markersize=6, alpha=0.8) 
tip_A2_plot, = ax.plot([], [], 'D', color=C_A2, markersize=6, alpha=0.8) 
tip_Ego_plot, = ax.plot([], [], 'D', color=C_EGO, markersize=6, alpha=0.8)

stats_A1 = ax.text(0, 0, "", color=C_A1, fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
stats_A2 = ax.text(0, 0, "", color=C_A2, fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
stats_Ego = ax.text(0, 0, "EGO", color=C_EGO, fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

reward_panel_text = fig.text(0.76, 0.5, "", fontsize=10, color='white', verticalalignment='center', family='monospace', bbox=dict(facecolor='#111111', edgecolor='#AAAAAA', alpha=0.8, pad=10))
text_status = ax.text(0, 55, "", fontsize=14, fontweight='bold', color='white', bbox=dict(facecolor='black', edgecolor='white', alpha=0.8, pad=5))

# ==========================================
# 4. Controls UI Layout
# ==========================================
sim_state = {'is_playing': False, 'saved_config': None, 'drag_mode': None, 'active_veh': None}

# 3 Columns for 3 Vehicles
ax_a1s = fig.add_axes([0.05, 0.20, 0.20, 0.03])
ax_a1h = fig.add_axes([0.05, 0.10, 0.20, 0.03])
ax_a2s = fig.add_axes([0.30, 0.20, 0.20, 0.03])
ax_a2h = fig.add_axes([0.30, 0.10, 0.20, 0.03])
ax_egos = fig.add_axes([0.55, 0.20, 0.20, 0.03])
ax_egoh = fig.add_axes([0.55, 0.10, 0.20, 0.03])

sl_a1s = Slider(ax_a1s, 'A1 Spd', 0.0, 40.0, valinit=veh_A1.speed, color=C_A1)
sl_a1h = Slider(ax_a1h, 'A1 Hdg', 0.0, 360.0, valinit=np.rad2deg(veh_A1.heading), color=C_A1)
sl_a2s = Slider(ax_a2s, 'A2 Spd', 0.0, 40.0, valinit=veh_A2.speed, color=C_A2)
sl_a2h = Slider(ax_a2h, 'A2 Hdg', 0.0, 360.0, valinit=np.rad2deg(veh_A2.heading), color=C_A2)
sl_egos = Slider(ax_egos, 'Ego Spd', 0.0, 40.0, valinit=veh_Ego.speed, color=C_EGO)
sl_egoh = Slider(ax_egoh, 'Ego Hdg', 0.0, 360.0, valinit=np.rad2deg(veh_Ego.heading), color=C_EGO)
sliders = [sl_a1s, sl_a1h, sl_a2s, sl_a2h, sl_egos, sl_egoh]

btn_play = Button(fig.add_axes([0.80, 0.20, 0.10, 0.05]), '▶ Play', color='#2E7D32', hovercolor='#4CAF50')
btn_pause = Button(fig.add_axes([0.80, 0.14, 0.10, 0.05]), '⏸ Pause', color='#F57C00', hovercolor='#FF9800')
btn_reset = Button(fig.add_axes([0.80, 0.08, 0.10, 0.05]), '↺ Reset', color='#1565C0', hovercolor='#2196F3')
for btn in [btn_play, btn_pause, btn_reset]: btn.label.set_color('white')

# ==========================================
# 5. Master Visual Update Logic
# ==========================================
def sync_slider_to_physics(slider, val):
    slider.eventson = False
    slider.set_val(val)
    slider.eventson = True

def update_from_sliders(val=None):
    if sim_state['is_playing']: return
    veh_A1.heading, veh_A1.speed = np.deg2rad(sl_a1h.val), sl_a1s.val
    veh_A2.heading, veh_A2.speed = np.deg2rad(sl_a2h.val), sl_a2s.val
    veh_Ego.heading, veh_Ego.speed = np.deg2rad(sl_egoh.val), sl_egos.val
    update_visuals()

for s in sliders: s.on_changed(update_from_sliders)

def get_tail_pos(veh): return veh.position - 6.0 * np.array([np.cos(veh.heading), np.sin(veh.heading)])

def compute_all_rewards():
    """Computes continuous rewards for TWO adversaries vs Ego"""
    
    ttc_1_ego = PolygonTTCCalculator.compute_ttc(veh_A1, veh_Ego)
    ttc_2_ego = PolygonTTCCalculator.compute_ttc(veh_A2, veh_Ego)
    ttc_adv_adv = PolygonTTCCalculator.compute_ttc(veh_A1, veh_A2)
    
    # Sandwiching / Swarm Bonus (Passing both agents)
    r_sandwich_array = sandwiching_reward_calculator.compute_reward([veh_A1, veh_A2], veh_Ego)
    
    # Adv-Adv Collision Penalty
    r_adv_adv = adv_adv_reward_calculator.compute_reward(ttc_adv_adv)

    res = {'A1': {}, 'A2': {}}
    
    for i, adv in enumerate([veh_A1, veh_A2]):
        key = f'A{i+1}'
        ttc_ego = ttc_1_ego if i == 0 else ttc_2_ego
        
        r_ttc_ego = adv_ego_reward_calculator.compute_reward(ttc_ego)
        r_dist = dist_to_ego_reward_calculator.compute_reward(np.linalg.norm(adv.position - veh_Ego.position))
        r_lane = lane_keeping_reward_calculator.compute_reward(adv.position[1], 0.0, 16.0)
        r_speed = speed_matching_reward_calculator.compute_reward(abs(adv.speed - veh_Ego.speed))
        
        total = r_ttc_ego + r_adv_adv + r_sandwich_array[i] + r_dist + r_lane + r_speed
        
        res[key] = {
            'TTC_Ego': r_ttc_ego,
            'TTC_Adv': r_adv_adv,
            'Sandwich': r_sandwich_array[i],
            'Dist': r_dist,
            'Lane': r_lane,
            'Speed': r_speed,
            'Total': total
        }
    
    res['Raw_TTC_1'] = ttc_1_ego
    res['Raw_TTC_2'] = ttc_2_ego
    res['Raw_TTC_Adv'] = ttc_adv_adv
    return res

def update_visuals():
    for v in [veh_A1, veh_A2, veh_Ego]: v.update_velocity()
    
    r = compute_all_rewards()
    
    # Update Polygons & Arrows
    vehicles = [(veh_A1, poly_A1, arrow_A1, tip_A1_plot, tail_A1_plot),
                (veh_A2, poly_A2, arrow_A2, tip_A2_plot, tail_A2_plot),
                (veh_Ego, poly_Ego, arrow_Ego, tip_Ego_plot, tail_Ego_plot)]
    
    for veh, poly, arrow, tip, tail_p in vehicles:
        corners = PolygonTTCCalculator.get_bounding_box_corners(veh.position[0], veh.position[1], veh.heading, veh.LENGTH, veh.WIDTH)
        poly.set_xy(corners)
        arrow.set_offsets(veh.position)
        arrow.set_UVC(veh.velocity[0], veh.velocity[1])
        tip.set_data([veh.position[0] + veh.velocity[0]], [veh.position[1] + veh.velocity[1]])
        tail = get_tail_pos(veh)
        tail_p.set_data([tail[0]], [tail[1]])

    # Update Floating Text
    stats_A1.set_text(f"Reward: {r['A1']['Total']:.2f}")
    stats_A1.set_position((veh_A1.position[0] - 5, veh_A1.position[1] + 5))
    stats_A2.set_text(f"Reward: {r['A2']['Total']:.2f}")
    stats_A2.set_position((veh_A2.position[0] - 5, veh_A2.position[1] + 5))
    stats_Ego.set_text(f"EGO\nSpd: {veh_Ego.speed:.1f}")
    stats_Ego.set_position((veh_Ego.position[0] - 5, veh_Ego.position[1] + 5))

    # Update Right Panel
    panel_text = (
        "=== ADV 1 (CYAN) ===\n"
        f"TTC Ego   : {r['A1']['TTC_Ego']:>7.2f}\n"
        f"TTC Adv   : {r['A1']['TTC_Adv']:>7.2f}\n"
        f"Sandwich  : {r['A1']['Sandwich']:>7.2f}\n"
        f"Proximity : {r['A1']['Dist']:>7.2f}\n"
        f"Lane/Speed: {r['A1']['Lane']+r['A1']['Speed']:>7.2f}\n"
        f"NET REWARD: {r['A1']['Total']:>7.2f}\n\n"
        "=== ADV 2 (ORNG) ===\n"
        f"TTC Ego   : {r['A2']['TTC_Ego']:>7.2f}\n"
        f"TTC Adv   : {r['A2']['TTC_Adv']:>7.2f}\n"
        f"Sandwich  : {r['A2']['Sandwich']:>7.2f}\n"
        f"Proximity : {r['A2']['Dist']:>7.2f}\n"
        f"Lane/Speed: {r['A2']['Lane']+r['A2']['Speed']:>7.2f}\n"
        f"NET REWARD: {r['A2']['Total']:>7.2f}\n\n"
        "--- RAW METRICS ---\n"
        f"TTC A1-Ego : {r['Raw_TTC_1']:.2f}s\n"
        f"TTC A2-Ego : {r['Raw_TTC_2']:.2f}s\n"
        f"TTC A1-A2  : {r['Raw_TTC_Adv']:.2f}s"
    )
    reward_panel_text.set_text(panel_text)

    # Status Bar logic (Fratricide check)
    status = "[PLAYING]" if sim_state['is_playing'] else "[PAUSED]"
    if r['Raw_TTC_Adv'] < 2.0:
        text_status.set_text(f" {status} WARNING: FRATRICIDE IMMINENT ")
        text_status.set_color('#FF3333')
    else:
        text_status.set_text(f" {status} Team Platoon Metrics Active ")
        text_status.set_color('#00FFFF')
        
    # Polygon Color Shifts on Net Reward
    poly_A1.set_facecolor('#00FF00' if r['A1']['Total'] > 0 else C_A1)
    poly_A2.set_facecolor('#00FF00' if r['A2']['Total'] > 0 else C_A2)
    
    fig.canvas.draw_idle()

# ==========================================
# 6. Interactive Dragging Logic
# ==========================================
def on_press(event):
    if sim_state['is_playing'] or event.inaxes != ax: return
    pts = {
        'speed_A1': veh_A1.position + veh_A1.velocity, 'head_A1': get_tail_pos(veh_A1), 'move_A1': veh_A1.position,
        'speed_A2': veh_A2.position + veh_A2.velocity, 'head_A2': get_tail_pos(veh_A2), 'move_A2': veh_A2.position,
        'speed_Ego': veh_Ego.position + veh_Ego.velocity, 'head_Ego': get_tail_pos(veh_Ego), 'move_Ego': veh_Ego.position,
    }
    closest_mode, min_dist = None, float('inf')
    for mode, pt in pts.items():
        dist = np.hypot(event.xdata - pt[0], event.ydata - pt[1])
        if dist < (3.5 if 'move' in mode else 2.5) and dist < min_dist:
            min_dist, closest_mode = dist, mode

    if closest_mode:
        sim_state['drag_mode'] = closest_mode
        if '_A1' in closest_mode: sim_state['active_veh'] = veh_A1
        elif '_A2' in closest_mode: sim_state['active_veh'] = veh_A2
        else: sim_state['active_veh'] = veh_Ego

def on_motion(event):
    if sim_state['drag_mode'] is None or event.inaxes != ax: return
    veh, mode = sim_state['active_veh'], sim_state['drag_mode']
    sim_state['saved_config'] = None 

    if 'move' in mode:
        veh.position = np.array([event.xdata, event.ydata])
    elif 'speed' in mode:
        veh.speed = min(max(np.hypot(event.xdata - veh.position[0], event.ydata - veh.position[1]), 0.0), 40.0)
        s_slider = sl_a1s if veh == veh_A1 else (sl_a2s if veh == veh_A2 else sl_egos)
        sync_slider_to_physics(s_slider, veh.speed)
    elif 'head' in mode:
        veh.heading = np.arctan2(event.ydata - veh.position[1], event.xdata - veh.position[0]) + np.pi
        h_slider = sl_a1h if veh == veh_A1 else (sl_a2h if veh == veh_A2 else sl_egoh)
        sync_slider_to_physics(h_slider, np.rad2deg(veh.heading) % 360.0)
    update_visuals()

def on_release(event): sim_state['drag_mode'] = sim_state['active_veh'] = None

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# ==========================================
# 7. Playback Controls
# ==========================================
def save_config():
    sim_state['saved_config'] = {
        'A1': (veh_A1.position.copy(), veh_A1.speed, veh_A1.heading),
        'A2': (veh_A2.position.copy(), veh_A2.speed, veh_A2.heading),
        'Ego': (veh_Ego.position.copy(), veh_Ego.speed, veh_Ego.heading)
    }

def on_play(event):
    if not sim_state['is_playing']:
        if sim_state['saved_config'] is None: save_config()
        sim_state['is_playing'] = True

def on_pause(event): 
    sim_state['is_playing'] = False
    update_visuals()

def on_reset(event):
    sim_state['is_playing'] = False
    if sim_state['saved_config']:
        for veh, key, s_sl, h_sl in [(veh_A1, 'A1', sl_a1s, sl_a1h), (veh_A2, 'A2', sl_a2s, sl_a2h), (veh_Ego, 'Ego', sl_egos, sl_egoh)]:
            veh.position, veh.speed, veh.heading = sim_state['saved_config'][key][0].copy(), sim_state['saved_config'][key][1], sim_state['saved_config'][key][2]
            sync_slider_to_physics(s_sl, veh.speed)
            sync_slider_to_physics(h_sl, np.rad2deg(veh.heading))
        update_visuals()

btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)

def simulation_step(frame):
    if sim_state['is_playing']:
        for v in [veh_A1, veh_A2, veh_Ego]: v.step(0.05)
        update_visuals()

ani = FuncAnimation(fig, simulation_step, interval=50, blit=False, cache_frame_data=False)
update_visuals()
plt.show()
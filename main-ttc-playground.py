import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import random

from src.env.risk_calculators import PolygonTTCCalculator


# ==========================================
# 2. Vehicle State Container
# ==========================================
class MockVehicle:
    def __init__(self, x, y, speed, heading):
        self.position = np.array([float(x), float(y)])
        self.speed = float(speed)
        self.heading = np.deg2rad(heading)
        self.LENGTH = 5.0
        self.WIDTH = 2.0
        self.update_velocity()

    def update_velocity(self):
        self.velocity = np.array([self.speed * np.cos(self.heading), self.speed * np.sin(self.heading)])

    def step(self, dt):
        self.position += self.velocity * dt

veh_A = MockVehicle(10, 25, 20, 0)
veh_B = MockVehicle(50, 25, 5, 180)

# ==========================================
# 3. Main Plot Setup
# ==========================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(13, 8))
plt.subplots_adjust(bottom=0.35) 

ax.set_xlim(-10, 100)
ax.set_ylim(-10, 60)
ax.set_aspect('equal')
ax.grid(color='#333333', linestyle='--', linewidth=0.5)
ax.set_title("2D Time-to-Collision (TTC) Simulator", color='white', pad=15, fontsize=16)

# Polygons
poly_A = patches.Polygon([[0,0]], closed=True, fc='#00FFFF', alpha=0.6)
poly_B = patches.Polygon([[0,0]], closed=True, fc='#FF00FF', alpha=0.6)
ax.add_patch(poly_A)
ax.add_patch(poly_B)

# Vectors (Velocity)
arrow_A = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#00FFFF', width=0.004)
arrow_B = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#FF00FF', width=0.004)

# Interactive Handles (Tail Dots for Rotation, Invisible Hitboxes for Tips)
tail_A_plot, = ax.plot([], [], 'o', color='#00FFFF', markersize=8, alpha=0.8)
tail_B_plot, = ax.plot([], [], 'o', color='#FF00FF', markersize=8, alpha=0.8)
tip_A_plot, = ax.plot([], [], 'D', color='#00FFFF', markersize=6, alpha=0.8) 
tip_B_plot, = ax.plot([], [], 'D', color='#FF00FF', markersize=6, alpha=0.8)

# Floating Stats Text
stats_A = ax.text(0, 0, "", color='#00FFFF', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
stats_B = ax.text(0, 0, "", color='#FF00FF', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

# Ray-Casts
ray_lines = [ax.plot([], [], color='yellow', linestyle='--', alpha=0.6, linewidth=1.5)[0] for _ in range(4)]

# Top Status Bar
text_ttc = ax.text(0, 53, "", fontsize=15, fontweight='bold', color='white', bbox=dict(facecolor='black', edgecolor='white', alpha=0.8, pad=5))

# ==========================================
# 4. Controls UI Layout
# ==========================================
sim_state = {
    'is_playing': False,
    'saved_config': None,
    'drag_mode': None,   # 'move_A', 'speed_A', 'head_A', etc.
    'active_veh': None
}

ax_as = fig.add_axes([0.15, 0.20, 0.30, 0.03])
ax_ah = fig.add_axes([0.15, 0.10, 0.30, 0.03])
ax_bs = fig.add_axes([0.60, 0.20, 0.30, 0.03])
ax_bh = fig.add_axes([0.60, 0.10, 0.30, 0.03])

sl_as = Slider(ax_as, 'Veh A - Spd', 0.0, 40.0, valinit=veh_A.speed, color='#00FFFF')
sl_ah = Slider(ax_ah, 'Veh A - Hdg°', 0.0, 360.0, valinit=np.rad2deg(veh_A.heading), color='#00FFFF')
sl_bs = Slider(ax_bs, 'Veh B - Spd', 0.0, 40.0, valinit=veh_B.speed, color='#FF00FF')
sl_bh = Slider(ax_bh, 'Veh B - Hdg°', 0.0, 360.0, valinit=np.rad2deg(veh_B.heading), color='#FF00FF')
sliders = [sl_as, sl_ah, sl_bs, sl_bh]

btn_play = Button(fig.add_axes([0.30, 0.02, 0.10, 0.05]), '▶ Play', color='#2E7D32', hovercolor='#4CAF50')
btn_pause = Button(fig.add_axes([0.42, 0.02, 0.10, 0.05]), '⏸ Pause', color='#F57C00', hovercolor='#FF9800')
btn_reset = Button(fig.add_axes([0.54, 0.02, 0.10, 0.05]), '↺ Reset', color='#1565C0', hovercolor='#2196F3')
btn_rand = Button(fig.add_axes([0.66, 0.02, 0.12, 0.05]), '🎲 Random', color='#444444', hovercolor='#666666')
for btn in [btn_play, btn_pause, btn_reset, btn_rand]: btn.label.set_color('white')

# ==========================================
# 5. Master Visual Update Logic
# ==========================================
def sync_slider_to_physics(slider, val):
    """Silently updates a slider without triggering its callback loop."""
    slider.eventson = False
    slider.set_val(val)
    slider.eventson = True

def update_from_sliders(val=None):
    if sim_state['is_playing']: return
    veh_A.heading = np.deg2rad(sl_ah.val)
    veh_A.speed = sl_as.val
    veh_A.update_velocity()
    
    veh_B.heading = np.deg2rad(sl_bh.val)
    veh_B.speed = sl_bs.val
    veh_B.update_velocity()
    update_visuals()

for s in sliders: s.on_changed(update_from_sliders)

def get_tail_pos(veh):
    return veh.position - 6.0 * np.array([np.cos(veh.heading), np.sin(veh.heading)])

def update_visuals():
    veh_A.update_velocity()
    veh_B.update_velocity()
    ttc = PolygonTTCCalculator.compute_ttc(veh_A, veh_B)
    
    # Update Polygons
    corners_A = PolygonTTCCalculator.get_bounding_box_corners(veh_A.position[0], veh_A.position[1], veh_A.heading, veh_A.LENGTH, veh_A.WIDTH)
    corners_B = PolygonTTCCalculator.get_bounding_box_corners(veh_B.position[0], veh_B.position[1], veh_B.heading, veh_B.LENGTH, veh_B.WIDTH)
    poly_A.set_xy(corners_A)
    poly_B.set_xy(corners_B)
    
    # Update Arrows & Handles
    arrow_A.set_offsets(veh_A.position)
    arrow_A.set_UVC(veh_A.velocity[0], veh_A.velocity[1])
    tip_A_plot.set_data([veh_A.position[0] + veh_A.velocity[0]], [veh_A.position[1] + veh_A.velocity[1]])
    
    arrow_B.set_offsets(veh_B.position)
    arrow_B.set_UVC(veh_B.velocity[0], veh_B.velocity[1])
    tip_B_plot.set_data([veh_B.position[0] + veh_B.velocity[0]], [veh_B.position[1] + veh_B.velocity[1]])

    tail_A = get_tail_pos(veh_A)
    tail_A_plot.set_data([tail_A[0]], [tail_A[1]])
    tail_B = get_tail_pos(veh_B)
    tail_B_plot.set_data([tail_B[0]], [tail_B[1]])

    # Update Stats Text
    stats_A.set_text(f"Spd: {veh_A.speed:.1f}\nHdg: {np.rad2deg(veh_A.heading)%360:.0f}°")
    stats_A.set_position((veh_A.position[0] - 5, veh_A.position[1] + 5))
    stats_B.set_text(f"Spd: {veh_B.speed:.1f}\nHdg: {np.rad2deg(veh_B.heading)%360:.0f}°")
    stats_B.set_position((veh_B.position[0] - 5, veh_B.position[1] + 5))

    # Ray-Casts
    v_rel = veh_A.velocity - veh_B.velocity
    for i in range(4):
        p_start = corners_A[i]
        p_end = p_start + (v_rel * 20.0)
        ray_lines[i].set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])
        ray_lines[i].set_alpha(0 if np.linalg.norm(v_rel) < 1e-5 else 0.7)
            
    # Collision Warning Visuals
    if ttc < 3.0:
        poly_A.set_facecolor('red')
        poly_B.set_facecolor('red')
        text_ttc.set_color('red')
        for ray in ray_lines: ray.set_color('red')
    else:
        poly_A.set_facecolor('#00FFFF')
        poly_B.set_facecolor('#FF00FF')
        text_ttc.set_color('white')
        for ray in ray_lines: ray.set_color('yellow')

    status = "[PLAYING] " if sim_state['is_playing'] else ""
    text_ttc.set_text(f"  {status}Computed TTC: {f'{ttc:.2f} s' if ttc != np.inf else 'Infinity'}  ")
    fig.canvas.draw_idle()

# ==========================================
# 6. Interactive Canvas Dragging Events
# ==========================================
def on_press(event):
    if sim_state['is_playing'] or event.inaxes != ax: return
    
    # Positions to check (Centers, Tips, Tails)
    pts = {
        'speed_A': veh_A.position + veh_A.velocity,
        'head_A': get_tail_pos(veh_A),
        'move_A': veh_A.position,
        'speed_B': veh_B.position + veh_B.velocity,
        'head_B': get_tail_pos(veh_B),
        'move_B': veh_B.position,
    }
    
    closest_mode = None
    min_dist = float('inf')
    
    for mode, pt in pts.items():
        dist = np.hypot(event.xdata - pt[0], event.ydata - pt[1])
        # Give priority to tips/tails over center body dragging
        threshold = 3.5 if 'move' in mode else 2.5
        if dist < threshold and dist < min_dist:
            min_dist = dist
            closest_mode = mode

    if closest_mode:
        sim_state['drag_mode'] = closest_mode
        sim_state['active_veh'] = veh_A if '_A' in closest_mode else veh_B

def on_motion(event):
    if sim_state['drag_mode'] is None or event.inaxes != ax: return
    veh = sim_state['active_veh']
    mode = sim_state['drag_mode']
    
    sim_state['saved_config'] = None # Invalidate reset memory

    if 'move' in mode:
        veh.position = np.array([event.xdata, event.ydata])
        
    elif 'speed' in mode:
        # Distance from mouse to center
        new_speed = np.hypot(event.xdata - veh.position[0], event.ydata - veh.position[1])
        veh.speed = min(max(new_speed, 0.0), 40.0) # Clamp 0 to 40
        slider = sl_as if veh == veh_A else sl_bs
        sync_slider_to_physics(slider, veh.speed)
        
    elif 'head' in mode:
        # Vector from tail dot TO the vehicle center
        dx = veh.position[0] - event.xdata
        dy = veh.position[1] - event.ydata
        veh.heading = np.arctan2(dy, dx)
        slider = sl_ah if veh == veh_A else sl_bh
        sync_slider_to_physics(slider, np.rad2deg(veh.heading) % 360.0)

    update_visuals()

def on_release(event):
    sim_state['drag_mode'] = None
    sim_state['active_veh'] = None

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# ==========================================
# 7. Play/Pause Engine Hooks
# ==========================================
def save_config():
    sim_state['saved_config'] = {
        'A_pos': veh_A.position.copy(), 'A_spd': veh_A.speed, 'A_hdg': veh_A.heading,
        'B_pos': veh_B.position.copy(), 'B_spd': veh_B.speed, 'B_hdg': veh_B.heading
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
        veh_A.position = sim_state['saved_config']['A_pos'].copy()
        veh_A.speed = sim_state['saved_config']['A_spd']
        veh_A.heading = sim_state['saved_config']['A_hdg']
        
        veh_B.position = sim_state['saved_config']['B_pos'].copy()
        veh_B.speed = sim_state['saved_config']['B_spd']
        veh_B.heading = sim_state['saved_config']['B_hdg']
        
        sync_slider_to_physics(sl_as, veh_A.speed)
        sync_slider_to_physics(sl_ah, np.rad2deg(veh_A.heading))
        sync_slider_to_physics(sl_bs, veh_B.speed)
        sync_slider_to_physics(sl_bh, np.rad2deg(veh_B.heading))
        
        sim_state['saved_config'] = None
        update_visuals()

def on_random(event):
    sim_state['is_playing'] = False
    sim_state['saved_config'] = None
    for veh in [veh_A, veh_B]:
        veh.position = np.array([random.uniform(5, 85), random.uniform(5, 45)])
        veh.speed = random.uniform(5, 30)
        veh.heading = np.deg2rad(random.uniform(0, 360))
        
    sync_slider_to_physics(sl_as, veh_A.speed)
    sync_slider_to_physics(sl_ah, np.rad2deg(veh_A.heading))
    sync_slider_to_physics(sl_bs, veh_B.speed)
    sync_slider_to_physics(sl_bh, np.rad2deg(veh_B.heading))
    update_visuals()

btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)
btn_rand.on_clicked(on_random)

def simulation_step(frame):
    if sim_state['is_playing']:
        dt = 0.05
        veh_A.step(dt)
        veh_B.step(dt)
        update_visuals()

ani = FuncAnimation(fig, simulation_step, interval=50, blit=False, cache_frame_data=False)

update_visuals()
plt.show()
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==========================================
# 1. SLIDE-READY BIG FONT CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 3,
    'figure.titlesize': 24
})

OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. TRAJECTORY & METRIC CALCULATION ENGINE
# ==========================================

def calculate_dtw_distance(traj1, traj2):
    """Calculates Dynamic Time Warping (DTW) distance between two (T, 2) trajectories."""
    N, M = len(traj1), len(traj2)
    cost_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            cost_matrix[i, j] = np.linalg.norm(traj1[i] - traj2[j])
            
    dtw = np.zeros((N, M))
    dtw[0, 0] = cost_matrix[0, 0]
    
    for i in range(1, N):
        dtw[i, 0] = dtw[i - 1, 0] + cost_matrix[i, 0]
    for j in range(1, M):
        dtw[0, j] = dtw[0, j - 1] + cost_matrix[0, j]
        
    for i in range(1, N):
        for j in range(1, M):
            dtw[i, j] = cost_matrix[i, j] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            
    return dtw[N - 1, M - 1]

def process_scenario_json(filepath, dt=0.1, ttc_threshold=4.0):
    """Processes a single 40-frame risk situation JSON, safely handling missing adversaries."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    climax = data.get('climax_features', {})
    trajectories = data.get('trajectories', [])
    num_frames = len(trajectories)
    
    if num_frames == 0:
        return None

    # Safely pad climax features if the second adversary is missing entirely
    has_adv2 = 'adv2_rel_x' in climax
    if not has_adv2:
        climax['adv2_rel_x'] = 0.0
        climax['adv2_rel_y'] = 0.0
        climax['adv2_rel_vx'] = 0.0
        climax['inter_adv_dist'] = 0.0
        climax['inter_adv_rel_vel'] = 0.0
    
    adv1_ttcs, adv2_ttcs = [], []
    adv1_dracs, adv2_dracs = [], []
    adv1_positions, adv2_positions = [], []

    for frame in trajectories:
        ego_x, ego_y = frame['ego_x'], frame['ego_y']
        advs = frame.get('advs', [])
        
        # Robust extraction: If an adversary disappears, freeze it at its last known relative position
        if len(advs) > 0:
            adv1_positions.append([advs[0]['x'] - ego_x, advs[0]['y'] - ego_y])
        else:
            adv1_positions.append(adv1_positions[-1] if adv1_positions else [0.0, 0.0])

        if len(advs) > 1:
            adv2_positions.append([advs[1]['x'] - ego_x, advs[1]['y'] - ego_y])
        else:
            adv2_positions.append(adv2_positions[-1] if adv2_positions else [0.0, 0.0])

    adv1_pos_arr = np.array(adv1_positions)
    adv2_pos_arr = np.array(adv2_positions)

    # Reconstruct velocities and accelerations across 40 frames safely
    for i in range(num_frames):
        # Adv 1
        dx1 = adv1_pos_arr[i, 0]
        vx1 = (adv1_pos_arr[i, 0] - adv1_pos_arr[i-1, 0]) / dt if i > 0 else climax.get('adv1_rel_vx', 0.0)
        
        ttc1 = -dx1 / vx1 if (vx1 != 0 and dx1 * vx1 < 0) else 30.0
        ttc1 = max(0.1, min(ttc1, 30.0))
        adv1_ttcs.append(ttc1)
        
        drac1 = (vx1**2) / (2 * abs(dx1)) if (vx1 != 0 and dx1 * vx1 < 0 and abs(dx1) > 0.1) else 0.0
        adv1_dracs.append(drac1)

        # Adv 2
        dx2 = adv2_pos_arr[i, 0]
        vx2 = (adv2_pos_arr[i, 0] - adv2_pos_arr[i-1, 0]) / dt if i > 0 else climax.get('adv2_rel_vx', 0.0)
        
        ttc2 = -dx2 / vx2 if (vx2 != 0 and dx2 * vx2 < 0) else 30.0
        ttc2 = max(0.1, min(ttc2, 30.0))
        adv2_ttcs.append(ttc2)
        
        drac2 = (vx2**2) / (2 * abs(dx2)) if (vx2 != 0 and dx2 * vx2 < 0 and abs(dx2) > 0.1) else 0.0
        adv2_dracs.append(drac2)

    # 1. Time Integrated TTC (TIT) [tau = 4s]
    tit1 = sum([max(0.0, ttc_threshold - ttc) * dt for ttc in adv1_ttcs])
    tit2 = sum([max(0.0, ttc_threshold - ttc) * dt for ttc in adv2_ttcs])
    total_tit = tit1 + tit2

    # 2. Maximum DRAC (MDRAC)
    mdrac = max(max(adv1_dracs, default=0.0), max(adv2_dracs, default=0.0))


    # ======
    # 3. Adversarial Jerk Calculation
    vel1 = np.diff(adv1_pos_arr, axis=0) / dt
    acc1 = np.diff(vel1, axis=0) / dt
    jerk1 = np.diff(acc1, axis=0) / dt
    max_jerk1 = np.max(np.abs(jerk1)) if len(jerk1) > 0 else 0.0

    vel2 = np.diff(adv2_pos_arr, axis=0) / dt
    acc2 = np.diff(vel2, axis=0) / dt
    jerk2 = np.diff(acc2, axis=0) / dt
    max_jerk2 = np.max(np.abs(jerk2)) if len(jerk2) > 0 else 0.0

    metrics = {
        'total_tit': total_tit,
        'mdrac': mdrac,
        'max_jerk': max(max_jerk1, max_jerk2),
        'adv1_traj': adv1_pos_arr,
        'adv2_traj': adv2_pos_arr,
        'has_adv2': float(has_adv2) # Stored for clustering
    }
    # =======
    
    # Inside process_scenario_json:
    vel1 = np.diff(adv1_pos_arr, axis=0) / dt
    acc1 = np.diff(vel1, axis=0) / dt
    max_acc1 = np.max(np.abs(acc1)) if len(acc1) > 0 else 0.0

    vel2 = np.diff(adv2_pos_arr, axis=0) / dt
    acc2 = np.diff(vel2, axis=0) / dt
    max_acc2 = np.max(np.abs(acc2)) if len(acc2) > 0 else 0.0

    max_acc = max(max_acc1, max_acc2)


    # ===
    # --- ROBUST KINEMATICS CALCULATION ---
    # Calculate velocities (diff of position)
    vel1 = np.diff(adv1_pos_arr, axis=0) / dt
    vel2 = np.diff(adv2_pos_arr, axis=0) / dt

    # Filter 1: If velocity exceeds realistic bounds (e.g., > 50 m/s relative), it was an index shift.
    # We replace those glitch frames with 0.0 to stop the derivative explosion.
    vel1 = np.where(np.abs(vel1) > 50.0, 0.0, vel1)
    vel2 = np.where(np.abs(vel2) > 50.0, 0.0, vel2)

    # Calculate accelerations
    acc1 = np.diff(vel1, axis=0) / dt
    acc2 = np.diff(vel2, axis=0) / dt
    
    # Filter 2: The agent's action space is (-4.0, 1.0). Even accounting for ego braking, 
    # a relative acceleration > 15.0 m/s^2 is a collision rubber-band glitch.
    acc1 = np.where(np.abs(acc1) > 15.0, 0.0, acc1)
    acc2 = np.where(np.abs(acc2) > 15.0, 0.0, acc2)

    # Calculate jerk safely
    jerk1 = np.diff(acc1, axis=0) / dt
    jerk2 = np.diff(acc2, axis=0) / dt
    
    max_jerk1 = np.max(np.abs(jerk1)) if len(jerk1) > 0 else 0.0
    max_jerk2 = np.max(np.abs(jerk2)) if len(jerk2) > 0 else 0.0
    max_jerk = max(max_jerk1, max_jerk2)
    
    max_acc = max(np.max(np.abs(acc1)) if len(acc1) > 0 else 0.0, 
                  np.max(np.abs(acc2)) if len(acc2) > 0 else 0.0)
    # -------------------------------------
    # ===


    metrics = {
        'total_tit': total_tit,
        'mdrac': mdrac,
        'max_jerk': max(max_jerk1, max_jerk2), # You can keep this for observation
        'max_acc': max_acc,                    # Add this new metric!
        'adv1_traj': adv1_pos_arr,
        'adv2_traj': adv2_pos_arr,
        'has_adv2': float(has_adv2)
    }

    row = {**climax, **metrics}
    return row


def load_and_evaluate_dataset(folder="risk_events"):
    """Loads all JSON files, computes metrics, and executes K-Means analysis."""
    search_pattern = os.path.join(folder, "**", "*.json")

    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print(f"Warning: No JSON files found in '{folder}'. Generating synthetic demo dataset.")
        os.makedirs(folder, exist_ok=True)
        generate_synthetic_data(folder, count=30)
        json_files = glob.glob(os.path.join(folder, "*.json"))

    rows = []
    for fp in json_files:
        rows.append(process_scenario_json(fp))
        
    df = pd.DataFrame(rows)
    return df

# Helper to generate synthetic test data if folder is empty
def generate_synthetic_data(folder, count=30):
    for i in range(count):
        demo = {
            "climax_features": {
                "adv1_rel_x": float(np.random.uniform(5, 20)),
                "adv1_rel_y": float(np.random.choice([-3.5, 0.0, 3.5])),
                "adv1_rel_vx": float(np.random.uniform(-8, -2)),
                "adv1_rel_vy": 0.02,
                "ego_abs_vel": 24.5,
                "ego_lane_id": 0,
                "collective_reward": float(np.random.uniform(-18, -5)),
                "raw_individual_rewards": [-14.2, -10.706],
                "adv2_rel_x": float(np.random.uniform(-25, -5)),
                "adv2_rel_y": float(np.random.choice([-3.5, 0.0, 3.5])),
                "adv2_rel_vx": float(np.random.uniform(1, 6)),
                "inter_adv_dist": float(np.random.uniform(15, 45)),
                "inter_adv_rel_vel": float(np.random.uniform(3, 12))
            },
            "trajectories": [
                {
                    "ego_x": 100.0 + t*2.5,
                    "ego_y": 0.0,
                    "advs": [
                        {"x": 100.0 + t*2.5 + (20 - t*0.3), "y": 0.0},
                        {"x": 100.0 + t*2.5 - (15 - t*0.2), "y": -3.5}
                    ]
                } for t in range(40)
            ]
        }
        with open(os.path.join(folder, f"scenario_{i+1}.json"), 'w') as f:
            json.dump(demo, f, indent=4)

# ==========================================
# 3. K-MEANS & STATISTICAL ANALYSIS
# ==========================================


def perform_clustering_and_anova(df, n_clusters=3):
    """Clusters scenarios and executes ANOVA + Tukey's HSD post-hoc tests."""
    df['is_sandwich'] = ((df['adv1_rel_x'] > 0) & (df['adv2_rel_x'] < 0)).astype(float)
    
    # We add 'has_adv2' to the K-means features to force separation of 1-adv scenarios
    if 'has_adv2' not in df.columns:
        df['has_adv2'] = (df['inter_adv_dist'] > 0.1).astype(float)
        
    feature_cols = ['adv1_rel_x', 'adv1_rel_vx', 'adv2_rel_x', 'adv2_rel_vx', 'inter_adv_dist', 'is_sandwich', 'has_adv2']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # Feature weighting
    X_scaled[:, 0] *= 2.0  
    X_scaled[:, 2] *= 2.0  
    X_scaled[:, 4] *= 2.0  
    X_scaled[:, 5] *= 3.0  
    X_scaled[:, 6] *= 20.0 # Massive weight guarantees 1-adversary events cluster together
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    print("\n" + "="*50)
    print("      ONE-WAY ANOVA & TUKEY HSD POST-HOC RESULTS")
    print("="*50)
    
    test_variables = ['total_tit', 'mdrac', 'inter_adv_dist', 'collective_reward', 'max_jerk']
    
    for var in test_variables:
        groups = [df[df['cluster'] == c][var].values for c in range(n_clusters)]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            print(f"\nANOVA for Variable: '{var}' -> F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
            
            if p_val < 0.05:
                print(f" -> Statistically Significant Difference across clusters (p < 0.05)!")
                tukey = pairwise_tukeyhsd(endog=df[var], groups=df['cluster'], alpha=0.05)
                print(tukey)
            else:
                print(" -> No significant difference detected across clusters.")
                
    return df

# ==========================================
# 4. PLOTTING FUNCTIONS (SAVED TO PNG)
# ==========================================

def plot_climax_scatter(df, n_clusters=3):
    """Generates and saves the spatial highway snapshot for all clusters."""
    fig, axes = plt.subplots(1, n_clusters, figsize=(7 * n_clusters, 6), sharey=True)
    if n_clusters == 1: axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        ax.scatter(0, 0, color='red', marker='*', s=450, edgecolor='black', label='Ego (Fixed)', zorder=5)
        
        cluster_df = df[df['cluster'] == cluster_id]
        ax.scatter(cluster_df['adv1_rel_x'], cluster_df['adv1_rel_y'], color='blue', alpha=0.6, s=100, label='Adv 1 Climax')
        ax.scatter(cluster_df['adv2_rel_x'], cluster_df['adv2_rel_y'], color='orange', alpha=0.6, s=100, label='Adv 2 Climax')

        ax.set_title(f"Cluster {cluster_id} Climax", fontweight='bold')
        ax.set_xlabel("Longitudinal Distance (m)")
        if cluster_id == 0:
            ax.set_ylabel("Lateral Distance (m)")
        
        ax.set_xlim(-40, 40)
        ax.set_ylim(6, -6) 
        ax.grid(True, linestyle='--', alpha=0.5)
        if cluster_id == n_clusters - 1:
            ax.legend(loc='upper right')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "climax_scatter.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_cluster_spider_plot(df, n_clusters=3):
    """Consolidates metrics & reward functions into a unified Spider/Radar Plot."""
    metrics_to_plot = [
        ('total_tit', 'TIT (s)'),
        ('mdrac', 'MDRAC (m/s²)'),
        ('inter_adv_dist', 'Inter-Adv Dist (m)'),
        ('inter_adv_rel_vel', 'Inter Rel Vel (m/s)'),
        ('collective_reward', 'Reward'),
        ('is_sandwich', 'Sandwich %')
    ]
    
    col_keys = [m[0] for m in metrics_to_plot]
    col_labels = [m[1] for m in metrics_to_plot]
    
    cluster_means = df.groupby('cluster')[col_keys].mean().reset_index()
    
    # Normalize features between 0 and 1 for the spider plot
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(cluster_means[col_keys])
    
    num_vars = len(col_keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(n_clusters):
        values = normalized_values[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=3, color=colors[i % len(colors)], label=f'Cluster {i}')
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(col_labels, size=18, fontweight='bold')
    ax.set_title("Consolidated Cluster Metric & Reward Profiles", y=1.08, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    
    filepath = os.path.join(OUTPUT_DIR, "cluster_spider_plot.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_behavior_dtw_heatmaps(df):
    """Computes and plots separate DTW distance heatmaps for specific meta-behavior groups."""
    
    # Define the meta-behaviors and their corresponding clusters
    behavior_groups = {
        # "Cruise_Blocking": [4],# [4, 8, 11],
        # "Walling": [3,7],#[1, 3, 7],
        # "Intentional_Fratricide": [9],
        # "Sandwiching": [6],#[5, 6, 10]
        "Cruise_Blocking_11": [11],
        "Cruise_Blocking_4": [4],
        "Cruise_Blocking_8": [8],
        "Walling_3": [3],
        "Walling_1": [1],
        "Walling_7": [7],
        "Intentional_Fratricide_9": [9],
        "Sandwiching_6": [6],
        "Sandwiching_5": [5],
        "Sandwiching_10": [10]
    }
    
    for behavior_name, clusters in behavior_groups.items():
        # Filter the dataframe for the specific clusters in this behavior
        behavior_df = df[df['cluster'].isin(clusters)].reset_index(drop=True)
        
        num_samples = len(behavior_df)
        if num_samples == 0:
            print(f"Skipping {behavior_name.replace('_', ' ')}: No valid scenarios found in these clusters.")
            continue
            
        # Cap the number of samples to prevent extremely long computation times
        num_samples = min(30, num_samples)
        sample_df = behavior_df.iloc[:num_samples]
        
        dtw_matrix = np.zeros((num_samples, num_samples))

        
        
        print(f"Calculating DTW for {behavior_name.replace('_', ' ')} (n={num_samples})...")
        for i in range(num_samples):
            for j in range(num_samples):
                dist1 = calculate_dtw_distance(sample_df.loc[i, 'adv1_traj'], sample_df.loc[j, 'adv1_traj'])
                dist2 = calculate_dtw_distance(sample_df.loc[i, 'adv2_traj'], sample_df.loc[j, 'adv2_traj'])
                dtw_matrix[i, j] = dist1 + dist2

        # Extract the upper triangle of the matrix, excluding the diagonal (k=1)
        upper_triangle_indices = np.triu_indices_from(dtw_matrix, k=1)
        pairwise_distances = dtw_matrix[upper_triangle_indices]
        
        if len(pairwise_distances) > 0:
            mean_dtw = np.mean(pairwise_distances)
            std_dtw = np.std(pairwise_distances)
            max_dtw = np.max(pairwise_distances)
            
            print(f"--- Diversity Metrics for {behavior_name} ---")
            print(f"Mean Pairwise DTW: {mean_dtw:.2f}")
            print(f"Std Dev of DTW:    {std_dtw:.2f}")
            print(f"Max DTW Distance:  {max_dtw:.2f}\n")
                
        # Generate and save the heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(dtw_matrix, cmap='viridis', origin='upper')
        plt.colorbar(im, label='DTW Trajectory Distance')
        plt.title(f"Intra-Class Diversity: {behavior_name.replace('_', ' ')}", fontweight='bold')
        plt.xlabel("Scenario Index")
        plt.ylabel("Scenario Index")
        
        filepath = os.path.join(OUTPUT_DIR, f"dtw_heatmap_{behavior_name.lower()}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")

def plot_statistical_bar_charts(df, n_clusters=3):
    """Plots comparative bar charts for TIT, MDRAC, Jerk, and Rewards per cluster."""
    cluster_stats = df.groupby('cluster')[['total_tit', 'mdrac', 'max_jerk', 'collective_reward']].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # TIT Chart
    axes[0, 0].bar(cluster_stats.index.astype(str), cluster_stats['total_tit'], color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Mean Time Integrated TTC (TIT)")
    axes[0, 0].set_ylabel("TIT (seconds)")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # MDRAC Chart
    axes[0, 1].bar(cluster_stats.index.astype(str), cluster_stats['mdrac'], color='salmon', edgecolor='black')
    axes[0, 1].set_title("Mean Maximum DRAC")
    axes[0, 1].set_ylabel("DRAC (m/s²)")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Jerk Chart (Plausibility)
    axes[1, 0].bar(cluster_stats.index.astype(str), cluster_stats['max_jerk'], color='lightgreen', edgecolor='black')
    axes[1, 0].axhline(y=5.0, color='red', linestyle='--', label='Plausibility Limit (5.0 m/s³)')
    axes[1, 0].set_title("Max Adversarial Jerk")
    axes[1, 0].set_ylabel("Jerk (m/s³)")
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Collective Reward
    axes[1, 1].bar(cluster_stats.index.astype(str), cluster_stats['collective_reward'], color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Mean Collective Reward")
    axes[1, 1].set_ylabel("Reward Value")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "statistical_metrics_summary.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def print_cluster_summary(df):
    """Prints a clean tabular summary of mean metrics per cluster to the terminal."""
    summary_df = df.groupby('cluster').agg(
        Sample_Count=('total_tit', 'count'),
        Mean_Reward=('collective_reward', 'mean'),
        Max_Reward=('collective_reward', 'max'),
        Mean_TIT=('total_tit', 'mean'),
        Mean_MDRAC=('mdrac', 'mean'),
        Mean_Max_Jerk=('max_jerk', 'mean'),
        Sandwich_Ratio=('is_sandwich', 'mean')
    ).reset_index()

    print("\n" + "="*65)
    print("                CLUSTER METRICS SUMMARY TABLE")
    print("="*65)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("="*65 + "\n")

# ==========================================
# 5. MAIN EXECUTION ENTRY POINT
# ==========================================

# if __name__ == "__main__":
#     print("Starting MASAC Risk Scenario Analysis...")
    
#     # 1. Load data and process trajectories
#     df = load_and_evaluate_dataset(folder="risk_events")
    
#     # 2. Run K-Means and ANOVA / Tukey Tests
#     N_CLUSTERS = 10
#     df = perform_clustering_and_anova(df, n_clusters=N_CLUSTERS)
    
#     # 3. Output Figures (Saved directly as high-res PNGs)
#     print("\nGenerating slide-ready figures...")
#     plot_climax_scatter(df, n_clusters=N_CLUSTERS)
#     plot_cluster_spider_plot(df, n_clusters=N_CLUSTERS)
#     plot_dtw_trajectory_heatmap(df)
#     plot_statistical_bar_charts(df, n_clusters=N_CLUSTERS)

#     print_cluster_summary(df)
    
#     print(f"\nAll analyses complete! Check the '{OUTPUT_DIR}' directory for figures.")

if __name__ == "__main__":
    print("Starting MASAC Risk Scenario Analysis...")
    
    # 1. Load data and process trajectories
    df = load_and_evaluate_dataset(folder="risk_events")
    
    # --- THE NEW FILTER ---
    original_count = len(df)

    # Keep only the rows where 'max_acc' is physically plausible (under 10 m/s^2)
    #df = df[df['max_acc'] <= 10.0].copy() 
    
    filtered_count = len(df)
    print(f"\n[Physics Filter Applied] Removed {original_count - filtered_count} physically impossible scenarios (Acceleration > 10.0 m/s²).")
    
    # 2. Run K-Means and ANOVA / Tukey Tests
    N_CLUSTERS = 12
    
    # Safety check: ensure we still have enough data to cluster after filtering
    if len(df) >= N_CLUSTERS:
        df = perform_clustering_and_anova(df, n_clusters=N_CLUSTERS)
        
        # Call the summary printer
        print_cluster_summary(df)
        
        # 3. Output Figures (Saved directly as high-res PNGs)
        print("\nGenerating slide-ready figures...")
        plot_climax_scatter(df, n_clusters=N_CLUSTERS)
        plot_cluster_spider_plot(df, n_clusters=N_CLUSTERS)
        
        # Call the newly created behavior-specific DTW mapping function
        plot_behavior_dtw_heatmaps(df) 
        
        plot_statistical_bar_charts(df, n_clusters=N_CLUSTERS)
        
        print(f"\nAll analyses complete! Check the '{OUTPUT_DIR}' directory for figures.")
    else:
        print(f"\nError: Not enough plausible scenarios left to form {N_CLUSTERS} clusters. Consider adjusting your RL reward function to penalize high jerk during training.")
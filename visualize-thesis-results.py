import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Globally increase font sizes for presentation/thesis ---
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def load_farmed_data(folder="risk_events"):
    climax_rows = []
    trajectory_data = []
    
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), 'r') as f:
                data = json.load(f)
                climax_rows.append(data['climax_features'])
                trajectory_data.append(data['trajectories'])
                
    return pd.DataFrame(climax_rows), trajectory_data

def generate_visualizations(n_clusters=3):
    df, trajectories = load_farmed_data()
    if len(df) == 0:
        print("No risk events found. Run training first!")
        return

    # 1. PURGE THE NOISE
    valid_indices = df['inter_adv_dist'] > 0.1
    df = df[valid_indices].copy()
    
    if len(df) < n_clusters:
        print("Not enough valid multi-agent events to cluster!")
        return

    # 2. FEATURE ENGINEERING
    df['is_sandwich'] = ((df['adv1_rel_x'] > 0) & (df['adv2_rel_x'] < 0)).astype(float)

    feature_cols = [
        'adv1_rel_x',
        'adv1_rel_vx',
        'adv2_rel_x',
        'adv2_rel_vx',
        'inter_adv_dist',
        'is_sandwich'
    ]
    
    # 3. K-MEANS CLUSTERING WITH WEIGHTS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    X_scaled[:, 0] *= 2.0  
    X_scaled[:, 2] *= 2.0  
    X_scaled[:, 4] *= 2.0  
    X_scaled[:, 5] *= 3.0  
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. GENERATE TRAJECTORY HEATMAPS
    fig, axes = plt.subplots(1, n_clusters, figsize=(7 * n_clusters, 6), sharey=True)
    if n_clusters == 1: axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        # Plot Ego Vehicle at the center
        ax.plot(0, 0, 'r*', markersize=24, markeredgecolor='black', label='Ego (Fixed)', zorder=5)
        
        cluster_indices = df.index[df['cluster'] == cluster_id].tolist()
        
        for idx in cluster_indices:
            episode_traj = trajectories[idx]
            adv1_x, adv1_y, adv2_x, adv2_y = [], [], [], []
            
            for frame in episode_traj:
                ego_x, ego_y = frame['ego_x'], frame['ego_y']
                if len(frame['advs']) >= 2:
                    adv1_x.append(frame['advs'][0]['x'] - ego_x)
                    adv1_y.append(frame['advs'][0]['y'] - ego_y)
                    adv2_x.append(frame['advs'][1]['x'] - ego_x)
                    adv2_y.append(frame['advs'][1]['y'] - ego_y)
            
            ax.plot(adv1_x, adv1_y, color='red', alpha=0.08, linewidth=5)
            ax.plot(adv2_x, adv2_y, color='orange', alpha=0.08, linewidth=5)

        ax.set_title(f"Cluster {cluster_id} Trajectories", fontweight='bold')
        ax.set_xlabel("Longitudinal Distance (m)")
        if cluster_id == 0:
            ax.set_ylabel("Lateral Distance (m)")
        
        ax.set_xlim(-40, 40)
        ax.set_ylim(-2, 4) 
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # --- Custom Legend ---
        if cluster_id == n_clusters - 1:
            custom_lines = [
                Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, markeredgecolor='black'),
                Line2D([0], [0], color='blue', lw=4, alpha=0.6),
                Line2D([0], [0], color='orange', lw=4, alpha=0.6),
                Line2D([0], [0], color='purple', lw=4, alpha=0.6)
            ]
            ax.legend(custom_lines, ['Ego (Fixed)', 'Adv 1', 'Adv 2', 'Dense Overlap (Purple)'], loc='upper right')

    plt.tight_layout()
    plt.savefig("trajectory_heatmaps_large.png", dpi=300)
    print("Saved 'trajectory_heatmaps_large.png'!")
    plt.show()

if __name__ == "__main__":
    generate_visualizations(n_clusters=8)
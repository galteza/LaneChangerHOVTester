import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

    # 1. DATA CLEANING & FILTERING
    # Keep only events where Adversary 2 is actually present/active (distance > 0.1m)
    # df.index is preserved, so trajectories[idx] will still align perfectly
    valid_indices = df['inter_adv_dist'] > 0.1
    df = df[valid_indices].copy()
    
    if len(df) < n_clusters:
        print("Not enough valid multi-agent events to cluster!")
        return

    # 2. FEATURE ISOLATION FOR 1-LANE HIGHWAY
    # Dropping Y-axis noise; K-Means now only judges based on longitudinal tactics
    feature_cols = [
        'adv1_rel_x',
        'adv1_rel_vx',
        'adv2_rel_x',
        'inter_adv_dist'
    ]
    
    # 3. K-MEANS CLUSTERING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. GENERATE TRAJECTORY HEATMAPS
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 6), sharey=True)
    if n_clusters == 1: axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        # Plot Ego Vehicle at the center
        ax.plot(0, 0, 'r*', markersize=20, markeredgecolor='black', label='Ego (Fixed)')
        
        # Find all episodes belonging to this cluster
        cluster_indices = df.index[df['cluster'] == cluster_id].tolist()
        
        # Calculate average collective reward for this specific maneuver
        avg_reward = 0.0
        if 'collective_reward' in df.columns:
            avg_reward = df.loc[df['cluster'] == cluster_id, 'collective_reward'].mean()
        
        # Overlay trajectories to create the heatmap effect
        for idx in cluster_indices:
            episode_traj = trajectories[idx]
            
            # Extract coordinates for Adv 1 and Adv 2 relative to Ego
            adv1_x, adv1_y = [], []
            adv2_x, adv2_y = [], []
            
            for frame in episode_traj:
                ego_x, ego_y = frame['ego_x'], frame['ego_y']
                if len(frame['advs']) >= 2:
                    adv1_x.append(frame['advs'][0]['x'] - ego_x)
                    adv1_y.append(frame['advs'][0]['y'] - ego_y)
                    adv2_x.append(frame['advs'][1]['x'] - ego_x)
                    adv2_y.append(frame['advs'][1]['y'] - ego_y)
            
            # Plot with high transparency (alpha) so overlapping paths create solid colors
            ax.plot(adv1_x, adv1_y, color='blue', alpha=0.08, linewidth=4)
            ax.plot(adv2_x, adv2_y, color='orange', alpha=0.08, linewidth=4)

        # Updated title to display the reward score
        title_text = f"Cluster {cluster_id} Maneuver\nAvg Reward: {avg_reward:.2f}"
        ax.set_title(title_text, fontweight='bold')
        ax.set_xlabel("Longitudinal Distance (m)")
        ax.set_ylabel("Lateral Distance (m)")
        
        # Tightened Y-axis limits for the 1-lane track geometry
        ax.set_xlim(-40, 40)
        ax.set_ylim(-3, 3) 
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend to the first plot only to avoid clutter
        if cluster_id == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("trajectory_heatmaps.png", dpi=300)
    print("Saved 'trajectory_heatmaps.png'!")
    plt.show()

if __name__ == "__main__":
    generate_visualizations(n_clusters=3)
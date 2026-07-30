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

    # 1. K-MEANS CLUSTERING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 2. GENERATE TRAJECTORY HEATMAPS
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 6), sharey=True)
    if n_clusters == 1: axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        # Plot Ego Vehicle at the center
        ax.plot(0, 0, 'r*', markersize=20, markeredgecolor='black', label='Ego (Fixed)')
        
        # Find all episodes belonging to this cluster
        cluster_indices = df.index[df['cluster'] == cluster_id].tolist()
        
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

        ax.set_title(f"Cluster {cluster_id} Emergent Behavior", fontweight='bold')
        ax.set_xlabel("Longitudinal Distance (m)")
        ax.set_ylabel("Lateral Distance (m)")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-10, 10)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("trajectory_heatmaps.png", dpi=300)
    print("Saved 'trajectory_heatmaps.png'!")
    plt.show()

if __name__ == "__main__":
    generate_visualizations(n_clusters=3)
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), 'r') as f:
                data = json.load(f)
                climax_rows.append(data['climax_features'])
                
    return pd.DataFrame(climax_rows)

def generate_climax_scatter(n_clusters=3):
    df = load_farmed_data()
    if len(df) == 0:
        print("No risk events found.")
        return

    valid_indices = df['inter_adv_dist'] > 0.1
    df = df[valid_indices].copy()
    
    if len(df) < n_clusters:
        print("Not enough valid multi-agent events.")
        return

    # Same K-Means prep to ensure clusters match the heatmap script perfectly
    df['is_sandwich'] = ((df['adv1_rel_x'] > 0) & (df['adv2_rel_x'] < 0)).astype(float)
    feature_cols = ['adv1_rel_x', 'adv1_rel_vx', 'adv2_rel_x', 'adv2_rel_vx', 'inter_adv_dist', 'is_sandwich']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    X_scaled[:, 0] *= 2.0  
    X_scaled[:, 2] *= 2.0  
    X_scaled[:, 4] *= 2.0  
    X_scaled[:, 5] *= 3.0  
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Scatter Plotting
    fig, axes = plt.subplots(1, n_clusters, figsize=(7 * n_clusters, 6), sharey=True)
    if n_clusters == 1: axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        # Plot Ego
        ax.scatter(0, 0, color='red', marker='*', s=400, edgecolor='black', label='Ego (Fixed)', zorder=5)
        
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Plot Climax Points with solid dots
        ax.scatter(cluster_df['adv1_rel_x'], cluster_df['adv1_rel_y'], 
                   color='blue', alpha=0.5, s=80, label='Adv 1 Climax')
        ax.scatter(cluster_df['adv2_rel_x'], cluster_df['adv2_rel_y'], 
                   color='orange', alpha=0.5, s=80, label='Adv 2 Climax')

        ax.set_title(f"Cluster {cluster_id} Climax Points", fontweight='bold')
        ax.set_xlabel("Longitudinal Distance (m)")
        if cluster_id == 0:
            ax.set_ylabel("Lateral Distance (m)")
        
        ax.set_xlim(-40, 40)
        ax.set_ylim(6, -6) 
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if cluster_id == n_clusters - 1:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("climax_scatter.png", dpi=300)
    print("Saved 'climax_scatter.png'!")
    plt.show()

if __name__ == "__main__":
    generate_climax_scatter(n_clusters=17)
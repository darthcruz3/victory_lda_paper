# LDA model (8 groups), LDA-dataframe (components 1 & 2), and Bhattacharyya distances
# Author: Dr. Leon G. D'Cruz - Portsmouth Hospitals University NHS Trust

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE

# Set working directory
os.chdir(r"C:\WORKFILES\PEOPLE\Leon_DCruz\projects\projects\VICTORY\VICTORY_LDA_paper\spss\csv")

# Load dataset
dataset = pd.read_csv('lauren_merged_onlyessentials_groupsorted_3groups_LCA3Pneu2_others1.csv')
dataset = dataset.reset_index(drop=True)

# Features & Target
X = dataset.iloc[:, np.r_[8:9, 10, 11, 12, 15, 16]].values
y = dataset.iloc[:, -2]

# Step 1: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 2: Apply SMOTE on training set
smote = SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 3: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Fit LDA model
lda_model = LDA(solver='svd')

skf = StratifiedKFold(n_splits=10)
all_y_test, all_y_pred = [], []

for train_index, val_index in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    lda_model.fit(X_train_fold, y_train_fold)
    y_pred = lda_model.predict(X_val_fold)

    all_y_test.extend(y_val_fold)
    all_y_pred.extend(y_pred)

# Project to LDA space
X_lda = lda_model.transform(X)
df_lda = pd.DataFrame(X_lda, columns=[f'Component {i+1}' for i in range(X_lda.shape[1])])
df_lda['Group_ID'] = y.values


# ---------- Bhattacharyya Distance Computation ----------

def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """Compute the Bhattacharyya distance between two Gaussian distributions."""
    cov_mean = (cov1 + cov2) / 2
    inv_cov_mean = np.linalg.inv(cov_mean)
    diff = mu1 - mu2
    term1 = 0.5 * np.log(
        np.linalg.det(cov_mean) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )
    term2 = 0.25 * diff.T @ inv_cov_mean @ diff
    return term1 + term2


def compute_group_statistics(df_lda):
    """Compute mean and covariance for each group in LDA space."""
    groups = df_lda['Group_ID'].unique()
    group_stats = {}
    for group in groups:
        group_data = df_lda[df_lda['Group_ID'] == group][['Component 1', 'Component 2']]
        mu = group_data.mean().values
        cov = np.cov(group_data.T, ddof=0)
        group_stats[group] = (mu, cov)
    return group_stats


def compute_bhattacharyya_distances(group_stats):
    """Compute pairwise Bhattacharyya distances between groups."""
    groups = list(group_stats.keys())
    distances = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            g1, g2 = groups[i], groups[j]
            mu1, cov1 = group_stats[g1]
            mu2, cov2 = group_stats[g2]
            dist = bhattacharyya_distance(mu1, cov1, mu2, cov2)
            distances.loc[g1, g2] = dist
            distances.loc[g2, g1] = dist
    return distances


# Compute group statistics and distances
group_stats_lda = compute_group_statistics(df_lda)
bhattacharyya_distances_lda = compute_bhattacharyya_distances(group_stats_lda)

# Save distances to CSV
os.chdir(r"C:\WORKFILES\PEOPLE\Leon_DCruz\projects\projects\VICTORY\VICTORY_LDA_paper\pythons_den\csv\Bhattacharyya_distances")
bhattacharyya_distances_lda.to_csv('bhattacharyya_distances_lda_8grpmodel.csv')


# ---------- Network Plot of Bhattacharyya Distances ----------

def plot_network(distances, title, filename=None):
    """Create and optionally save a network diagram based on Bhattacharyya distances."""
    G = nx.Graph()
    for group in distances.index:
        G.add_node(group)

    for i in range(len(distances.index)):
        for j in range(i + 1, len(distances.index)):
            g1, g2 = distances.index[i], distances.index[j]
            weight = distances.loc[g1, g2]
            if not np.isnan(weight):
                G.add_edge(g1, g2, weight=weight * 1000)

    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    if weights:
        weight_max = max(weights)
        weights = [w / weight_max * 10 for w in weights]

    plt.figure(figsize=(12, 10))
    nx.draw(
        G, pos, with_labels=True, node_color='lightblue', node_size=3000,
        edge_color='gray', font_size=12, font_weight='bold',
        width=weights, alpha=0.7
    )

    edge_labels = {(u, v): f'{d["weight"] / 1000:.2f}' for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title(title)

    if filename:
        plt.savefig(filename, format=os.path.splitext(filename)[1][1:], dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

    plt.show()


# Plot and save network
os.chdir(r"C:\WORKFILES\PEOPLE\Leon_DCruz\projects\projects\VICTORY\VICTORY_LDA_paper\pythons_den\figures")
plot_network(
    bhattacharyya_distances_lda,
    'Network Diagram of Bhattacharyya Distances - LDA 8-group model',
    filename='lda_bhattacharyya_network_LDA_8group_model.png'
)
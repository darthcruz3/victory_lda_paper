# LDA Model with 3 Diagnostic Groups and Bhattacharyya Distance Analysis
# Author: Dr. Leon G. D'Cruz - Portsmouth Hospitals University NHS Trust

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from imblearn.over_sampling import SMOTE

from scipy.spatial.distance import mahalanobis


# ========== Step 1: Load and Prepare Dataset ========== #

# Set your path or make it configurable
DATA_PATH = r"C:\WORKFILES\PEOPLE\Leon_DCruz\projects\projects\VICTORY\VICTORY_LDA_paper\spss\csv"
FILENAME = "lauren_merged_onlyessentials_groupsorted_3groups_LCA3Pneu2_others1.csv"
os.chdir(DATA_PATH)

dataset = pd.read_csv(FILENAME).reset_index(drop=True)
X = dataset.iloc[:, np.r_[8:9, 10, 11, 12, 15, 16]].values
y = dataset.iloc[:, -1]


# ========== Step 2: Train-Test Split, SMOTE, and Scaling ========== #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

smote = SMOTE(k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


# ========== Step 3: Fit LDA Model Using Cross-Validation ========== #

lda_model = LDA(solver='svd')
skf = StratifiedKFold(n_splits=10)

all_y_test = []
all_y_pred = []

for train_index, val_index in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    lda_model.fit(X_train_fold, y_train_fold)
    y_pred = lda_model.predict(X_val_fold)

    all_y_test.extend(y_val_fold)
    all_y_pred.extend(y_pred)

all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)


# ========== Step 4: Project Data into LDA Space ========== #

X_lda = lda_model.transform(X)
df_lda = pd.DataFrame(X_lda, columns=[f'Component {i+1}' for i in range(X_lda.shape[1])])
df_lda['Group'] = y.values


# ========== Step 5: Bhattacharyya Distance Calculations ========== #

def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """Compute Bhattacharyya distance between two Gaussian distributions."""
    cov_mean = (cov1 + cov2) / 2
    try:
        inv_cov_mean = np.linalg.inv(cov_mean)
    except np.linalg.LinAlgError:
        inv_cov_mean = np.linalg.pinv(cov_mean)

    diff = mu1 - mu2
    term1 = 0.125 * diff.T @ inv_cov_mean @ diff
    term2 = 0.5 * np.log(np.linalg.det(cov_mean) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    return term1 + term2


def compute_group_statistics(df_lda):
    """Compute mean and covariance matrix for each class."""
    groups = df_lda['Group'].unique()
    stats = {}
    for group in groups:
        group_data = df_lda[df_lda['Group'] == group][['Component 1', 'Component 2']]
        mu = group_data.mean().values
        cov = np.cov(group_data.T, ddof=0)
        stats[group] = (mu, cov)
    return stats


def compute_bhattacharyya_distances(stats):
    """Pairwise Bhattacharyya distances between all groups."""
    groups = list(stats.keys())
    dist_matrix = pd.DataFrame(index=groups, columns=groups)
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            g1, g2 = groups[i], groups[j]
            mu1, cov1 = stats[g1]
            mu2, cov2 = stats[g2]
            dist = bhattacharyya_distance(mu1, cov1, mu2, cov2)
            dist_matrix.loc[g1, g2] = dist
            dist_matrix.loc[g2, g1] = dist
    return dist_matrix


# Compute and save
group_stats = compute_group_statistics(df_lda)
bhatt_distances = compute_bhattacharyya_distances(group_stats)

# Save distances
OUTPUT_PATH = r"C:\WORKFILES\PEOPLE\Leon_DCruz\projects\projects\VICTORY\VICTORY_LDA_paper\pythons_den\csv\Bhattacharyya_distances"
os.makedirs(OUTPUT_PATH, exist_ok=True)
bhatt_distances.to_csv(os.path.join(OUTPUT_PATH, "bhattacharyya_distances_lda_3grpmodel.csv"))


# ========== Step 6: Network Plot of Bhattacharyya Distances ========== #

def plot_network(distances, title):
    """Plot network graph of Bhattacharyya distances."""
    G = nx.Graph()
    for node in distances.index:
        G.add_node(node)

    for i in range(len(distances.index)):
        for j in range(i + 1, len(distances.index)):
            g1, g2 = distances.index[i], distances.index[j]
            weight = distances.loc[g1, g2]
            if not np.isnan(weight):
                G.add_edge(g1, g2, weight=weight * 1000)

    pos = nx.spring_layout(G, seed=42)
    weights = [d['weight'] for (_, _, d) in G.edges(data=True)]
    scaled_weights = [w / max(weights) * 10 for w in weights]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000,
            edge_color='gray', width=scaled_weights, font_size=12, font_weight='bold', alpha=0.7)

    edge_labels = {(u, v): f'{d["weight"] / 1000:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_network(bhatt_distances, "Network Diagram of Bhattacharyya Distances - LDA Components")
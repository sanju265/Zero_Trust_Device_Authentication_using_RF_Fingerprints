"""
Graph Construction module.
Builds a device similarity graph from RF fingerprint features.
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from config import TRANSMITTER_IDS


def build_similarity_graph(scaled_train_features):
    """
    Build a device similarity graph where:
    - Nodes = transmitters
    - Edges = similarity between RF fingerprint centroids

    Uses RBF (Gaussian) kernel for non-negative adjacency weights:
        w_ij = exp(-||c_i - c_j||^2 / (2 * sigma^2))

    Returns:
        adj_matrix: (N, N) numpy adjacency matrix (non-negative)
        centroids:  (N, n_features) per-device mean feature vectors
        G:          NetworkX graph for visualization
    """
    # Compute per-device centroid (mean feature vector)
    centroids = []
    for tx_id in TRANSMITTER_IDS:
        if tx_id in scaled_train_features:
            values = scaled_train_features[tx_id]
            if values.ndim == 3:
                centroid = np.mean(values, axis=(0, 1))
            else:
                centroid = np.mean(values, axis=0)
            centroids.append(centroid)
    centroids = np.array(centroids, dtype=np.float32)

    # Compute pairwise Euclidean distances
    dist_matrix = euclidean_distances(centroids)

    # RBF kernel: exp(-d^2 / (2*sigma^2))
    # sigma = median distance (robust choice)
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    sigma = float(np.median(upper_tri)) if len(upper_tri) > 0 else 1.0
    sigma = max(sigma, 0.1)  # prevent zero sigma

    adj_matrix = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
    # adj_matrix is in (0, 1], with 1 on diagonal

    # Build NetworkX graph for visualization
    G = nx.Graph()
    for i, tx_id in enumerate(TRANSMITTER_IDS):
        G.add_node(tx_id, centroid=centroids[i])

    for i in range(len(TRANSMITTER_IDS)):
        for j in range(i + 1, len(TRANSMITTER_IDS)):
            weight = float(adj_matrix[i, j])
            if weight > 0.1:  # Only add meaningful edges
                G.add_edge(TRANSMITTER_IDS[i], TRANSMITTER_IDS[j], weight=round(weight, 3))

    print(f"[Graph] Built similarity graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  RBF sigma={sigma:.3f}, adj range=[{adj_matrix.min():.3f}, {adj_matrix.max():.3f}]")

    return adj_matrix, centroids, G


def build_temporal_graph(scaled_train_features):
    """
    Build a 256 x 256 temporal-spectral graph for packet-level TSGAE training.

    Nodes represent packet time steps, and edges capture similarity between
    their mean RF feature profiles across the training set.
    """
    step_means = []
    first_tx = TRANSMITTER_IDS[0]
    n_steps = scaled_train_features[first_tx].shape[1]

    for step_idx in range(n_steps):
        values = []
        for tx_id in TRANSMITTER_IDS:
            values.append(scaled_train_features[tx_id][:, step_idx, :])
        step_means.append(np.mean(np.vstack(values), axis=0))

    step_means = np.array(step_means, dtype=np.float32)
    dist_matrix = euclidean_distances(step_means)

    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    sigma = float(np.median(upper_tri)) if len(upper_tri) > 0 else 1.0
    sigma = max(sigma, 0.1)

    adj_matrix = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))

    # Strengthen short-range temporal continuity.
    for idx in range(n_steps):
        for offset, weight in ((1, 0.95), (2, 0.85), (3, 0.75)):
            if idx + offset < n_steps:
                adj_matrix[idx, idx + offset] = max(adj_matrix[idx, idx + offset], weight)
                adj_matrix[idx + offset, idx] = max(adj_matrix[idx + offset, idx], weight)

    np.fill_diagonal(adj_matrix, 1.0)

    print(f"[Graph] Built temporal graph: {n_steps} nodes")
    print(f"  RBF sigma={sigma:.3f}, adj range=[{adj_matrix.min():.3f}, {adj_matrix.max():.3f}]")

    return adj_matrix, step_means


def get_adjacency_tensor(adj_matrix):
    """Convert adjacency matrix to normalized form for GCN: D^{-1/2} A D^{-1/2}."""
    import torch

    A = adj_matrix.copy()

    # Degree matrix
    D_vec = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_vec + 1e-8))

    # Symmetric normalization
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Safety: replace any remaining NaN/Inf
    A_norm = np.nan_to_num(A_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.FloatTensor(A_norm)

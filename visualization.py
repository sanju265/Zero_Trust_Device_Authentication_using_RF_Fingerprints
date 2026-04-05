"""
Visualization module for generating PPT-ready graphs.
All graphs are saved to the results/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

from config import (
    TRANSMITTER_IDS, RESULTS_DIR, DEVICE_PROFILES,
    TRANSMITTER_LABELS, NUM_FEATURES,
)
from feature_extraction import FEATURE_NAMES


# Color palette for transmitters
TX_COLORS = {
    "TX0": "#2196F3",  # Blue
    "TX1": "#4CAF50",  # Green
    "TX2": "#FF9800",  # Orange
    "TX3": "#F44336",  # Red
    "TX4": "#9C27B0",  # Purple
    "TX5": "#00BCD4",  # Cyan
}


def _draw_annotated_heatmap(ax, data, xlabels, ylabels, cmap="YlOrRd"):
    """Render a heatmap without seaborn so startup does not depend on notebook packages."""
    image = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = float(data[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    return image


def save_fig(fig, name):
    """Save figure to results directory."""
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_raw_iq_signals(dataset, n_samples=3):
    """Plot raw I/Q signal waveforms for each transmitter."""
    fig, axes = plt.subplots(len(TRANSMITTER_IDS), 2, figsize=(14, 3 * len(TRANSMITTER_IDS)))
    fig.suptitle("Raw I/Q Signal Waveforms per Transmitter", fontsize=16, fontweight="bold")

    for i, tx_id in enumerate(TRANSMITTER_IDS):
        if tx_id not in dataset:
            continue
        sample = dataset[tx_id][0]  # First sample
        I, Q = sample[:, 0], sample[:, 1]

        axes[i, 0].plot(I, color=TX_COLORS[tx_id], linewidth=0.8)
        axes[i, 0].set_ylabel(f"{tx_id}\nI-channel")
        axes[i, 0].set_title(f"{tx_id} ({TRANSMITTER_LABELS[tx_id]}) - In-Phase" if i == 0 else "")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(Q, color=TX_COLORS[tx_id], linewidth=0.8, alpha=0.8)
        axes[i, 1].set_ylabel("Q-channel")
        axes[i, 1].set_title(f"{tx_id} ({TRANSMITTER_LABELS[tx_id]}) - Quadrature" if i == 0 else "")
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Sample Index")
    axes[-1, 1].set_xlabel("Sample Index")
    fig.tight_layout()
    save_fig(fig, "01_raw_iq_signals")


def plot_iq_constellation(dataset, n_samples=200):
    """Plot I/Q constellation diagrams for each transmitter."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("I/Q Constellation Diagrams per Transmitter", fontsize=16, fontweight="bold")

    for i, tx_id in enumerate(TRANSMITTER_IDS):
        ax = axes[i // 3, i % 3]
        if tx_id not in dataset:
            continue
        # Plot multiple samples overlaid
        for j in range(min(n_samples, dataset[tx_id].shape[0])):
            sample = dataset[tx_id][j]
            ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.05, color=TX_COLORS[tx_id])

        ax.set_title(f"{tx_id} ({DEVICE_PROFILES[tx_id]['name']})")
        ax.set_xlabel("I (In-Phase)")
        ax.set_ylabel("Q (Quadrature)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, "02_iq_constellation")


def plot_feature_distributions(train_features):
    """Plot distribution of each feature across transmitters."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("RF Fingerprint Feature Distributions per Transmitter", fontsize=16, fontweight="bold")

    for f_idx in range(NUM_FEATURES):
        ax = axes[f_idx // 3, f_idx % 3]
        for tx_id in TRANSMITTER_IDS:
            if tx_id in train_features:
                values = train_features[tx_id][:, f_idx]
                ax.hist(values, bins=30, alpha=0.5, label=tx_id, color=TX_COLORS[tx_id])

        ax.set_title(FEATURE_NAMES[f_idx])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "03_feature_distributions")


def plot_feature_heatmap(train_features):
    """Plot heatmap of mean features per transmitter."""
    means = []
    for tx_id in TRANSMITTER_IDS:
        if tx_id in train_features:
            means.append(np.mean(train_features[tx_id], axis=0))

    means = np.array(means)

    fig, ax = plt.subplots(figsize=(12, 5))
    image = _draw_annotated_heatmap(
        ax,
        means,
        FEATURE_NAMES,
        [f"{tx} ({DEVICE_PROFILES[tx]['name']})" for tx in TRANSMITTER_IDS],
        cmap="YlOrRd",
    )
    ax.set_title("Mean RF Fingerprint Features per Transmitter", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "04_feature_heatmap")


def plot_pca_tsne(scaled_train_features):
    """Plot PCA and t-SNE visualizations of device clusters."""
    # Combine all features with labels
    all_features = []
    all_labels = []
    for tx_id in TRANSMITTER_IDS:
        if tx_id in scaled_train_features:
            # Use a subset for speed
            subset = scaled_train_features[tx_id][:200]
            all_features.append(subset)
            all_labels.extend([tx_id] * subset.shape[0])

    X = np.vstack(all_features)
    labels = np.array(all_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Device Cluster Visualization in Feature Space", fontsize=16, fontweight="bold")

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    for tx_id in TRANSMITTER_IDS:
        mask = labels == tx_id
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.6,
                    label=tx_id, color=TX_COLORS[tx_id])
    ax1.set_title(f"PCA (Var explained: {sum(pca.explained_variance_ratio_):.1%})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    for tx_id in TRANSMITTER_IDS:
        mask = labels == tx_id
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=15, alpha=0.6,
                    label=tx_id, color=TX_COLORS[tx_id])
    ax2.set_title("t-SNE Embedding")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "05_pca_tsne_clusters")


def plot_similarity_graph(G):
    """Plot the device similarity graph."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Device RF Fingerprint Similarity Graph", fontsize=14, fontweight="bold")

    pos = nx.spring_layout(G, seed=42)
    node_colors = [TX_COLORS[n] for n in G.nodes()]
    node_labels = {n: f"{n}\n{DEVICE_PROFILES[n]['name']}" for n in G.nodes()}

    # Draw edges with weights
    edges = G.edges(data=True)
    edge_weights = [d.get("weight", 0.5) * 3 for _, _, d in edges]

    nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_weights, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight="bold", ax=ax)

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    ax.set_facecolor("#f0f0f0")
    fig.tight_layout()
    save_fig(fig, "06_similarity_graph")


def plot_training_loss(loss_history):
    """Plot TSGAE training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, color="#2196F3", linewidth=1.5)
    ax.set_title("TSGAE Training Loss Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    save_fig(fig, "07_training_loss")


def plot_reconstruction_errors(error_stats, thresholds):
    """Plot RF verification score distributions with threshold lines."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Per-Device RF Verification Score Distribution with Thresholds",
                 fontsize=14, fontweight="bold")

    for i, tx_id in enumerate(TRANSMITTER_IDS):
        ax = axes[i // 3, i % 3]
        if tx_id in error_stats:
            errors = error_stats[tx_id]["all_errors"]
            threshold = thresholds[tx_id]
            mean = error_stats[tx_id]["mean"]

            ax.hist(errors, bins=40, color=TX_COLORS[tx_id], alpha=0.7, edgecolor="white")
            ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.4f}")
            ax.axvline(mean, color="black", linestyle="-", linewidth=1, label=f"Mean: {mean:.4f}")

            ax.set_title(f"{tx_id} ({DEVICE_PROFILES[tx_id]['name']})")
            ax.set_xlabel("RF Verification Score")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "08_reconstruction_errors")


def plot_trust_score_evolution(trust_evaluator):
    """Plot trust score evolution over monitoring cycles."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Trust Score Evolution During Monitoring", fontsize=14, fontweight="bold")

    for tx_id in TRANSMITTER_IDS:
        history = trust_evaluator.get_trust_history(tx_id)
        if history:
            cycles = [h["cycle"] for h in history]
            scores = [h["trust"] for h in history]
            ax.plot(cycles, scores, marker="o", markersize=4,
                    label=f"{tx_id} ({DEVICE_PROFILES[tx_id]['name']})",
                    color=TX_COLORS[tx_id], linewidth=2)

    ax.axhline(0.6, color="orange", linestyle="--", alpha=0.5, label="Suspicious threshold")
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Compromised threshold")
    ax.set_xlabel("Monitoring Cycle")
    ax.set_ylabel("Trust Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "09_trust_score_evolution")


def plot_attack_detection_timeline(trust_evaluator, attack_start_cycle):
    """Plot timeline showing normal operation -> attack -> detection."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Attack Detection Timeline", fontsize=16, fontweight="bold")

    for tx_id in TRANSMITTER_IDS:
        history = trust_evaluator.get_trust_history(tx_id)
        if not history:
            continue

        cycles = [h["cycle"] for h in history]
        scores = [h["trust"] for h in history]
        deviations = [h["deviation"] for h in history]

        ax1.plot(cycles, scores, marker="o", markersize=4,
                 label=tx_id, color=TX_COLORS[tx_id], linewidth=2)
        ax2.plot(cycles, deviations, marker="s", markersize=4,
                 label=tx_id, color=TX_COLORS[tx_id], linewidth=2)

    # Mark attack start
    ax1.axvline(attack_start_cycle, color="red", linestyle="-.", linewidth=2, alpha=0.7)
    ax1.text(attack_start_cycle + 0.1, 0.95, "Attack Start", color="red", fontsize=10)
    ax2.axvline(attack_start_cycle, color="red", linestyle="-.", linewidth=2, alpha=0.7)

    ax1.axhline(0.6, color="orange", linestyle="--", alpha=0.4)
    ax1.axhline(0.3, color="red", linestyle="--", alpha=0.4)
    ax1.set_ylabel("Trust Score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Trust Scores Over Time")

    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.4, label="Anomaly threshold (ratio=1)")
    ax2.set_xlabel("Monitoring Cycle")
    ax2.set_ylabel("Threshold Ratio")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Threshold Ratio Over Time (>1 = RF mismatch)")

    fig.tight_layout()
    save_fig(fig, "10_attack_detection_timeline")


def plot_confusion_summary(trust_evaluator, attack_start_cycle):
    """Plot summary of detection accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Detection Performance Summary", fontsize=16, fontweight="bold")

    # Bar chart of final trust scores
    tx_ids = []
    final_scores = []
    colors = []
    for tx_id in TRANSMITTER_IDS:
        history = trust_evaluator.get_trust_history(tx_id)
        if history:
            tx_ids.append(tx_id)
            final_scores.append(history[-1]["trust"])
            colors.append(TX_COLORS[tx_id])

    bars = ax1.bar(tx_ids, final_scores, color=colors, edgecolor="white", linewidth=1.5)
    ax1.axhline(0.6, color="orange", linestyle="--", alpha=0.5, label="Suspicious")
    ax1.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Compromised")
    ax1.set_ylabel("Final Trust Score")
    ax1.set_title("Final Trust Scores After Monitoring")
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Pie chart of auth results
    total_pass = sum(trust_evaluator.total_passes.get(tx, 0) for tx in TRANSMITTER_IDS)
    total_fail = sum(
        trust_evaluator.total_checks.get(tx, 0) - trust_evaluator.total_passes.get(tx, 0)
        for tx in TRANSMITTER_IDS
    )

    if total_pass + total_fail > 0:
        ax2.pie(
            [total_pass, total_fail],
            labels=["PASS (Legitimate)", "FAIL (Anomaly)"],
            colors=["#4CAF50", "#F44336"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
    ax2.set_title("Overall Authentication Results")

    fig.tight_layout()
    save_fig(fig, "11_detection_summary")


def plot_claim_confusion_counts(paper_metrics):
    """Plot claim-level confusion counts from the held-out authentication evaluation."""
    summary = (paper_metrics or {}).get("summary") or {}
    labels = ["TP", "TN", "FP", "FN"]
    values = [
        summary.get("tp", 0),
        summary.get("tn", 0),
        summary.get("fp", 0),
        summary.get("fn", 0),
    ]
    colors = ["#4CAF50", "#62E2FF", "#FFD36B", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Claim-Level Authentication Confusion Counts", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(values) * 0.015 if max(values) else 0.05,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.text(
        0.5,
        -0.16,
        (
            f"Legitimate claims={paper_metrics.get('total_legitimate_claims', 0)} | "
            f"Attack claims={paper_metrics.get('total_attack_claims', 0)} | "
            f"Attack pairs/round={paper_metrics.get('attack_pairs_per_round', 0)}"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#455A64",
    )
    fig.tight_layout()
    save_fig(fig, "12_claim_confusion_counts")


def plot_attack_pair_heatmap(paper_metrics):
    """Plot a heatmap of detection rate for every attacker -> target pair."""
    attack_pair_metrics = (paper_metrics or {}).get("per_attack_pair") or {}
    tx_count = len(TRANSMITTER_IDS)
    matrix = np.full((tx_count, tx_count), np.nan, dtype=np.float32)

    for attacker_idx, attacker_tx in enumerate(TRANSMITTER_IDS):
        for target_idx, target_tx in enumerate(TRANSMITTER_IDS):
            if attacker_tx == target_tx:
                continue
            pair_key = f"{attacker_tx}->{target_tx}"
            matrix[attacker_idx, target_idx] = attack_pair_metrics.get(pair_key, {}).get("detection_rate", 0.0)

    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.cm.get_cmap("YlOrRd").copy()
    cmap.set_bad("#E6EEF8")

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    image = ax.imshow(masked_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(tx_count))
    ax.set_xticklabels(TRANSMITTER_IDS)
    ax.set_yticks(np.arange(tx_count))
    ax.set_yticklabels(TRANSMITTER_IDS)
    ax.set_xlabel("Claimed target identity")
    ax.set_ylabel("Attacker RF source")
    ax.set_title("Cross-Device Attack Pair Detection Rate", fontsize=14, fontweight="bold")

    for attacker_idx, attacker_tx in enumerate(TRANSMITTER_IDS):
        for target_idx, target_tx in enumerate(TRANSMITTER_IDS):
            if attacker_tx == target_tx:
                ax.text(target_idx, attacker_idx, "—", ha="center", va="center", fontsize=12, color="#607D8B")
                continue

            value = matrix[attacker_idx, target_idx]
            ax.text(
                target_idx,
                attacker_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Detection rate")
    fig.tight_layout()
    save_fig(fig, "13_attack_pair_heatmap")


def plot_system_architecture():
    """Plot system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("RF Fingerprint Zero Trust System Architecture", fontsize=16, fontweight="bold")

    # Draw boxes for components
    components = [
        (1, 6.5, 2.5, 0.8, "RF Signal\nCapture (I/Q)", "#E3F2FD"),
        (4, 6.5, 2.5, 0.8, "Feature\nExtraction (9)", "#E8F5E9"),
        (7, 6.5, 2.5, 0.8, "StandardScaler\nNormalization", "#FFF3E0"),
        (1, 4.5, 2.5, 0.8, "Similarity\nGraph Builder", "#F3E5F5"),
        (4, 4.5, 2.5, 0.8, "TSGAE\nAutoencoder", "#FFEBEE"),
        (7, 4.5, 2.5, 0.8, "Anomaly\nDetection", "#E0F7FA"),
        (2.5, 2.5, 2.5, 0.8, "Bayesian\nZero Trust", "#FFF9C4"),
        (5.5, 2.5, 2.5, 0.8, "Real-time\nDashboard", "#F1F8E9"),
        (4, 0.8, 3, 0.8, "Enterprise WiFi\nSecurity Decision", "#FFCDD2"),
    ]

    for x, y, w, h, text, color in components:
        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                              edgecolor="black", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=3)

    # Draw arrows
    arrows = [
        (2.25, 6.5, 4.25, 6.9),   # RF -> Feature
        (5.25, 6.5, 7.25, 6.9),   # Feature -> Scaler
        (2.25, 5.3, 2.25, 4.55),  # RF -> Graph (down)
        (5.25, 6.5, 5.25, 5.35),  # Feature -> TSGAE
        (3.5, 4.9, 4, 4.9),       # Graph -> TSGAE
        (6.5, 4.9, 7, 4.9),       # TSGAE -> Anomaly
        (8.25, 4.5, 6.75, 3.35),  # Anomaly -> Dashboard
        (8.25, 4.5, 5.0, 3.35),   # Anomaly -> ZeroTrust
        (3.75, 2.5, 4.5, 1.65),   # ZeroTrust -> Decision
        (6.75, 2.5, 6.0, 1.65),   # Dashboard -> Decision
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    fig.tight_layout()
    save_fig(fig, "00_system_architecture")


def generate_all_visualizations(dataset, train_features, scaled_train_features,
                                 G, loss_history, error_stats, thresholds,
                                 trust_evaluator, attack_start_cycle, paper_metrics=None):
    """Generate all visualization graphs for PPT presentation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n[Visualization] Generating PPT graphs...")

    plot_steps = [
        ("system architecture", plot_system_architecture, ()),
        ("raw I/Q signals", plot_raw_iq_signals, (dataset,)),
        ("I/Q constellation", plot_iq_constellation, (dataset,)),
        ("feature distributions", plot_feature_distributions, (train_features,)),
        ("feature heatmap", plot_feature_heatmap, (train_features,)),
        ("PCA/t-SNE clusters", plot_pca_tsne, (scaled_train_features,)),
        ("similarity graph", plot_similarity_graph, (G,)),
        ("training loss", plot_training_loss, (loss_history,)),
        ("RF score distributions", plot_reconstruction_errors, (error_stats, thresholds)),
        ("trust score evolution", plot_trust_score_evolution, (trust_evaluator,)),
        ("attack detection timeline", plot_attack_detection_timeline, (trust_evaluator, attack_start_cycle)),
        ("detection summary", plot_confusion_summary, (trust_evaluator, attack_start_cycle)),
    ]

    if paper_metrics:
        plot_steps.extend(
            [
                ("claim confusion counts", plot_claim_confusion_counts, (paper_metrics,)),
                ("attack pair heatmap", plot_attack_pair_heatmap, (paper_metrics,)),
            ]
        )

    for label, fn, args in plot_steps:
        try:
            fn(*args)
        except Exception as exc:
            print(f"  Skipped {label}: {exc}")

    print(f"\n[Visualization] All {len(plot_steps)} graphs saved to {RESULTS_DIR}/")

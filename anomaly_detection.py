"""
Anomaly Detection module.
Uses combined TSGAE reconstruction error + per-device Mahalanobis distance
for robust RF fingerprint anomaly detection.

Each monitoring cycle averages multiple RF samples per device for stability,
matching how real enterprise systems analyze multiple signal captures per check.
"""

import numpy as np
import pickle
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from config import (
    NUM_FEATURES,
    PROFILE_DISTANCE_CONFIDENCE,
    THRESHOLD_SIGMA,
    THRESHOLD_SAVE_PATH,
    TRANSMITTER_IDS,
)
from model import compute_reconstruction_error

SAMPLES_PER_CHECK = 5   # average 5 samples per device per monitoring cycle
PROFILE_DISTANCE_THRESHOLD = float(np.sqrt(chi2.ppf(PROFILE_DISTANCE_CONFIDENCE, df=NUM_FEATURES)))


class DeviceProfiler:
    """Maintains per-device statistical profiles for Mahalanobis distance."""

    def __init__(self):
        self.centroids = {}     # mean feature vector per device
        self.inv_covs = {}      # inverse covariance per device

    def fit(self, scaled_train_features):
        """Learn per-device profiles from training data."""
        for tx_id in TRANSMITTER_IDS:
            features = scaled_train_features[tx_id]
            if features.ndim == 3:
                features = np.mean(features, axis=1)
            self.centroids[tx_id] = np.mean(features, axis=0)

            # Covariance with regularization for numerical stability
            cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-3
            self.inv_covs[tx_id] = np.linalg.inv(cov)

    def compute_distance(self, feature_vector, claimed_tx_id):
        """Compute Mahalanobis distance of a sample to a device's profile."""
        if feature_vector.ndim == 2:
            feature_vector = np.mean(feature_vector, axis=0)
        return mahalanobis(
            feature_vector,
            self.centroids[claimed_tx_id],
            self.inv_covs[claimed_tx_id],
        )


def compute_dynamic_thresholds(model, scaled_train_features, adj_tensor, n_eval_rounds=500):
    """
    Compute per-device anomaly detection thresholds from legitimate training data.

    Each round averages SAMPLES_PER_CHECK samples per device (matching monitoring).
    Threshold = mean + THRESHOLD_SIGMA * std for the training reconstruction errors.
    """
    profiler = DeviceProfiler()
    profiler.fit(scaled_train_features)

    train_arrays = [scaled_train_features[tx_id] for tx_id in TRANSMITTER_IDS]

    # Collect reconstruction errors over many legitimate rounds.
    all_reconstruction = {i: [] for i in range(len(TRANSMITTER_IDS))}

    for _ in range(n_eval_rounds):
        # Average multiple samples per device (same as monitoring does)
        snapshot = []
        for arr in train_arrays:
            indices = np.random.randint(0, arr.shape[0], size=SAMPLES_PER_CHECK)
            avg_features = np.mean(arr[indices], axis=0)
            snapshot.append(avg_features)

        x = np.array(snapshot, dtype=np.float32)

        recon_errors = compute_reconstruction_error(model, x, adj_tensor)

        for i, _tx_id in enumerate(TRANSMITTER_IDS):
            all_reconstruction[i].append(float(recon_errors[i]))

    thresholds = {}
    error_stats = {}
    for i, tx_id in enumerate(TRANSMITTER_IDS):
        recon_arr = np.array(all_reconstruction[i], dtype=np.float32)
        mean_c = float(np.mean(recon_arr))
        std_c = float(np.std(recon_arr))
        threshold = mean_c + THRESHOLD_SIGMA * std_c

        thresholds[tx_id] = float(threshold)
        error_stats[tx_id] = {
            "mean": mean_c,
            "std": std_c,
            "threshold": float(threshold),
            "min": float(np.min(recon_arr)),
            "max": float(np.max(recon_arr)),
            "p95": float(np.percentile(recon_arr, 95)),
            "p99": float(np.percentile(recon_arr, 99)),
            "all_errors": recon_arr.tolist(),
        }

        print(f"  {tx_id}: mean={mean_c:.4f}, std={std_c:.4f}, threshold={threshold:.4f}")

    # Save everything
    with open(THRESHOLD_SAVE_PATH, "wb") as f:
        pickle.dump({
            "thresholds": thresholds,
            "stats": error_stats,
            "profiler_centroids": profiler.centroids,
            "profiler_inv_covs": profiler.inv_covs,
        }, f)

    print(f"[Anomaly] Thresholds saved to {THRESHOLD_SAVE_PATH}")
    return thresholds, error_stats, profiler


def load_thresholds():
    """Load previously computed thresholds and profiler."""
    with open(THRESHOLD_SAVE_PATH, "rb") as f:
        data = pickle.load(f)
    profiler = DeviceProfiler()
    profiler.centroids = data["profiler_centroids"]
    profiler.inv_covs = data["profiler_inv_covs"]
    return data["thresholds"], data["stats"], profiler


def _safe_ratio(score, threshold):
    """Return a stable ratio for UI and report displays."""
    if score is None or threshold is None or threshold < 1e-10:
        return 0.0
    return float(score / threshold)


def detect_anomaly(recon_score, maha_dist, claimed_tx_id, thresholds):
    """Check reconstruction and profile-distance evidence for impersonation."""
    recon_threshold = thresholds[claimed_tx_id]
    recon_ratio = _safe_ratio(recon_score, recon_threshold)
    profile_ratio = _safe_ratio(maha_dist, PROFILE_DISTANCE_THRESHOLD)
    decision_ratio = max(recon_ratio, profile_ratio)
    decision_source = "profile_distance" if profile_ratio >= recon_ratio else "reconstruction"

    return {
        "is_anomaly": decision_ratio > 1.0,
        "decision_ratio": float(decision_ratio),
        "decision_source": decision_source,
        "recon_threshold": float(recon_threshold),
        "recon_ratio": float(recon_ratio),
        "profile_threshold": float(PROFILE_DISTANCE_THRESHOLD),
        "profile_ratio": float(profile_ratio),
    }


def evaluate_snapshot(model, snapshot_features, adj_tensor, thresholds, online_mask, profiler):
    """
    Evaluate a full monitoring snapshot using TSGAE reconstruction error.

    Args:
        model: trained TSGAE
        snapshot_features: (N_devices, N_features) averaged feature vectors
        adj_tensor: adjacency matrix tensor
        thresholds: per-device thresholds
        online_mask: dict {tx_id: bool}
        profiler: DeviceProfiler with learned device profiles

    Returns:
        results: dict per device
    """
    recon_errors = compute_reconstruction_error(model, snapshot_features, adj_tensor)

    results = {}
    for i, tx_id in enumerate(TRANSMITTER_IDS):
        if not online_mask.get(tx_id, True):
            results[tx_id] = {
                "error": None,
                "is_anomaly": False,
                "deviation_ratio": 0.0,
                "status": "offline",
                "recon_error": None,
                "maha_dist": None,
                "recon_threshold": float(thresholds.get(tx_id, 0.0)),
                "recon_ratio": 0.0,
                "profile_threshold": float(PROFILE_DISTANCE_THRESHOLD),
                "profile_ratio": 0.0,
                "decision_source": None,
            }
            continue

        recon_score = float(recon_errors[i])
        maha_dist = float(profiler.compute_distance(snapshot_features[i], tx_id))
        decision = detect_anomaly(recon_score, maha_dist, tx_id, thresholds)

        results[tx_id] = {
            "error": recon_score,
            "is_anomaly": decision["is_anomaly"],
            "deviation_ratio": decision["decision_ratio"],
            "status": "anomaly" if decision["is_anomaly"] else "normal",
            "recon_error": recon_score,
            "maha_dist": maha_dist,
            "recon_threshold": decision["recon_threshold"],
            "recon_ratio": decision["recon_ratio"],
            "profile_threshold": decision["profile_threshold"],
            "profile_ratio": decision["profile_ratio"],
            "decision_source": decision["decision_source"],
        }

    return results

"""
Preprocessing module for RF fingerprint data.
Handles StandardScaler fitting on training data and transformation.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from config import SCALER_SAVE_PATH, TRANSMITTER_IDS


def fit_scaler(train_features):
    """
    Fit StandardScaler on ALL training features combined.
    This same scaler MUST be used during testing/monitoring.
    """
    matrices = []
    for tx_id in TRANSMITTER_IDS:
        if tx_id not in train_features:
            continue
        features = train_features[tx_id]
        if features.ndim == 3:
            matrices.append(features.reshape(-1, features.shape[-1]))
        else:
            matrices.append(features)

    all_features = np.vstack(matrices)

    scaler = StandardScaler()
    scaler.fit(all_features)

    # Save scaler for reuse
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[Preprocessing] Scaler fitted on {all_features.shape[0]} samples, {all_features.shape[1]} features")
    return scaler


def load_scaler():
    """Load previously fitted scaler."""
    with open(SCALER_SAVE_PATH, "rb") as f:
        return pickle.load(f)


def scale_features(features_dict, scaler):
    """
    Apply the fitted scaler to feature dictionaries.
    Input:  dict {tx_id: (n_samples, 9)}
    Output: dict {tx_id: (n_samples, 9)} scaled
    """
    scaled = {}
    for tx_id, features in features_dict.items():
        if features.ndim == 3:
            shape = features.shape
            flat = features.reshape(-1, shape[-1])
            scaled[tx_id] = scaler.transform(flat).reshape(shape).astype(np.float32)
        else:
            scaled[tx_id] = scaler.transform(features).astype(np.float32)
    return scaled


def scale_single(features, scaler):
    """
    Scale a single feature vector or batch.
    Input:  (9,) or (n, 9)
    Output: same shape, scaled
    """
    if features.ndim == 1:
        return scaler.transform(features.reshape(1, -1)).flatten().astype(np.float32)
    return scaler.transform(features).astype(np.float32)

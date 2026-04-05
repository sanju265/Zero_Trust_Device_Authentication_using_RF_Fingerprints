"""
Data Loader module for ManySig RF dataset.
Loads pickle data and organizes it by transmitter.

ManySig structure:
    data[tx_idx][rx_idx][date_idx][eq_idx] = ndarray(1000, 256, 2)
    tx_list = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']

We pick one receiver (rx=0), one date (date=0), non-equalized (eq=0)
to get 1000 clean samples per transmitter with shape (1000, 256, 2).
"""

import pickle
import numpy as np
import os
from datetime import datetime
from config import DATASET_PATH, TRANSMITTER_IDS, TRANSMITTER_LABELS


def describe_dataset_source(path=None):
    """Return filesystem metadata for the ManySig dataset file."""
    dataset_path = path or DATASET_PATH

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            "Please place ManySig.pkl in the project root directory."
        )

    stat = os.stat(dataset_path)
    return {
        "path": os.path.abspath(dataset_path),
        "filename": os.path.basename(dataset_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_timestamp": int(stat.st_mtime),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_manysig_dataset(path=None):
    """
    Load ManySig.pkl dataset.
    Returns dict: {tx_id: np.array of shape (1000, 256, 2)}
    """
    if path is None:
        path = DATASET_PATH

    dataset_info = describe_dataset_source(path)

    print(
        "  Loading verified dataset source:\n"
        f"    File     : {dataset_info['path']}\n"
        f"    Size     : {dataset_info['size_mb']} MB\n"
        f"    Modified : {dataset_info['modified_at']}"
    )
    print("  Reading ManySig.pkl into memory (this may take a moment for the large file)...")
    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    tx_list = raw_data["tx_list"]   # ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
    data = raw_data["data"]         # nested list: data[tx][rx][date][eq] = (1000,256,2)

    # Map TX labels to TX IDs
    label_to_idx = {}
    for i, label in enumerate(tx_list):
        label_to_idx[label] = i

    dataset = {}
    # Use rx=0, date=0, equalized=0 for clean non-equalized data
    RX_IDX = 0
    DATE_IDX = 0
    EQ_IDX = 0

    for tx_id in TRANSMITTER_IDS:
        label = TRANSMITTER_LABELS[tx_id]
        if label in label_to_idx:
            tx_idx = label_to_idx[label]
            samples = np.array(data[tx_idx][RX_IDX][DATE_IDX][EQ_IDX], dtype=np.float32)
            dataset[tx_id] = samples
        else:
            raise ValueError(f"Transmitter label '{label}' for {tx_id} not found in dataset")

    print(f"[DataLoader] Loaded {len(dataset)} transmitters:")
    for tx_id, d in dataset.items():
        print(f"  {tx_id} ({TRANSMITTER_LABELS[tx_id]}): {d.shape}")

    return dataset


def split_train_test(dataset, train_ratio=0.8):
    """Split dataset into training and testing sets per transmitter."""
    train_data = {}
    test_data = {}

    for tx_id, samples in dataset.items():
        n_total = samples.shape[0]
        n_train = int(n_total * train_ratio)

        # Shuffle before splitting (fixed seed for reproducibility)
        indices = np.random.RandomState(42).permutation(n_total)
        train_data[tx_id] = samples[indices[:n_train]]
        test_data[tx_id] = samples[indices[n_train:]]

    print(f"\n[DataLoader] Train/Test split ({train_ratio:.0%}/{1-train_ratio:.0%}):")
    for tx_id in dataset:
        print(f"  {tx_id}: train={train_data[tx_id].shape[0]}, test={test_data[tx_id].shape[0]}")

    return train_data, test_data

"""
Temporal Spectral Graph Autoencoder (TSGAE) model.
Learns normal RF fingerprint patterns for each device using graph-aware encoding.

DATA STRUCTURE:
- 6 devices (TX0-TX5)
- 1000 packets per device
- 256 I/Q samples per packet
- 9 engineered features per time step

Training:
- 80% of 6000 total packets (4800 packets) across 6 devices
- Each packet is represented as a (256, 9) temporal RF feature map
- Offline training is done once and the saved model is reused during live testing
"""

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    MODEL_ARTIFACT_VERSION,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    NUM_FEATURES,
    TRAIN_RATIO,
    TRANSMITTER_IDS,
)


class GraphConvLayer(nn.Module):
    """Graph convolution layer supporting both batched and unbatched inputs."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        return torch.matmul(adj, support) + self.bias


class TSGAE(nn.Module):
    """
    Temporal Spectral Graph Autoencoder.

    Input:
    - x: (batch_size, 256, 9)
    - adj: (256, 256) temporal-spectral adjacency matrix
    """

    def __init__(self, n_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()

        temporal_channels = hidden_dim

        self.temporal_conv1 = nn.Conv1d(n_features, temporal_channels, kernel_size=5, padding=2)
        self.temporal_conv2 = nn.Conv1d(temporal_channels, hidden_dim, kernel_size=5, padding=2)

        self.enc_gcn1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.enc_gcn2 = GraphConvLayer(hidden_dim, latent_dim)

        self.dec_gcn1 = GraphConvLayer(latent_dim, hidden_dim)
        self.dec_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, n_features)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def encode(self, x, adj):
        x = x.transpose(1, 2)
        x = self.relu(self.temporal_conv1(x))
        x = self.relu(self.temporal_conv2(x))
        x = x.transpose(1, 2)

        h = self.relu(self.enc_gcn1(x, adj))
        h = self.dropout(h)
        z = self.enc_gcn2(h, adj)
        return z

    def decode(self, z, adj):
        h = self.relu(self.dec_gcn1(z, adj))
        h = self.relu(self.dec_fc1(h))
        x_hat = self.dec_fc2(h)
        return x_hat

    def forward(self, x, adj):
        z = self.encode(x, adj)
        x_hat = self.decode(z, adj)
        return x_hat, z


def create_training_batches(scaled_train_features, batch_size=32):
    """
    Create packet-level mini-batches for offline training.

    scaled_train_features[tx_id] has shape (n_packets, 256, 9).
    Each packet is one complete temporal RF fingerprint sample.
    """

    all_samples = []
    all_device_labels = []

    for idx, tx_id in enumerate(TRANSMITTER_IDS):
        device_packets = scaled_train_features[tx_id]
        n_packets = device_packets.shape[0]

        print(f"  {tx_id}: {n_packets} packets")

        for packet_idx in range(n_packets):
            all_samples.append(device_packets[packet_idx])
            all_device_labels.append(idx)

    all_samples = np.array(all_samples, dtype=np.float32)
    all_device_labels = np.array(all_device_labels, dtype=np.int64)

    total_packets = len(all_samples)
    print(f"\n  Total training packets: {total_packets}")
    print(f"  Packets per device: {total_packets // len(TRANSMITTER_IDS)}")

    perm = np.random.permutation(total_packets)
    all_samples = all_samples[perm]
    all_device_labels = all_device_labels[perm]

    dataset = TensorDataset(
        torch.FloatTensor(all_samples),
        torch.LongTensor(all_device_labels),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_tsgae(scaled_train_features, adj_tensor, epochs=NUM_EPOCHS, artifact_meta=None):
    """
    Train the TSGAE model on legitimate packet-level RF fingerprints.

    The held-out 20% test split is not used here. It is reserved for
    offline metrics and repeated live dashboard testing.
    """

    model = TSGAE()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    batch_size = 32
    train_loader = create_training_batches(scaled_train_features, batch_size=batch_size)

    total_train_packets = sum(scaled_train_features[tx_id].shape[0] for tx_id in TRANSMITTER_IDS)
    packets_per_device_train = total_train_packets // len(TRANSMITTER_IDS)
    packets_per_device_test = int(round(packets_per_device_train * (1 - TRAIN_RATIO) / TRAIN_RATIO))
    steps_per_epoch = len(train_loader)

    print(f"\n{'=' * 60}")
    print("[TSGAE] TRAINING STARTING")
    print(f"{'=' * 60}")
    print(f"  Training packets:    {total_train_packets} ({packets_per_device_train} per device)")
    print(f"  Held-out packets:    {packets_per_device_test * len(TRANSMITTER_IDS)} ({packets_per_device_test} per device)")
    print(f"  Packets shape:       ({total_train_packets}, 256, {NUM_FEATURES})")
    print(f"  Features per sample: {NUM_FEATURES}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Total epochs:        {epochs}")
    print(f"  Total training steps: {steps_per_epoch * epochs:,}")
    print(f"{'=' * 60}\n")

    loss_history = []
    print("  Training progress:")
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Time':<10}")
    print(f"  {'-' * 34}")

    start_time = datetime.now()
    model.train()

    for epoch in range(epochs):
        epoch_losses = []

        for batch_x, _batch_y in train_loader:
            batch_x = batch_x.to(adj_tensor.device)

            x_hat, _z = model(batch_x, adj_tensor)
            loss = criterion(x_hat, batch_x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_train_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"  {epoch + 1:3d}/{epochs}   {avg_train_loss:.6f}    {elapsed:.1f} min")

    duration_seconds = round((datetime.now() - start_time).total_seconds(), 2)
    training_info = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "total_training_packets": total_train_packets,
        "packets_per_device_train": packets_per_device_train,
        "packets_per_device_test": packets_per_device_test,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * epochs,
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60, 2),
        "final_train_loss": float(loss_history[-1]),
        "train_ratio": TRAIN_RATIO,
        "uses_held_out_test_for_runtime": True,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "loss_history": loss_history,
            "val_loss_history": [],
            "training_info": training_info,
            "artifact_meta": artifact_meta or {},
        },
        MODEL_SAVE_PATH,
    )

    print(f"\n{'=' * 60}")
    print("[TSGAE] TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total time:        {training_info['duration_minutes']:.1f} minutes")
    print(f"  Final train loss:  {loss_history[-1]:.6f}")
    print(f"  Model saved to:    {MODEL_SAVE_PATH}")
    print(f"{'=' * 60}\n")

    model.eval()
    return model, loss_history, training_info


def load_tsgae():
    """Load a previously trained TSGAE model."""

    model = TSGAE()
    payload = torch.load(MODEL_SAVE_PATH, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
        metadata = {
            "loss_history": payload.get("loss_history", []),
            "val_loss_history": payload.get("val_loss_history", []),
            "training_info": payload.get("training_info", {}),
            "artifact_meta": payload.get("artifact_meta", {}),
        }
    else:
        model.load_state_dict(payload)
        metadata = {}

    model.eval()
    return model, metadata


def compute_reconstruction_error(model, packet_features, adj_tensor):
    """
    Compute reconstruction error for packet-level temporal RF features.

    Args:
        packet_features: (256, 9) or (batch, 256, 9)
        adj_tensor: temporal-spectral adjacency matrix

    Returns:
        numpy array of per-packet reconstruction errors
    """

    model.eval()
    with torch.no_grad():
        if not isinstance(packet_features, torch.Tensor):
            packet_features = torch.FloatTensor(packet_features)

        if packet_features.dim() == 2:
            packet_features = packet_features.unsqueeze(0)

        packet_features = packet_features.to(adj_tensor.device)
        x_hat, _latent = model(packet_features, adj_tensor)
        error = torch.mean((x_hat - packet_features) ** 2, dim=(1, 2))
        return error.cpu().numpy()


if __name__ == "__main__":
    from train_model import main as offline_training_main

    offline_training_main()

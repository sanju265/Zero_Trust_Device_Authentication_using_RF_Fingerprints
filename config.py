"""
Configuration module for WiFi RF Fingerprint Authentication System.
Centralized settings for all system components.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ManySig.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "results", "reports")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "tsgae_model.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "scaler.pkl")
THRESHOLD_SAVE_PATH = os.path.join(BASE_DIR, "threshold.pkl")
PAPER_METRICS_PATH = os.path.join(REPORTS_DIR, "paper_metrics.txt")
RUNTIME_CACHE_PATH = os.path.join(BASE_DIR, "runtime_cache.pkl")

# ─── Admin Credentials ──────────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "rf@secure2026"

# ─── Dataset ─────────────────────────────────────────────────────────
TRANSMITTER_IDS = ["TX0", "TX1", "TX2", "TX3", "TX4", "TX5"]
TRANSMITTER_LABELS = {
    "TX0": "14-10",
    "TX1": "14-7",
    "TX2": "20-15",
    "TX3": "20-19",
    "TX4": "6-15",
    "TX5": "8-20",
}
NUM_TRANSMITTERS = len(TRANSMITTER_IDS)
SAMPLE_SHAPE = (256, 2)  # (time_steps, I/Q channels)
TRAIN_RATIO = 0.8  # 80% train, 20% test

# ─── Enterprise Device Mapping ───────────────────────────────────────
DEVICE_PROFILES = {
    "TX0": {"name": "Admin-Laptop",     "mac": "AA:BB:CC:DD:EE:01", "type": "Laptop"},
    "TX1": {"name": "CEO-Phone",        "mac": "AA:BB:CC:DD:EE:02", "type": "Smartphone"},
    "TX2": {"name": "Server-Room-AP",   "mac": "AA:BB:CC:DD:EE:03", "type": "Access Point"},
    "TX3": {"name": "Engineer-Tablet",  "mac": "AA:BB:CC:DD:EE:04", "type": "Tablet"},
    "TX4": {"name": "IoT-Sensor-01",    "mac": "AA:BB:CC:DD:EE:05", "type": "IoT Device"},
    "TX5": {"name": "Finance-Desktop",  "mac": "AA:BB:CC:DD:EE:06", "type": "Desktop"},
}

# ─── Feature Extraction ─────────────────────────────────────────────
NUM_FEATURES = 9  # Number of engineered RF fingerprint features

# ─── TSGAE Model ─────────────────────────────────────────────────────
LATENT_DIM = 64
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
SNAPSHOTS_PER_EPOCH = 800
MODEL_ARTIFACT_VERSION = 6

# ─── Anomaly Detection ──────────────────────────────────────────────
THRESHOLD_SIGMA = 3.0  # mean + 3*std for dynamic threshold
PROFILE_DISTANCE_CONFIDENCE = 0.985  # chi-square confidence for profile mismatch checks

# ─── Zero Trust ──────────────────────────────────────────────────────
INITIAL_TRUST_SCORE = 0.85
TRUST_DECAY_RATE = 0.15       # How much trust drops per anomaly
TRUST_RECOVERY_RATE = 0.05    # How much trust recovers per normal cycle
TRUST_THRESHOLD_SUSPICIOUS = 0.6
TRUST_THRESHOLD_COMPROMISED = 0.3

# ─── Monitoring ──────────────────────────────────────────────────────
MONITOR_INTERVAL = 20         # seconds between authentication checks
OFFLINE_TRANSMITTERS = ["TX4"]  # TX4 is offline by default

# ─── Attack Simulation (file-based IPC) ─────────────────────────────
# Edit ATTACKER_TX and TARGET_TX to change the attack scenario
ATTACKER_TX = "TX3"           # Transmitter performing the attack
TARGET_TX = "TX1"             # Device identity being impersonated
ATTACK_SIGNAL_FILE = os.path.join(BASE_DIR, ".attack_signal")  # trigger file

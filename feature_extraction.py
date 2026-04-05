"""
Feature Extraction module for RF fingerprinting.
Extracts 9 engineered features from raw I/Q signal data.
"""

import numpy as np


def extract_features_single(sample):
    """
    Extract 9 RF fingerprint features from a single (256, 2) I/Q sample.

    Features:
        0. Amplitude Mean        - mean of signal envelope
        1. Amplitude Std         - variation in signal strength
        2. Amplitude Max         - peak signal strength
        3. Phase Mean            - average signal phase
        4. Phase Std             - phase stability
        5. Signal Variance       - total I/Q power variance
        6. Spectral Energy       - frequency domain energy
        7. IQ Correlation        - correlation between I and Q
        8. Temporal Gradient     - rate of amplitude change
    """
    I = sample[:, 0]  # In-phase component
    Q = sample[:, 1]  # Quadrature component

    # Amplitude (envelope)
    amplitude = np.sqrt(I**2 + Q**2)

    # Phase
    phase = np.arctan2(Q, I)

    # Complex signal for FFT
    complex_signal = I + 1j * Q

    # ── Feature 0: Amplitude Mean ──
    amp_mean = np.mean(amplitude)

    # ── Feature 1: Amplitude Std ──
    amp_std = np.std(amplitude)

    # ── Feature 2: Amplitude Max ──
    amp_max = np.max(amplitude)

    # ── Feature 3: Phase Mean ──
    phase_mean = np.mean(phase)

    # ── Feature 4: Phase Std ──
    phase_std = np.std(phase)

    # ── Feature 5: Signal Variance ──
    signal_var = np.var(I) + np.var(Q)

    # ── Feature 6: Spectral Energy ──
    fft_vals = np.fft.fft(complex_signal)
    spectral_energy = np.sum(np.abs(fft_vals) ** 2) / len(fft_vals)

    # ── Feature 7: IQ Correlation ──
    if np.std(I) > 1e-10 and np.std(Q) > 1e-10:
        iq_corr = np.corrcoef(I, Q)[0, 1]
    else:
        iq_corr = 0.0

    # ── Feature 8: Temporal Gradient ──
    temporal_grad = np.mean(np.abs(np.diff(amplitude)))

    features = np.array([
        amp_mean, amp_std, amp_max,
        phase_mean, phase_std,
        signal_var, spectral_energy,
        iq_corr, temporal_grad
    ], dtype=np.float32)

    # Replace any NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_sequence_features_single(sample):
    """
    Extract 9 temporal RF features for each of the 256 time steps.

    Output shape: (256, 9)
    """
    I = sample[:, 0]
    Q = sample[:, 1]

    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.arctan2(Q, I))
    dI = np.gradient(I)
    dQ = np.gradient(Q)
    dAmplitude = np.gradient(amplitude)
    power = I**2 + Q**2
    phase_grad = np.gradient(phase)
    iq_product = I * Q

    features = np.stack(
        [
            I,
            Q,
            amplitude,
            phase,
            dI,
            dQ,
            dAmplitude,
            power,
            phase_grad + iq_product,
        ],
        axis=1,
    ).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_batch(samples):
    """
    Extract features from a batch of samples.
    Input:  (n_samples, 256, 2)
    Output: (n_samples, 9)
    """
    return np.array([extract_features_single(s) for s in samples], dtype=np.float32)


def extract_sequence_features_batch(samples):
    """
    Extract temporal RF feature maps from a batch of samples.
    Input:  (n_samples, 256, 2)
    Output: (n_samples, 256, 9)
    """
    return np.array([extract_sequence_features_single(s) for s in samples], dtype=np.float32)


def extract_all_features(dataset):
    """
    Extract features for all transmitters.
    Input:  dict {tx_id: (n_samples, 256, 2)}
    Output: dict {tx_id: (n_samples, 9)}
    """
    features = {}
    for tx_id, samples in dataset.items():
        features[tx_id] = extract_features_batch(samples)
        print(f"  {tx_id}: extracted {features[tx_id].shape[0]} feature vectors ({features[tx_id].shape[1]} features)")
    return features


def extract_all_sequence_features(dataset):
    """
    Extract temporal RF feature maps for all transmitters.
    Input:  dict {tx_id: (n_samples, 256, 2)}
    Output: dict {tx_id: (n_samples, 256, 9)}
    """
    features = {}
    for tx_id, samples in dataset.items():
        features[tx_id] = extract_sequence_features_batch(samples)
        print(
            f"  {tx_id}: extracted {features[tx_id].shape[0]} packet feature maps "
            f"({features[tx_id].shape[1]} x {features[tx_id].shape[2]})"
        )
    return features


def collapse_sequence_dataset(sequence_dataset):
    """
    Collapse packet-level temporal features into one 9-feature vector per packet.
    This is useful for summaries, profiling, and visualization helpers.
    """
    collapsed = {}
    for tx_id, packets in sequence_dataset.items():
        collapsed[tx_id] = np.mean(packets, axis=1).astype(np.float32)
    return collapsed


FEATURE_NAMES = [
    "Amplitude Mean",
    "Amplitude Std",
    "Amplitude Max",
    "Phase Mean",
    "Phase Std",
    "Signal Variance",
    "Spectral Energy",
    "IQ Correlation",
    "Temporal Gradient",
]

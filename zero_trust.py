"""
Bayesian Zero Trust Evaluation Layer.
Continuously calculates trust scores based on RF fingerprint similarity,
behavioural consistency, and historical authentication results.
"""

import numpy as np
from config import (
    TRANSMITTER_IDS, INITIAL_TRUST_SCORE,
    TRUST_DECAY_RATE, TRUST_RECOVERY_RATE,
    TRUST_THRESHOLD_SUSPICIOUS, TRUST_THRESHOLD_COMPROMISED,
)


class ZeroTrustEvaluator:
    """
    Bayesian Zero Trust evaluation for enterprise WiFi devices.

    Trust score components:
    - RF Fingerprint Similarity (primary signal)
    - Behavioural Consistency (stability over time)
    - Historical Authentication (cumulative track record)
    """

    def __init__(self):
        # Trust scores per device
        self.trust_scores = {tx: INITIAL_TRUST_SCORE for tx in TRANSMITTER_IDS}

        # History tracking
        self.auth_history = {tx: [] for tx in TRANSMITTER_IDS}  # list of (cycle, result)
        self.consecutive_normal = {tx: 0 for tx in TRANSMITTER_IDS}
        self.consecutive_anomaly = {tx: 0 for tx in TRANSMITTER_IDS}
        self.total_checks = {tx: 0 for tx in TRANSMITTER_IDS}
        self.total_passes = {tx: 0 for tx in TRANSMITTER_IDS}

    def update_trust(self, tx_id, anomaly_result, cycle_num):
        """
        Update trust score for a device based on current anomaly detection result.

        Uses Bayesian-inspired update:
        - Prior: current trust score
        - Likelihood: RF fingerprint match quality
        - Posterior: updated trust score
        """
        if anomaly_result["status"] == "offline":
            return self.get_device_status(tx_id)

        is_anomaly = anomaly_result["is_anomaly"]
        deviation_ratio = anomaly_result["deviation_ratio"]
        current_trust = self.trust_scores[tx_id]

        self.total_checks[tx_id] += 1

        if is_anomaly:
            # ── Anomaly detected: decrease trust ──
            self.consecutive_anomaly[tx_id] += 1
            self.consecutive_normal[tx_id] = 0

            # Severity-based decay: larger deviations cause bigger drops
            severity = min(deviation_ratio, 5.0)  # cap severity
            decay = TRUST_DECAY_RATE * (severity / 2.0)

            # Accelerate decay for consecutive anomalies
            consecutive_penalty = 1.0 + (self.consecutive_anomaly[tx_id] - 1) * 0.3
            decay *= consecutive_penalty

            new_trust = max(0.0, current_trust - decay)
            auth_result = "FAIL"

        else:
            # ── Normal behaviour: slowly recover trust ──
            self.consecutive_normal[tx_id] += 1
            self.consecutive_anomaly[tx_id] = 0
            self.total_passes[tx_id] += 1

            # Historical reliability factor
            if self.total_checks[tx_id] > 0:
                reliability = self.total_passes[tx_id] / self.total_checks[tx_id]
            else:
                reliability = 1.0

            # Recovery is slower than decay (security principle)
            recovery = TRUST_RECOVERY_RATE * reliability
            new_trust = min(1.0, current_trust + recovery)
            auth_result = "PASS"

        self.trust_scores[tx_id] = new_trust
        self.auth_history[tx_id].append({
            "cycle": cycle_num,
            "result": auth_result,
            "trust": new_trust,
            "deviation": deviation_ratio,
        })

        return self.get_device_status(tx_id)

    def get_device_status(self, tx_id):
        """Get current status classification for a device."""
        trust = self.trust_scores[tx_id]

        if trust >= TRUST_THRESHOLD_SUSPICIOUS:
            status = "trusted"
            color = "green"
        elif trust >= TRUST_THRESHOLD_COMPROMISED:
            status = "suspicious"
            color = "yellow"
        else:
            status = "compromised"
            color = "red"

        return {
            "tx_id": tx_id,
            "trust_score": trust,
            "status": status,
            "color": color,
            "total_checks": self.total_checks[tx_id],
            "total_passes": self.total_passes[tx_id],
            "consecutive_anomaly": self.consecutive_anomaly[tx_id],
        }

    def get_trust_history(self, tx_id):
        """Get full authentication history for a device."""
        return self.auth_history[tx_id]

    def get_all_scores(self):
        """Get current trust scores for all devices."""
        return {tx: self.trust_scores[tx] for tx in TRANSMITTER_IDS}

    def reset_device(self, tx_id):
        """Reset trust state for a device (e.g., after re-provisioning)."""
        self.trust_scores[tx_id] = INITIAL_TRUST_SCORE
        self.consecutive_normal[tx_id] = 0
        self.consecutive_anomaly[tx_id] = 0

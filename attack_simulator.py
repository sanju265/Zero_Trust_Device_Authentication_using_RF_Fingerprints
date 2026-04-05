"""
Attack Simulator - Interactive RF identity injection tool.

Run this file in a separate terminal while main.py is running.
The simulator does not change MAC addresses. It injects the attacker's
RF signature under the target identity claim so the dashboard can prove
that RF fingerprints do not match the learned profile.
"""

import json
import os
import numpy as np

from config import ATTACK_SIGNAL_FILE, DEVICE_PROFILES, TRANSMITTER_IDS


ATTACKER = "TX3"    # RF source performing the attack
TARGET = "TX1"      # Claimed identity being imitated
SAMPLES_PER_CHECK = 5  # must match anomaly_detection.py


class AttackSimulator:
    """
    Used by main.py internally. See the standalone section below
    for the interactive command-line tool.
    """

    def __init__(self, test_features, attacker_tx=None, target_tx=None):
        self.attacker_tx = attacker_tx or ATTACKER
        self.target_tx = target_tx or TARGET
        self.test_features = test_features
        self.attack_active = False
        self.attack_cycle = None
        self.sample_indices = {tx: 0 for tx in TRANSMITTER_IDS}

    def check_attack_signal(self):
        """Check if the attack signal file exists (written by the CLI tool)."""
        if os.path.exists(ATTACK_SIGNAL_FILE):
            try:
                with open(ATTACK_SIGNAL_FILE, "r", encoding="utf-8") as handle:
                    signal = json.load(handle)
                if signal.get("active", False):
                    self.attacker_tx = signal.get("attacker", self.attacker_tx)
                    self.target_tx = signal.get("target", self.target_tx)
                    return True
                if self.attack_active:
                    self._stop_attack()
                return False
            except (json.JSONDecodeError, OSError):
                return False

        if self.attack_active:
            self._stop_attack()
        return False

    def start_attack(self, cycle_num):
        """Activate the RF identity injection attack."""
        self.attack_active = True
        self.attack_cycle = cycle_num

    def _stop_attack(self):
        """Internal: deactivate the attack."""
        self.attack_active = False

    def stop_attack(self):
        """Public stop method."""
        self._stop_attack()
        if os.path.exists(ATTACK_SIGNAL_FILE):
            os.remove(ATTACK_SIGNAL_FILE)

    def _get_averaged_features(self, tx_id):
        """Average consecutive held-out test features for a transmitter."""
        features = self.test_features[tx_id]
        n_rows = features.shape[0]
        rows = []
        for _ in range(SAMPLES_PER_CHECK):
            idx = self.sample_indices[tx_id] % n_rows
            rows.append(features[idx])
            self.sample_indices[tx_id] += 1
        return np.mean(rows, axis=0).astype(np.float32)

    def build_monitoring_snapshot(self, online_mask):
        """
        Build a live snapshot for all devices.

        During attack, the attacker's RF features are injected into the target
        claim slot. This should raise the target's RF score above its threshold.
        """
        snapshot = []

        for tx_id in TRANSMITTER_IDS:
            if not online_mask.get(tx_id, True):
                sample_shape = self.test_features[TRANSMITTER_IDS[0]].shape[1:]
                snapshot.append(np.zeros(sample_shape, dtype=np.float32))
                continue

            if self.attack_active and tx_id == self.target_tx:
                snapshot.append(self._get_averaged_features(self.attacker_tx))
            else:
                snapshot.append(self._get_averaged_features(tx_id))

        return np.array(snapshot, dtype=np.float32)

    def get_attack_info(self):
        """Get current attack status information."""
        if not self.attack_active:
            return None

        return {
            "attacker_tx": self.attacker_tx,
            "attacker_name": DEVICE_PROFILES[self.attacker_tx]["name"],
            "target_tx": self.target_tx,
            "target_name": DEVICE_PROFILES[self.target_tx]["name"],
            "attack_cycle": self.attack_cycle,
            "attack_mode": "rf_identity_injection",
        }

    def get_mac_conflicts(self):
        """MAC addresses remain static in this RF-focused simulation."""
        return []


def _write_signal(active, attacker=ATTACKER, target=TARGET):
    signal = {
        "active": active,
        "attacker": attacker,
        "target": target,
    }
    with open(ATTACK_SIGNAL_FILE, "w", encoding="utf-8") as handle:
        json.dump(signal, handle)


def interactive_cli():
    """Interactive attack simulator CLI."""
    os.system("clear" if os.name == "posix" else "cls")

    print()
    print("=" * 60)
    print("  RF IDENTITY INJECTION SIMULATOR")
    print("  Interactive Control Panel")
    print("=" * 60)
    print()
    print("  Current Configuration:")
    print(f"    RF source : {ATTACKER} ({DEVICE_PROFILES[ATTACKER]['name']})")
    print(f"    Claimed ID: {TARGET} ({DEVICE_PROFILES[TARGET]['name']})")
    print()
    print("  To change attacker/claimed identity, edit the ATTACKER and TARGET")
    print("  values at the top of attack_simulator.py")
    print()
    print("  Commands:")
    print("    attack  - Launch RF identity injection")
    print("    stop    - Stop attack, restore normal operation")
    print("    status  - Show current attack status")
    print("    exit    - Quit simulator")
    print()
    print("-" * 60)

    while True:
        try:
            cmd = input("\n  [attack-sim] >> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting simulator.")
            if os.path.exists(ATTACK_SIGNAL_FILE):
                os.remove(ATTACK_SIGNAL_FILE)
            break

        if cmd == "attack":
            _write_signal(active=True)
            print()
            print("  " + "=" * 50)
            print("  RF IDENTITY INJECTION ACTIVE")
            print("  " + "=" * 50)
            print(f"  RF source : {ATTACKER} ({DEVICE_PROFILES[ATTACKER]['name']})")
            print(f"  Claimed ID: {TARGET} ({DEVICE_PROFILES[TARGET]['name']})")
            print()
            print(f"  The dashboard will compare {ATTACKER}'s live RF signature")
            print(f"  against {TARGET}'s learned threshold within 20 seconds.")
            print("  " + "=" * 50)

        elif cmd == "stop":
            if os.path.exists(ATTACK_SIGNAL_FILE):
                os.remove(ATTACK_SIGNAL_FILE)
            print()
            print("  " + "=" * 50)
            print("  ATTACK STOPPED - NORMAL OPERATION RESTORED")
            print("  " + "=" * 50)
            print("  Devices return to their normal RF signature windows.")
            print("  Threshold comparisons and trust scores will recover gradually.")
            print("  " + "=" * 50)

        elif cmd == "status":
            if os.path.exists(ATTACK_SIGNAL_FILE):
                try:
                    with open(ATTACK_SIGNAL_FILE, "r", encoding="utf-8") as handle:
                        signal = json.load(handle)
                    if signal.get("active"):
                        print("\n  Status: ATTACK ACTIVE")
                        print(
                            f"  RF source: {signal['attacker']} -> "
                            f"Claimed identity: {signal['target']}"
                        )
                    else:
                        print("\n  Status: Attack stopped (signal inactive)")
                except Exception:
                    print("\n  Status: Signal file corrupted")
            else:
                print("\n  Status: No active attack. System normal.")

        elif cmd in ("exit", "quit", "q"):
            if os.path.exists(ATTACK_SIGNAL_FILE):
                os.remove(ATTACK_SIGNAL_FILE)
            print("\n  Exiting simulator. Attack signal cleaned up.")
            break

        else:
            print("  Unknown command. Use: attack, stop, status, exit")


if __name__ == "__main__":
    interactive_cli()

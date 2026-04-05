"""
Offline ManySig preparation entrypoint.

Run this once when you want to:
- load ManySig.pkl
- split train/test
- extract RF fingerprint features
- train the TSGAE model
- compute thresholds
- save the prepared runtime cache
- print paper metrics and generate graphs

After this, use `python main.py` for repeated login + live testing runs.
"""

from main import generate_results, initialize_pipeline
from zero_trust import ZeroTrustEvaluator


def main():
    pipeline = initialize_pipeline()
    generate_results(pipeline, ZeroTrustEvaluator(), attack_start_cycle=0)
    print("\n[Training] Offline preparation complete.")
    print("[Training] You can now run `python main.py` for dashboard testing without retraining.")
    print("[Training] If you prefer, `python model.py` now launches this same offline preparation flow.")


if __name__ == "__main__":
    main()

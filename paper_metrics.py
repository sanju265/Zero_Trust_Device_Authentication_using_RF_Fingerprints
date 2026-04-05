"""
Held-out evaluation metrics for reports and paper-style summaries.
Computes claim-level authentication metrics on the ManySig test split.
"""

import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_detection import evaluate_snapshot
from config import (
    ATTACKER_TX,
    PAPER_METRICS_PATH,
    REPORTS_DIR,
    SAMPLE_SHAPE,
    TARGET_TX,
    TRAIN_RATIO,
    TRANSMITTER_IDS,
)


def _build_block_snapshot(scaled_test, round_idx, samples_per_check=5):
    """Average a non-overlapping block of test features per transmitter."""
    start = round_idx * samples_per_check
    stop = start + samples_per_check

    snapshot = []
    for tx_id in TRANSMITTER_IDS:
        block = scaled_test[tx_id][start:stop]
        snapshot.append(np.mean(block, axis=0))

    return np.array(snapshot, dtype=np.float32)


def _wilson_lower_bound(successes, trials, z=1.96):
    """Return a conservative 95% lower bound for a binomial proportion."""
    if trials <= 0:
        return 0.0

    phat = successes / trials
    z_sq = z ** 2
    denominator = 1.0 + z_sq / trials
    center = phat + z_sq / (2.0 * trials)
    margin = z * np.sqrt((phat * (1.0 - phat) + z_sq / (4.0 * trials)) / trials)
    return float(max(0.0, (center - margin) / denominator))


def paper_metrics_need_refresh(metrics):
    """Detect legacy cached metrics that predate claim-level evaluation."""
    if not metrics:
        return True
    summary = metrics.get("summary") or {}
    return (
        metrics.get("evaluation_mode") != "claim_level_authentication"
        or "conservative_recall" not in summary
        or "attack_pairs_per_round" not in metrics
    )


def compute_paper_metrics(
    model,
    scaled_test,
    adj_tensor,
    thresholds,
    profiler,
    attacker_tx=ATTACKER_TX,
    target_tx=TARGET_TX,
    samples_per_check=5,
):
    """Compute claim-level authentication metrics from held-out ManySig test windows."""
    available_rounds = min(
        scaled_test[tx_id].shape[0] // samples_per_check
        for tx_id in TRANSMITTER_IDS
    )

    online_mask = {tx_id: True for tx_id in TRANSMITTER_IDS}
    tx_index = {tx_id: idx for idx, tx_id in enumerate(TRANSMITTER_IDS)}
    attack_pairs = [
        (candidate_attacker, candidate_target)
        for candidate_attacker in TRANSMITTER_IDS
        for candidate_target in TRANSMITTER_IDS
        if candidate_attacker != candidate_target
    ]

    y_true = []
    y_pred = []
    y_score = []
    legitimate_scores = []
    legitimate_ratios = []
    attack_scores = []
    attack_ratios = []
    configured_attack_hits = 0
    per_attack_pair = {
        f"{candidate_attacker}->{candidate_target}": {
            "attacker_tx": candidate_attacker,
            "target_tx": candidate_target,
            "detections": 0,
            "ratios": [],
            "scores": [],
        }
        for candidate_attacker, candidate_target in attack_pairs
    }

    per_device_legitimate = {
        tx_id: {"scores": [], "ratios": []}
        for tx_id in TRANSMITTER_IDS
    }

    for round_idx in range(available_rounds):
        legitimate_snapshot = _build_block_snapshot(scaled_test, round_idx, samples_per_check)
        legitimate_results = evaluate_snapshot(
            model, legitimate_snapshot, adj_tensor, thresholds, online_mask, profiler
        )

        for tx_id in TRANSMITTER_IDS:
            result = legitimate_results[tx_id]
            y_true.append(0)
            y_pred.append(1 if result["is_anomaly"] else 0)
            y_score.append(result["deviation_ratio"])
            legitimate_scores.append(result["error"])
            legitimate_ratios.append(result["deviation_ratio"])
            per_device_legitimate[tx_id]["scores"].append(result["error"])
            per_device_legitimate[tx_id]["ratios"].append(result["deviation_ratio"])

        for candidate_attacker, candidate_target in attack_pairs:
            attack_snapshot = legitimate_snapshot.copy()
            attack_snapshot[tx_index[candidate_target]] = legitimate_snapshot[tx_index[candidate_attacker]]
            attack_result = evaluate_snapshot(
                model, attack_snapshot, adj_tensor, thresholds, online_mask, profiler
            )[candidate_target]

            key = f"{candidate_attacker}->{candidate_target}"
            pair_entry = per_attack_pair[key]
            pair_entry["scores"].append(attack_result["error"])
            pair_entry["ratios"].append(attack_result["deviation_ratio"])
            if attack_result["is_anomaly"]:
                pair_entry["detections"] += 1

            y_true.append(1)
            y_pred.append(1 if attack_result["is_anomaly"] else 0)
            y_score.append(attack_result["deviation_ratio"])
            attack_scores.append(attack_result["error"])
            attack_ratios.append(attack_result["deviation_ratio"])

            if candidate_attacker == attacker_tx and candidate_target == target_tx and attack_result["is_anomaly"]:
                configured_attack_hits += 1

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    legitimate_accept_rate = tn / (tn + fp) if (tn + fp) else 0.0
    attack_detection_rate = tp / (tp + fn) if (tp + fn) else 0.0
    configured_attack_detection_rate = configured_attack_hits / available_rounds if available_rounds else 0.0
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    average_precision = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    legitimate_p95 = float(np.percentile(legitimate_scores, 95)) if legitimate_scores else 0.0
    legitimate_p99 = float(np.percentile(legitimate_scores, 99)) if legitimate_scores else 0.0
    attack_p05 = float(np.percentile(attack_scores, 5)) if attack_scores else 0.0
    score_margin = attack_p05 - legitimate_p95
    conservative_precision = _wilson_lower_bound(tp, tp + fp)
    conservative_recall = _wilson_lower_bound(tp, tp + fn)
    attack_pair_count = len(attack_pairs)

    return {
        "evaluation_mode": "claim_level_authentication",
        "rounds": available_rounds,
        "samples_per_check": samples_per_check,
        "attacker_tx": attacker_tx,
        "target_tx": target_tx,
        "configured_scenario": {
            "attacker_tx": attacker_tx,
            "target_tx": target_tx,
        },
        "attack_pairs_per_round": attack_pair_count,
        "total_legitimate_claims": int(available_rounds * len(TRANSMITTER_IDS)),
        "total_attack_claims": int(available_rounds * attack_pair_count),
        "summary": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "observed_precision": precision,
            "observed_recall": recall,
            "conservative_precision": conservative_precision,
            "conservative_recall": conservative_recall,
            "precision_lower_bound_95": conservative_precision,
            "recall_lower_bound_95": conservative_recall,
            "f1_score": f1,
            "false_positive_rate": false_positive_rate,
            "legitimate_accept_rate": legitimate_accept_rate,
            "attack_detection_rate": attack_detection_rate,
            "configured_attack_detection_rate": configured_attack_detection_rate,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "legitimate_mean_score": float(np.mean(legitimate_scores)) if legitimate_scores else 0.0,
            "legitimate_mean_ratio": float(np.mean(legitimate_ratios)) if legitimate_ratios else 0.0,
            "legitimate_p95_score": legitimate_p95,
            "legitimate_p99_score": legitimate_p99,
            "attack_mean_score": float(np.mean(attack_scores)) if attack_scores else 0.0,
            "attack_mean_ratio": float(np.mean(attack_ratios)) if attack_ratios else 0.0,
            "attack_p05_score": attack_p05,
            "score_margin": score_margin,
        },
        "per_device": {
            tx_id: {
                "threshold": float(thresholds[tx_id]),
                "mean_legitimate_score": float(np.mean(values["scores"])) if values["scores"] else 0.0,
                "max_legitimate_score": float(np.max(values["scores"])) if values["scores"] else 0.0,
                "mean_legitimate_ratio": float(np.mean(values["ratios"])) if values["ratios"] else 0.0,
            }
            for tx_id, values in per_device_legitimate.items()
        },
        "per_attack_pair": {
            key: {
                "attacker_tx": values["attacker_tx"],
                "target_tx": values["target_tx"],
                "detections": int(values["detections"]),
                "rounds": available_rounds,
                "detection_rate": (values["detections"] / available_rounds) if available_rounds else 0.0,
                "mean_attack_score": float(np.mean(values["scores"])) if values["scores"] else 0.0,
                "mean_attack_ratio": float(np.mean(values["ratios"])) if values["ratios"] else 0.0,
            }
            for key, values in per_attack_pair.items()
        },
    }


def save_paper_metrics(metrics):
    """Persist the paper metrics report to the reports folder."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    summary = metrics["summary"]
    lines = [
        "=" * 68,
        "  PAPER METRICS SUMMARY",
        "=" * 68,
        "Evaluation mode          : Claim-level held-out authentication",
        f"Held-out test rounds        : {metrics['rounds']}",
        f"Samples averaged per check  : {metrics['samples_per_check']}",
        f"Legitimate claims tested    : {metrics['total_legitimate_claims']}",
        f"Attack claims tested        : {metrics['total_attack_claims']}",
        f"Attack pairs per round      : {metrics['attack_pairs_per_round']}",
        f"Configured live scenario    : {metrics['attacker_tx']} -> {metrics['target_tx']}",
        "",
        "Detection metrics:",
        f"  Accuracy                  : {summary['accuracy']:.4f}",
        f"  Precision (95% LB)        : {summary['conservative_precision']:.4f}",
        f"  Recall (95% LB)           : {summary['conservative_recall']:.4f}",
        f"  Observed precision        : {summary['observed_precision']:.4f}",
        f"  Observed recall           : {summary['observed_recall']:.4f}",
        f"  F1-score                  : {summary['f1_score']:.4f}",
        f"  False positive rate       : {summary['false_positive_rate']:.4f}",
        f"  Legitimate accept rate    : {summary['legitimate_accept_rate']:.4f}",
        f"  Cross-pair detect rate    : {summary['attack_detection_rate']:.4f}",
        f"  Configured detect rate    : {summary['configured_attack_detection_rate']:.4f}",
        f"  ROC-AUC                   : {summary['roc_auc']:.4f}",
        f"  Average precision         : {summary['average_precision']:.4f}",
        f"  Claim confusion counts    : TP={summary['tp']} TN={summary['tn']} FP={summary['fp']} FN={summary['fn']}",
        "",
        "Anomaly-score summary:",
        f"  Mean legitimate score     : {summary['legitimate_mean_score']:.6f}",
        f"  Mean legitimate ratio     : {summary['legitimate_mean_ratio']:.4f}x",
        f"  Legitimate P95 score      : {summary['legitimate_p95_score']:.6f}",
        f"  Legitimate P99 score      : {summary['legitimate_p99_score']:.6f}",
        f"  Mean attack score         : {summary['attack_mean_score']:.6f}",
        f"  Mean attack ratio         : {summary['attack_mean_ratio']:.4f}x",
        f"  Attack P05 score          : {summary['attack_p05_score']:.6f}",
        f"  Score margin (P05-P95)    : {summary['score_margin']:.6f}",
        "",
        "Per-device thresholds:",
        f"{'TX':<6} {'Threshold':<12} {'Mean Legit':<12} {'Max Legit':<12} {'Mean Ratio':<12}",
        "-" * 68,
    ]

    for tx_id, values in metrics["per_device"].items():
        lines.append(
            f"{tx_id:<6} {values['threshold']:<12.6f} {values['mean_legitimate_score']:<12.6f} "
            f"{values['max_legitimate_score']:<12.6f} {values['mean_legitimate_ratio']:<12.4f}"
        )

    lines.append("=" * 68)

    with open(PAPER_METRICS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return PAPER_METRICS_PATH


def print_paper_metrics(metrics, dataset_info, training_info):
    """Print a concise report in the terminal for demos and papers."""
    summary = metrics["summary"]
    artifact_mode = "cached artifacts" if training_info.get("from_cache") else "fresh training"
    total_packets_per_tx = training_info.get("packets_per_device_train", 0) + training_info.get("packets_per_device_test", 0)
    values_per_packet = SAMPLE_SHAPE[0] * SAMPLE_SHAPE[1]

    print("\n" + "=" * 68)
    print("  MANYSIG VERIFICATION & PAPER METRICS")
    print("=" * 68)
    print(f"  Dataset file      : {dataset_info['path']}")
    print(f"  Dataset size      : {dataset_info['size_mb']} MB")
    print(f"  Dataset modified  : {dataset_info['modified_at']}")
    print(f"  Artifact mode     : {artifact_mode}")
    print(f"  Epochs            : {training_info.get('epochs', 'N/A')}")
    print(f"  Steps / epoch     : {training_info.get('steps_per_epoch', 'N/A')}")
    print(f"  Training duration : {training_info.get('duration_seconds', 'N/A')} s")
    print(
        f"  Split per TX      : {total_packets_per_tx or 'N/A'} packets total | "
        f"{training_info.get('packets_per_device_train', 'N/A')} train | "
        f"{training_info.get('packets_per_device_test', 'N/A')} test"
    )
    print(
        f"  Packet geometry   : {SAMPLE_SHAPE[0]} samples x {SAMPLE_SHAPE[1]} I/Q channels = "
        f"{values_per_packet} values per packet"
    )
    print(f"  Train ratio       : {TRAIN_RATIO:.0%} one-time model fitting, 20% held-out runtime/test")
    print(f"  Test rounds       : {metrics['rounds']} windows of {metrics['samples_per_check']} samples")
    print(
        f"  Claims tested     : legit={metrics['total_legitimate_claims']} | "
        f"attack={metrics['total_attack_claims']} across {metrics['attack_pairs_per_round']} attack pairs/round"
    )
    print(f"  Live scenario     : {metrics['attacker_tx']} -> {metrics['target_tx']}")
    print("-" * 68)
    print(f"  Accuracy          : {summary['accuracy']:.4f}")
    print(f"  Precision (95%LB) : {summary['conservative_precision']:.4f}")
    print(f"  Recall (95%LB)    : {summary['conservative_recall']:.4f}")
    print(f"  Obs Precision     : {summary['observed_precision']:.4f}")
    print(f"  Obs Recall        : {summary['observed_recall']:.4f}")
    print(f"  F1-score          : {summary['f1_score']:.4f}")
    print(f"  False Positive    : {summary['false_positive_rate']:.4f}")
    print(f"  Legit Accept      : {summary['legitimate_accept_rate']:.4f}")
    print(f"  Cross-pair Detect : {summary['attack_detection_rate']:.4f}")
    print(f"  Live Pair Detect  : {summary['configured_attack_detection_rate']:.4f}")
    print(f"  ROC-AUC           : {summary['roc_auc']:.4f}")
    print(f"  Avg Precision     : {summary['average_precision']:.4f}")
    print(
        f"  Claim Confusion   : TP={summary['tp']} TN={summary['tn']} "
        f"FP={summary['fp']} FN={summary['fn']}"
    )
    print(
        f"  Score ratios      : legit={summary['legitimate_mean_ratio']:.4f}x "
        f"attack={summary['attack_mean_ratio']:.4f}x"
    )
    print(
        f"  Score spread      : legit P95={summary['legitimate_p95_score']:.6f} | "
        f"attack P05={summary['attack_p05_score']:.6f} | margin={summary['score_margin']:.6f}"
    )
    print("-" * 68)
    print(f"{'TX':<6} {'Threshold':<12} {'Mean Legit':<12} {'Max Legit':<12}")
    for tx_id, values in metrics["per_device"].items():
        print(
            f"{tx_id:<6} {values['threshold']:<12.6f} "
            f"{values['mean_legitimate_score']:<12.6f} {values['max_legitimate_score']:<12.6f}"
        )
    print("=" * 68)

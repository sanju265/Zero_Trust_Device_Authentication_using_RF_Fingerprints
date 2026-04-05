"""
WiFi Device Impersonation Detection using RF fingerprinting
with a Temporal Spectral Graph Autoencoder (TSGAE) and
Bayesian Zero Trust decision layer.

Workflow:
1. Training/preparation happens offline once from ManySig.pkl.
2. The prepared model, thresholds, and held-out test features are saved.
3. main.py opens the login page only for live testing/monitoring.
"""

import os
import pickle
import sys
import time
from datetime import datetime

import torch

from anomaly_detection import compute_dynamic_thresholds, evaluate_snapshot, load_thresholds
from attack_simulator import AttackSimulator, SAMPLES_PER_CHECK
from config import (
    ATTACK_SIGNAL_FILE,
    DATASET_PATH,
    DEVICE_PROFILES,
    HIDDEN_DIM,
    LATENT_DIM,
    MODEL_ARTIFACT_VERSION,
    MODEL_SAVE_PATH,
    MONITOR_INTERVAL,
    NUM_EPOCHS,
    NUM_FEATURES,
    OFFLINE_TRANSMITTERS,
    PAPER_METRICS_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    RUNTIME_CACHE_PATH,
    SAMPLE_SHAPE,
    SCALER_SAVE_PATH,
    SNAPSHOTS_PER_EPOCH,
    THRESHOLD_SAVE_PATH,
    THRESHOLD_SIGMA,
    TRAIN_RATIO,
    TRANSMITTER_IDS,
)
from dashboard import (
    get_dashboard_snapshot,
    get_dashboard_url,
    get_login_url,
    render_dashboard,
    save_attack_report,
    set_dashboard_phase,
    set_summary_report_path,
    start_dashboard_server,
    stop_dashboard_server,
    wait_for_login,
)
from data_loader import describe_dataset_source, load_manysig_dataset, split_train_test
from feature_extraction import collapse_sequence_dataset, extract_all_features, extract_all_sequence_features
from graph_construction import build_similarity_graph, build_temporal_graph, get_adjacency_tensor
from model import load_tsgae, train_tsgae
from paper_metrics import (
    compute_paper_metrics,
    paper_metrics_need_refresh,
    print_paper_metrics,
    save_paper_metrics,
)
from preprocessing import fit_scaler, load_scaler, scale_features
from zero_trust import ZeroTrustEvaluator


def publish_step(headline, detail, extra=None):
    """Keep the dashboard store in sync when a web session exists."""
    set_dashboard_phase("initializing", headline, detail, extra=extra)


def cleanup_attack_signal():
    """Remove stale attack signal file."""
    if os.path.exists(ATTACK_SIGNAL_FILE):
        os.remove(ATTACK_SIGNAL_FILE)


def build_artifact_meta(dataset_info):
    """Build a reproducible signature for model and runtime cache files."""
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "dataset_filename": dataset_info["filename"],
        "dataset_path": dataset_info["path"],
        "dataset_size_bytes": dataset_info["size_bytes"],
        "dataset_modified_timestamp": dataset_info["modified_timestamp"],
        "train_ratio": TRAIN_RATIO,
        "sample_shape": SAMPLE_SHAPE,
        "num_features": NUM_FEATURES,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": NUM_EPOCHS,
        "snapshots_per_epoch": SNAPSHOTS_PER_EPOCH,
        "threshold_sigma": THRESHOLD_SIGMA,
    }


def get_cache_status(expected_meta):
    """Check whether the saved core artifacts match the active ManySig dataset and config."""
    if os.environ.get("RF_FORCE_RETRAIN", "0") == "1":
        return False, "RF_FORCE_RETRAIN=1 requested."

    required_paths = [MODEL_SAVE_PATH, SCALER_SAVE_PATH, THRESHOLD_SAVE_PATH]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        return False, "Saved scaler/model/threshold files are missing."

    try:
        _model, metadata = load_tsgae()
    except Exception as exc:
        return False, f"Could not load saved model bundle: {exc}"

    saved_meta = metadata.get("artifact_meta") or {}
    if not saved_meta:
        return False, "Saved model bundle has no artifact metadata to verify."

    for key, value in expected_meta.items():
        if saved_meta.get(key) != value:
            return False, f"Artifact metadata mismatch for '{key}'."

    return True, metadata


def build_split_info(dataset, train_data, test_data):
    """Summarize the packet and scalar split for a single transmitter."""
    first_tx = TRANSMITTER_IDS[0]
    return {
        "total_samples_per_tx": int(dataset[first_tx].shape[0]),
        "train_samples_per_tx": int(train_data[first_tx].shape[0]),
        "test_samples_per_tx": int(test_data[first_tx].shape[0]),
        "samples_per_packet": int(dataset[first_tx].shape[1]),
        "iq_channels": int(dataset[first_tx].shape[2]),
        "total_signal_values_per_tx": int(dataset[first_tx].shape[0] * dataset[first_tx].shape[1] * dataset[first_tx].shape[2]),
        "train_signal_values_per_tx": int(train_data[first_tx].shape[0] * train_data[first_tx].shape[1] * train_data[first_tx].shape[2]),
        "test_signal_values_per_tx": int(test_data[first_tx].shape[0] * test_data[first_tx].shape[1] * test_data[first_tx].shape[2]),
        "samples_per_refresh": int(SAMPLES_PER_CHECK),
        "unique_refresh_cycles": int(test_data[first_tx].shape[0] // SAMPLES_PER_CHECK),
    }


def print_split_summary(split_info):
    """Print the train/test geometry that you present in demos or viva."""
    print(
        f"\n[Split] Per transmitter: total={split_info['total_samples_per_tx']}, "
        f"train={split_info['train_samples_per_tx']}, test={split_info['test_samples_per_tx']}"
    )
    print(
        f"[Split] Each refresh uses {split_info['samples_per_refresh']} held-out test samples, "
        f"so there are {split_info['unique_refresh_cycles']} unique refresh cycles before repetition."
    )
    print(
        f"[Geometry] Each transmitter has {split_info['total_samples_per_tx']} packets x "
        f"{split_info['samples_per_packet']} I/Q samples x {split_info['iq_channels']} channels = "
        f"{split_info['total_signal_values_per_tx']:,} scalar values."
    )
    print(
        f"[Geometry] Training uses {split_info['train_samples_per_tx']} packets "
        f"({split_info['train_signal_values_per_tx']:,} scalar values) and live monitoring uses the remaining "
        f"{split_info['test_samples_per_tx']} packets ({split_info['test_signal_values_per_tx']:,} scalar values)."
    )


def save_runtime_cache(artifact_meta, payload):
    """Persist the prepared runtime bundle for repeated dashboard testing."""
    runtime_payload = dict(payload)
    runtime_payload["artifact_meta"] = artifact_meta
    with open(RUNTIME_CACHE_PATH, "wb") as handle:
        pickle.dump(runtime_payload, handle)


def load_runtime_cache(expected_meta):
    """Load the prepared runtime bundle if it matches the current ManySig file."""
    cache_ok, _cache_details = get_cache_status(expected_meta)
    if not cache_ok:
        return None, "Core training artifacts are missing or outdated."

    if not os.path.exists(RUNTIME_CACHE_PATH):
        return None, "Runtime cache file is missing."

    try:
        with open(RUNTIME_CACHE_PATH, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return None, f"Could not load runtime cache: {exc}"

    saved_meta = payload.get("artifact_meta") or {}
    for key, value in expected_meta.items():
        if saved_meta.get(key) != value:
            return None, f"Runtime cache mismatch for '{key}'."

    model, metadata = load_tsgae()
    thresholds, error_stats, profiler = load_thresholds()
    training_info = dict(payload.get("training_info") or metadata.get("training_info") or {})
    training_info["from_cache"] = True
    training_info["cache_status"] = "verified"
    training_info["cache_message"] = "Runtime cache, model, scaler, and thresholds match the current ManySig file."

    pipeline = {
        "dataset_info": payload["dataset_info"],
        "split_info": payload["split_info"],
        "scaled_test": payload["scaled_test"],
        "adj_tensor": torch.FloatTensor(payload["adj_tensor"]),
        "model": model,
        "thresholds": thresholds,
        "error_stats": error_stats,
        "profiler": profiler,
        "loss_history": metadata.get("loss_history", []),
        "training_info": training_info,
        "paper_metrics": payload["paper_metrics"],
    }
    return pipeline, None


def prepare_model_artifacts(train_sequence_features, test_sequence_features, dataset_info):
    """Load verified cached artifacts or train fresh artifacts from ManySig."""
    artifact_meta = build_artifact_meta(dataset_info)
    cache_ok, cache_details = get_cache_status(artifact_meta)

    if cache_ok:
        print(f"\n[Artifacts] Using verified cached artifacts from {DATASET_PATH}")
        scaler = load_scaler()
        scaled_train = scale_features(train_sequence_features, scaler)
        scaled_test = scale_features(test_sequence_features, scaler)
        temporal_adj_matrix, _step_means = build_temporal_graph(scaled_train)
        device_graph_features = collapse_sequence_dataset(scaled_train)
        _device_adj_matrix, _centroids, graph = build_similarity_graph(device_graph_features)
        adj_tensor = get_adjacency_tensor(temporal_adj_matrix)
        model, metadata = load_tsgae()
        thresholds, error_stats, profiler = load_thresholds()

        training_info = dict(metadata.get("training_info") or {})
        training_info.setdefault("epochs", NUM_EPOCHS)
        training_info.setdefault("steps_per_epoch", training_info.get("steps_per_epoch", "N/A"))
        training_info["from_cache"] = True
        training_info["cache_status"] = "verified"
        training_info["cache_message"] = "Artifacts match the current ManySig file and config."
        loss_history = metadata.get("loss_history", [])
    else:
        print(f"\n[Artifacts] Training fresh artifacts from {DATASET_PATH}")
        print(f"[Artifacts] Reason: {cache_details}")

        scaler = fit_scaler(train_sequence_features)
        scaled_train = scale_features(train_sequence_features, scaler)
        scaled_test = scale_features(test_sequence_features, scaler)

        temporal_adj_matrix, _step_means = build_temporal_graph(scaled_train)
        device_graph_features = collapse_sequence_dataset(scaled_train)
        _device_adj_matrix, _centroids, graph = build_similarity_graph(device_graph_features)
        adj_tensor = get_adjacency_tensor(temporal_adj_matrix)

        model, loss_history, training_info = train_tsgae(
            scaled_train,
            adj_tensor,
            epochs=NUM_EPOCHS,
            artifact_meta=artifact_meta,
        )

        thresholds, error_stats, profiler = compute_dynamic_thresholds(model, scaled_train, adj_tensor)
        training_info["from_cache"] = False
        training_info["cache_status"] = "fresh"
        training_info["cache_message"] = "Artifacts were freshly trained from the current ManySig file."

    return {
        "artifact_meta": artifact_meta,
        "scaled_train": scaled_train,
        "scaled_test": scaled_test,
        "adj_tensor": adj_tensor,
        "G": graph,
        "model": model,
        "loss_history": loss_history,
        "thresholds": thresholds,
        "error_stats": error_stats,
        "profiler": profiler,
        "training_info": training_info,
    }


def initialize_pipeline():
    """Run the full one-time ManySig preparation pipeline and save runtime cache."""
    print("=" * 68)
    print("  PHASE 1: MANYSIG DATASET AND MODEL PREPARATION")
    print("=" * 68)

    dataset_info = describe_dataset_source(DATASET_PATH)
    dataset = load_manysig_dataset()
    train_data, test_data = split_train_test(dataset)
    split_info = build_split_info(dataset, train_data, test_data)
    print_split_summary(split_info)

    print("\n[Step 3] Extracting packet-level temporal RF features (256 x 9)...")
    print("  Training features:")
    train_sequence_features = extract_all_sequence_features(train_data)
    print("  Testing features:")
    test_sequence_features = extract_all_sequence_features(test_data)

    print("\n[Step 4] Extracting aggregate RF features for reports and graphs...")
    train_features = extract_all_features(train_data)

    artifact_bundle = prepare_model_artifacts(train_sequence_features, test_sequence_features, dataset_info)

    paper_metrics = compute_paper_metrics(
        model=artifact_bundle["model"],
        scaled_test=artifact_bundle["scaled_test"],
        adj_tensor=artifact_bundle["adj_tensor"],
        thresholds=artifact_bundle["thresholds"],
        profiler=artifact_bundle["profiler"],
    )
    metrics_path = save_paper_metrics(paper_metrics)
    artifact_bundle["training_info"]["metrics_report_path"] = metrics_path

    print_paper_metrics(paper_metrics, dataset_info, artifact_bundle["training_info"])

    save_runtime_cache(
        artifact_bundle["artifact_meta"],
        {
            "dataset_info": dataset_info,
            "split_info": split_info,
            "scaled_test": artifact_bundle["scaled_test"],
            "adj_tensor": artifact_bundle["adj_tensor"].cpu().numpy(),
            "training_info": artifact_bundle["training_info"],
            "paper_metrics": paper_metrics,
        },
    )
    print(f"\n[Runtime] Prepared dashboard runtime cache saved to: {RUNTIME_CACHE_PATH}")

    print("\n" + "=" * 68)
    print("  MANYSIG MODEL PREPARATION COMPLETE")
    print(f"  Paper metrics saved to: {metrics_path}")
    print("=" * 68)

    return {
        "dataset": dataset,
        "dataset_info": dataset_info,
        "split_info": split_info,
        "train_features": train_features,
        "scaled_train": collapse_sequence_dataset(artifact_bundle["scaled_train"]),
        "scaled_test": artifact_bundle["scaled_test"],
        "adj_tensor": artifact_bundle["adj_tensor"],
        "G": artifact_bundle["G"],
        "model": artifact_bundle["model"],
        "loss_history": artifact_bundle["loss_history"],
        "thresholds": artifact_bundle["thresholds"],
        "error_stats": artifact_bundle["error_stats"],
        "profiler": artifact_bundle["profiler"],
        "training_info": artifact_bundle["training_info"],
        "paper_metrics": paper_metrics,
    }


def load_or_prepare_runtime_pipeline():
    """Load a saved runtime bundle for testing only."""
    dataset_info = describe_dataset_source(DATASET_PATH)
    expected_meta = build_artifact_meta(dataset_info)

    cached_pipeline, reason = load_runtime_cache(expected_meta)
    if cached_pipeline is not None:
        if paper_metrics_need_refresh(cached_pipeline.get("paper_metrics")):
            print("\n[Runtime] Refreshing held-out metrics for claim-level evaluation...")
            refreshed_metrics = compute_paper_metrics(
                model=cached_pipeline["model"],
                scaled_test=cached_pipeline["scaled_test"],
                adj_tensor=cached_pipeline["adj_tensor"],
                thresholds=cached_pipeline["thresholds"],
                profiler=cached_pipeline["profiler"],
            )
            metrics_path = save_paper_metrics(refreshed_metrics)
            cached_pipeline["paper_metrics"] = refreshed_metrics
            cached_pipeline["training_info"]["metrics_report_path"] = metrics_path
            save_runtime_cache(
                expected_meta,
                {
                    "dataset_info": cached_pipeline["dataset_info"],
                    "split_info": cached_pipeline["split_info"],
                    "scaled_test": cached_pipeline["scaled_test"],
                    "adj_tensor": cached_pipeline["adj_tensor"].cpu().numpy(),
                    "training_info": cached_pipeline["training_info"],
                    "paper_metrics": refreshed_metrics,
                },
            )

        print("=" * 68)
        print("  PREPARED RUNTIME LOADED")
        print("=" * 68)
        print(f"\n[Runtime] Using verified prepared runtime from {RUNTIME_CACHE_PATH}")
        print("[Runtime] Offline training is already complete. Starting live testing flow only.")
        return cached_pipeline

    raise RuntimeError(
        "No valid prepared runtime was found.\n"
        f"Reason: {reason}\n"
        "Run `python model.py` first to complete offline training, save metrics, and store the testing cache."
    )


def hydrate_visualization_context(pipeline):
    """Load the extra dataset context needed for end-of-run graphs."""
    required_keys = {"dataset", "train_features", "scaled_train", "G"}
    if required_keys.issubset(pipeline):
        return pipeline

    print("\n[Visualization] Loading dataset context for final graphs...")
    dataset = load_manysig_dataset()
    train_data, _test_data = split_train_test(dataset)
    train_features = extract_all_features(train_data)
    train_sequence_features = extract_all_sequence_features(train_data)
    scaler = load_scaler()
    scaled_train_sequence = scale_features(train_sequence_features, scaler)
    device_graph_features = collapse_sequence_dataset(scaled_train_sequence)
    _device_adj_matrix, _centroids, graph = build_similarity_graph(device_graph_features)

    pipeline.update(
        {
            "dataset": dataset,
            "train_features": train_features,
            "scaled_train": device_graph_features,
            "G": graph,
        }
    )
    return pipeline


def _ratio_text(value):
    return f"{value:.2f}x" if value is not None else "N/A"


def build_dashboard_metrics(paper_metrics):
    """Return a compact metric payload for the dashboard cards."""
    if not paper_metrics:
        return None

    summary = paper_metrics.get("summary") or {}
    compact_summary_keys = [
        "accuracy",
        "conservative_precision",
        "conservative_recall",
        "observed_precision",
        "observed_recall",
        "f1_score",
        "false_positive_rate",
        "legitimate_accept_rate",
        "attack_detection_rate",
        "configured_attack_detection_rate",
        "roc_auc",
        "average_precision",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    return {
        "evaluation_mode": paper_metrics.get("evaluation_mode"),
        "rounds": paper_metrics.get("rounds"),
        "samples_per_check": paper_metrics.get("samples_per_check"),
        "attack_pairs_per_round": paper_metrics.get("attack_pairs_per_round"),
        "total_legitimate_claims": paper_metrics.get("total_legitimate_claims"),
        "total_attack_claims": paper_metrics.get("total_attack_claims"),
        "configured_scenario": paper_metrics.get("configured_scenario"),
        "summary": {key: summary.get(key) for key in compact_summary_keys},
    }


def save_session_summary_report(pipeline, trust_evaluator, attack_start_cycle):
    """Save a final session summary with the latest dashboard evidence."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    dashboard_state = get_dashboard_snapshot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORTS_DIR, f"monitoring_summary_{timestamp}.txt")
    metrics_path = pipeline["training_info"].get("metrics_report_path", PAPER_METRICS_PATH)
    latest_attack_report = dashboard_state.get("report_path") or "No attack report was saved."
    paper_summary = (pipeline.get("paper_metrics") or {}).get("summary", {})

    lines = [
        "=" * 72,
        "  RF DASHBOARD SESSION SUMMARY",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        f"Dataset source          : {pipeline['dataset_info']['path']}",
        f"Refresh interval        : {MONITOR_INTERVAL}s",
        f"Attack start cycle      : {attack_start_cycle if attack_start_cycle is not None else 'No attack triggered'}",
        f"Latest attack report    : {latest_attack_report}",
        f"Paper metrics report    : {metrics_path}",
        f"Graphs directory        : {RESULTS_DIR}",
        "",
    ]

    if paper_summary:
        lines += [
            "HELD-OUT DETECTION SUMMARY:",
            f"  Accuracy              : {paper_summary['accuracy']:.4f}",
            f"  Precision (95% LB)    : {paper_summary['conservative_precision']:.4f}",
            f"  Recall (95% LB)       : {paper_summary['conservative_recall']:.4f}",
            f"  Observed precision    : {paper_summary['observed_precision']:.4f}",
            f"  Observed recall       : {paper_summary['observed_recall']:.4f}",
            f"  F1-score              : {paper_summary['f1_score']:.4f}",
            f"  False positive rate   : {paper_summary['false_positive_rate']:.4f}",
            f"  Cross-pair detect     : {paper_summary['attack_detection_rate']:.4f}",
            f"  Configured detect     : {paper_summary['configured_attack_detection_rate']:.4f}",
            f"  Claim confusion       : TP={paper_summary['tp']} TN={paper_summary['tn']} FP={paper_summary['fp']} FN={paper_summary['fn']}",
            f"  Mean legitimate ratio : {paper_summary['legitimate_mean_ratio']:.4f}x",
            f"  Mean attack ratio     : {paper_summary['attack_mean_ratio']:.4f}x",
            "",
        ]

    lines += [
        "FINAL DASHBOARD DEVICE STATE:",
        f"{'TX':<6} {'Verdict':<24} {'Auth':<10} {'Decision':<10} {'Recon':<10} {'Profile':<10} {'Trust':<8}",
        "-" * 72,
    ]

    for device in dashboard_state.get("device_states", []):
        trust_score = device.get("trust_score")
        lines.append(
            f"{device.get('tx_id', '--'):<6} "
            f"{device.get('verdict', '--'):<24} "
            f"{device.get('auth_result', '--'):<10} "
            f"{_ratio_text(device.get('score_ratio')):<10} "
            f"{_ratio_text(device.get('recon_ratio')):<10} "
            f"{_ratio_text(device.get('profile_ratio')):<10} "
            f"{(f'{trust_score:.2f}' if trust_score is not None else 'N/A'):<8}"
        )

    lines += [
        "",
        "FINAL TRUST STATUS:",
        f"{'TX':<6} {'Status':<14} {'Trust':<8} {'Checks':<8} {'Passes':<8}",
        "-" * 72,
    ]

    for tx_id in TRANSMITTER_IDS:
        status = trust_evaluator.get_device_status(tx_id)
        lines.append(
            f"{tx_id:<6} "
            f"{status['status']:<14} "
            f"{status['trust_score']:<8.2f} "
            f"{trust_evaluator.total_checks.get(tx_id, 0):<8} "
            f"{trust_evaluator.total_passes.get(tx_id, 0):<8}"
        )

    lines += [
        "",
        "=" * 72,
    ]

    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    set_summary_report_path(report_file)
    return report_file


def run_monitoring(pipeline):
    """Run real-time monitoring on the held-out ManySig test split."""
    print("\n" + "=" * 68)
    print("  PHASE 2: LIVE THRESHOLD MONITORING")
    print("=" * 68)

    model = pipeline["model"]
    adj_tensor = pipeline["adj_tensor"]
    thresholds = pipeline["thresholds"]
    scaled_test = pipeline["scaled_test"]
    profiler = pipeline["profiler"]

    trust_evaluator = ZeroTrustEvaluator()
    attack_sim = AttackSimulator(scaled_test)
    online_mask = {tx: tx not in OFFLINE_TRANSMITTERS for tx in TRANSMITTER_IDS}

    cleanup_attack_signal()

    set_dashboard_phase(
        "monitoring",
        "Live RF legitimacy monitoring",
        "Each device now uses held-out ManySig test windows, reconstruction evidence, and profile-distance checks for legitimacy decisions.",
        extra={
            "metrics_report_path": pipeline["training_info"].get("metrics_report_path", PAPER_METRICS_PATH),
            "analysis_metrics": build_dashboard_metrics(pipeline.get("paper_metrics")),
        },
    )

    print("[Core] Model          : Temporal Spectral Graph Autoencoder (TSGAE)")
    print("[Core] Decision layer : Bayesian Zero Trust")
    print(f"\n[Monitoring] Dataset source: {pipeline['dataset_info']['path']}")
    print(f"[Monitoring] Refresh interval: {MONITOR_INTERVAL}s")
    print(f"[Monitoring] Offline devices: {OFFLINE_TRANSMITTERS}")
    print(f"[Monitoring] Browser dashboard: {get_dashboard_url()}")
    print("\n  To trigger the RF identity injection, open another terminal:")
    print(f"    cd {os.path.dirname(os.path.abspath(__file__))}")
    print("    python attack_simulator.py")
    print("    >> attack")
    print("\n  Press Ctrl+C to stop monitoring.")
    print("\n  Starting in 3 seconds...")
    time.sleep(3)

    cycle = 0
    start_time = time.time()
    attack_start_cycle = None
    report_saved_for_attack = False

    try:
        while True:
            cycle += 1
            elapsed = int(time.time() - start_time)
            refresh_time = datetime.now().isoformat(timespec="seconds")

            signal_active = attack_sim.check_attack_signal()
            if signal_active and not attack_sim.attack_active:
                attack_sim.start_attack(cycle)
                attack_start_cycle = cycle
                report_saved_for_attack = False

            snapshot = attack_sim.build_monitoring_snapshot(online_mask)
            attack_info = attack_sim.get_attack_info()

            results = evaluate_snapshot(
                model=model,
                snapshot_features=snapshot,
                adj_tensor=adj_tensor,
                thresholds=thresholds,
                online_mask=online_mask,
                profiler=profiler,
            )

            device_states = {}
            for tx_id in TRANSMITTER_IDS:
                result = results[tx_id]
                trust_status = trust_evaluator.update_trust(tx_id, result, cycle)
                current_score = result.get("error")
                threshold_value = result.get("recon_threshold", thresholds.get(tx_id))
                decision_ratio = result.get("deviation_ratio", 0.0)
                recon_ratio = result.get("recon_ratio", 0.0)
                profile_ratio = result.get("profile_ratio", 0.0)
                profile_distance = result.get("maha_dist")
                attack_role = None
                if attack_info and tx_id == attack_info["target_tx"]:
                    attack_role = "target"
                elif attack_info and tx_id == attack_info["attacker_tx"]:
                    attack_role = "attacker"

                if result["status"] == "offline":
                    verdict = "Offline"
                    live_state = "offline"
                    auth_result = "OFFLINE"
                elif result["is_anomaly"]:
                    verdict = "Impersonation detected" if attack_role == "target" else "RF mismatch"
                    live_state = "compromised"
                    auth_result = "DENY"
                else:
                    verdict = "Legitimate"
                    live_state = trust_status["status"] if trust_status["status"] != "trusted" else "trusted"
                    if decision_ratio >= 0.85 and live_state == "trusted":
                        live_state = "suspicious"
                    if live_state == "suspicious":
                        verdict = "Suspicious"
                    elif live_state == "compromised":
                        verdict = "Compromised"
                    auth_result = "GRANT"

                device_states[tx_id] = {
                    "device_name": DEVICE_PROFILES[tx_id]["name"],
                    "status": live_state,
                    "verdict": verdict,
                    "auth_result": auth_result,
                    "threshold": threshold_value,
                    "recon_threshold": result.get("recon_threshold"),
                    "rf_score": current_score,
                    "recon_error": result.get("recon_error"),
                    "maha_distance": profile_distance,
                    "score_ratio": decision_ratio,
                    "decision_ratio": decision_ratio,
                    "recon_ratio": recon_ratio,
                    "profile_threshold": result.get("profile_threshold"),
                    "profile_ratio": profile_ratio,
                    "decision_source": result.get("decision_source"),
                    "match_margin": (threshold_value - current_score) if current_score is not None else None,
                    "trust_score": trust_status["trust_score"] if result["status"] != "offline" else None,
                    "registered_mac": DEVICE_PROFILES[tx_id]["mac"],
                    "last_refresh": refresh_time,
                    "last_activity": refresh_time,
                    "attack_role": attack_role,
                    "is_refreshing": auth_result == "GRANT",
                    "auth_attempts": trust_status["total_checks"] if result["status"] != "offline" else 0,
                }

            render_dashboard(device_states, cycle, attack_info, [], elapsed)

            attack_detected = bool(attack_info) and device_states.get(attack_info["target_tx"], {}).get("auth_result") == "DENY"
            if attack_detected and not report_saved_for_attack:
                save_attack_report(
                    device_states=device_states,
                    attack_info=attack_info,
                    mac_conflicts=[],
                    cycle_num=cycle,
                    elapsed_time=elapsed,
                )
                report_saved_for_attack = True

            time.sleep(MONITOR_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n[Monitoring] Live monitoring stopped from terminal.")

    cleanup_attack_signal()
    return trust_evaluator, attack_start_cycle


def generate_results(pipeline, trust_evaluator, attack_start_cycle):
    """Generate graphs and paper-ready result files from the offline pipeline."""
    set_dashboard_phase(
        "finalizing",
        "Generating visualizations and reports",
        "Saving paper-ready graphs, final summaries, and attack reports to the results folder.",
        extra={"analysis_metrics": build_dashboard_metrics(pipeline.get("paper_metrics"))},
    )

    print("\n" + "=" * 68)
    print("  PHASE 3: VISUALIZATIONS AND REPORTS")
    print("=" * 68)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if attack_start_cycle is None:
        attack_start_cycle = 0

    pipeline = hydrate_visualization_context(pipeline)
    refreshed_metrics = compute_paper_metrics(
        model=pipeline["model"],
        scaled_test=pipeline["scaled_test"],
        adj_tensor=pipeline["adj_tensor"],
        thresholds=pipeline["thresholds"],
        profiler=pipeline["profiler"],
    )
    metrics_path = save_paper_metrics(refreshed_metrics)
    pipeline["paper_metrics"] = refreshed_metrics
    pipeline["training_info"]["metrics_report_path"] = metrics_path

    try:
        from visualization import generate_all_visualizations

        generate_all_visualizations(
            dataset=pipeline["dataset"],
            train_features=pipeline["train_features"],
            scaled_train_features=pipeline["scaled_train"],
            G=pipeline["G"],
            loss_history=pipeline["loss_history"],
            error_stats=pipeline["error_stats"],
            thresholds=pipeline["thresholds"],
            trust_evaluator=trust_evaluator,
            attack_start_cycle=attack_start_cycle,
            paper_metrics=refreshed_metrics,
        )
    except Exception as exc:
        print(f"\n[Visualization] Graph generation skipped due to an environment error: {exc}")

    summary_report_path = save_session_summary_report(pipeline, trust_evaluator, attack_start_cycle)

    print("\n" + "=" * 68)
    print("  ALL DONE")
    print(f"  Graphs saved to : {RESULTS_DIR}")
    print(f"  Metrics report  : {metrics_path}")
    print(f"  Summary report  : {summary_report_path}")
    print(f"  Reports saved to: {os.path.join(RESULTS_DIR, 'reports')}")
    print("=" * 68)

    return metrics_path, summary_report_path


def main():
    """Main entry point for live testing only."""
    try:
        pipeline = load_or_prepare_runtime_pipeline()

        dashboard_url = start_dashboard_server(open_browser=True)
        print("=" * 68)
        print("  LOCALHOST DASHBOARD READY")
        print("=" * 68)
        print(f"\n  Login page     : {get_login_url()}")
        print(f"  Dashboard root : {dashboard_url}")
        print("  Login from the browser with: admin / rf@secure2026")

        set_dashboard_phase(
            "awaiting_login",
            "Waiting for admin login",
            "Offline training is already prepared. Sign in to begin live testing.",
            extra={
                "metrics_report_path": pipeline["training_info"].get("metrics_report_path", PAPER_METRICS_PATH),
                "analysis_metrics": build_dashboard_metrics(pipeline.get("paper_metrics")),
            },
        )

        wait_for_login()
        print("\n[Login] Browser authentication successful. Starting live testing...")
        trust_evaluator, attack_start_cycle = run_monitoring(pipeline)
        metrics_path, summary_report_path = generate_results(pipeline, trust_evaluator, attack_start_cycle)

        set_dashboard_phase(
            "stopped",
            "Live testing complete",
            "Monitoring stopped and the final reports/graphs were written to the results folder.",
            extra={
                "metrics_report_path": metrics_path,
                "summary_report_path": summary_report_path,
                "analysis_metrics": build_dashboard_metrics(pipeline.get("paper_metrics")),
            },
        )
    except RuntimeError as exc:
        print("\n" + "=" * 68)
        print("  OFFLINE TRAINING REQUIRED")
        print("=" * 68)
        print(f"\n{exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        set_dashboard_phase(
            "stopped",
            "Application stopped",
            "The monitoring session was stopped from the terminal.",
        )
        print("\n[System] Shutdown requested. Closing application...")
    finally:
        cleanup_attack_signal()
        stop_dashboard_server()


if __name__ == "__main__":
    main()

"""
Browser-based dashboard for the RF fingerprint monitoring system.
Serves a localhost UI for login, live authentication, and status reporting.
The TSGAE model is trained once from ManySig and then reused by the dashboard.
"""

import os
import secrets
import threading
import webbrowser
from copy import deepcopy
from datetime import datetime

from flask import abort, Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.serving import make_server

from config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    MONITOR_INTERVAL,
    RESULTS_DIR,
    REPORTS_DIR,
    TRANSMITTER_IDS,
)


DASHBOARD_HOST = os.environ.get("RF_DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.environ.get("RF_DASHBOARD_PORT", "5000"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("RF_DASHBOARD_SECRET", "rf-dashboard-local-dev")


GRAPH_TITLES = {
    "00_system_architecture.png": "System Architecture",
    "01_raw_iq_signals.png": "Raw I/Q Signals",
    "02_iq_constellation.png": "I/Q Constellation",
    "03_feature_distributions.png": "Feature Distributions",
    "04_feature_heatmap.png": "Feature Heatmap",
    "05_pca_tsne_clusters.png": "PCA and t-SNE Clusters",
    "06_similarity_graph.png": "Similarity Graph",
    "07_training_loss.png": "Training Loss",
    "08_reconstruction_errors.png": "RF Score Distributions",
    "09_trust_score_evolution.png": "Trust Score Evolution",
    "10_attack_detection_timeline.png": "Attack Detection Timeline",
    "11_detection_summary.png": "Detection Summary",
    "12_claim_confusion_counts.png": "Claim Confusion Counts",
    "13_attack_pair_heatmap.png": "Attack Pair Detection Heatmap",
}


def get_dashboard_url():
    return f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"


def get_login_url():
    return f"{get_dashboard_url()}/login"


def _iso_now():
    return datetime.now().isoformat(timespec="seconds")


def _discover_analysis_graphs():
    """List saved analysis graph images for the dashboard gallery."""
    if not os.path.isdir(RESULTS_DIR):
        return []

    graphs = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        if not entry.lower().endswith(".png"):
            continue

        full_path = os.path.join(RESULTS_DIR, entry)
        graphs.append(
            {
                "filename": entry,
                "title": GRAPH_TITLES.get(entry, os.path.splitext(entry)[0].replace("_", " ").title()),
                "url": f"/results/assets/{entry}",
                "updated_at": datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat(timespec="seconds"),
            }
        )

    return graphs


class DashboardStore:
    """Thread-safe state shared between the monitoring loop and the web UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._login_event = threading.Event()
        self._state = {}
        self.reset()

    def reset(self):
        with self._lock:
            self._state = {
                "phase": "ready",
                "headline": "RF Authentication System Ready",
                "detail": "Model is prepared. Waiting for authentication requests.",
                "last_updated": _iso_now(),
                "current_user": None,
                "login_generation": secrets.token_hex(16),
                "monitor_interval": MONITOR_INTERVAL,
                "dashboard_url": get_dashboard_url(),
                "cycle": 0,
                "elapsed_time": 0,
                "device_states": [],
                "attack_info": None,
                "alerts": [],
                "activity": [],
                "stats": self._empty_stats(),
                "model_info": None,
                "threshold_info": None,
                "report_path": None,
                "metrics_report_path": None,
                "summary_report_path": None,
                "analysis_metrics": None,
            }
        self._login_event.clear()

    def _touch_locked(self):
        self._state["last_updated"] = _iso_now()

    def _empty_stats(self):
        return {
            "total_devices": len(TRANSMITTER_IDS),
            "legitimate": 0,
            "flagged": 0,
            "offline": 0,
            "avg_ratio": 0.0,
            "max_ratio": 0.0,
            "total_auth_attempts": 0,
            "successful_auth": 0,
            "blocked_auth": 0,
        }

    def snapshot(self):
        with self._lock:
            state = deepcopy(self._state)
        state["analysis_graphs"] = _discover_analysis_graphs()
        return state

    def set_logged_in(self, username):
        with self._lock:
            self._state["current_user"] = username
            self._touch_locked()
        self._login_event.set()

    def get_login_generation(self):
        with self._lock:
            return self._state["login_generation"]

    def wait_for_login(self, timeout=None):
        return self._login_event.wait(timeout=timeout)

    def set_model_info(self, model_info, threshold_info):
        with self._lock:
            self._state["model_info"] = model_info
            self._state["threshold_info"] = threshold_info
            self._touch_locked()

    def set_phase(self, phase, headline, detail="", extra=None):
        with self._lock:
            self._state["phase"] = phase
            self._state["headline"] = headline
            self._state["detail"] = detail
            if extra:
                self._state.update(extra)
            self._touch_locked()

    def set_report_path(self, report_path):
        with self._lock:
            self._state["report_path"] = report_path
            self._touch_locked()

    def set_summary_report_path(self, report_path):
        with self._lock:
            self._state["summary_report_path"] = report_path
            self._touch_locked()

    def publish_monitoring_state(self, device_states, cycle_num, attack_info, elapsed_time):
        """Publish authentication results to dashboard."""

        devices = []
        stats = self._empty_stats()
        ratios = []
        refresh_time = _iso_now()

        for tx_id in TRANSMITTER_IDS:
            state = device_states.get(tx_id, {})
            verdict = state.get("verdict", "Offline")
            decision_ratio = self._safe_float(state.get("decision_ratio", state.get("score_ratio")))
            auth_result = state.get("auth_result", "pending")

            if verdict == "Offline":
                stats["offline"] += 1
            elif auth_result == "DENY":
                stats["flagged"] += 1
                stats["blocked_auth"] += 1
            else:
                stats["legitimate"] += 1
                if auth_result == "GRANT":
                    stats["successful_auth"] += 1

            if decision_ratio is not None:
                ratios.append(decision_ratio)

            if state.get("auth_attempts", 0) > 0:
                stats["total_auth_attempts"] += state.get("auth_attempts", 0)

            devices.append(
                {
                    "tx_id": tx_id,
                    "device_name": state.get("device_name", tx_id),
                    "status": state.get("status", "offline"),
                    "verdict": verdict,
                    "auth_result": auth_result,
                    "registered_mac": state.get("registered_mac"),
                    "threshold": self._safe_float(state.get("threshold")),
                    "recon_threshold": self._safe_float(state.get("recon_threshold")),
                    "rf_score": self._safe_float(state.get("rf_score")),
                    "recon_error": self._safe_float(state.get("recon_error")),
                    "maha_distance": self._safe_float(state.get("maha_distance", state.get("profile_distance"))),
                    "score_ratio": decision_ratio,
                    "recon_ratio": self._safe_float(state.get("recon_ratio")),
                    "profile_threshold": self._safe_float(state.get("profile_threshold")),
                    "profile_ratio": self._safe_float(state.get("profile_ratio")),
                    "decision_source": state.get("decision_source"),
                    "match_margin": self._safe_float(state.get("match_margin")),
                    "trust_score": self._safe_float(state.get("trust_score")),
                    "last_refresh": state.get("last_refresh", refresh_time),
                    "last_activity": state.get("last_activity", refresh_time),
                    "attack_role": state.get("attack_role"),
                    "is_refreshing": bool(state.get("is_refreshing")),
                    "auth_attempts": state.get("auth_attempts", 0),
                }
            )

        if ratios:
            stats["avg_ratio"] = float(sum(ratios) / len(ratios))
            stats["max_ratio"] = float(max(ratios))

        alerts = self._build_alerts(attack_info, device_states)
        activity = self._build_activity(device_states, attack_info)

        with self._lock:
            self._state.update(
                {
                    "phase": "monitoring",
                    "headline": "Live RF Authentication",
                    "detail": (
                        f"Cycle {cycle_num}: Authenticating devices against learned RF fingerprints. "
                        "Thresholds are fixed from the one-time training phase."
                    ),
                    "cycle": cycle_num,
                    "elapsed_time": elapsed_time,
                    "device_states": devices,
                    "attack_info": attack_info,
                    "alerts": alerts,
                    "activity": activity,
                    "stats": stats,
                }
            )
            self._touch_locked()

    def _safe_float(self, value):
        if value is None:
            return None
        return float(value)

    def _ratio_text(self, value):
        return f"{value:.2f}x" if value is not None else "N/A"

    def _value_text(self, value, digits=3):
        return f"{value:.{digits}f}" if value is not None else "N/A"

    def _build_alerts(self, attack_info, device_states):
        alerts = []

        if attack_info:
            target_state = device_states.get(attack_info["target_tx"], {})
            decision_ratio = self._safe_float(target_state.get("decision_ratio", target_state.get("score_ratio")))
            recon_ratio = self._safe_float(target_state.get("recon_ratio"))
            profile_ratio = self._safe_float(target_state.get("profile_ratio"))

            if target_state.get("auth_result") == "DENY":
                alerts.append(
                    {
                        "level": "critical",
                        "title": "Impersonation Attack Detected",
                        "message": (
                            f"{attack_info['attacker_tx']} is impersonating {attack_info['target_tx']}. "
                            f"Decision ratio={self._ratio_text(decision_ratio)} "
                            f"(recon={self._ratio_text(recon_ratio)}, profile={self._ratio_text(profile_ratio)}). "
                            "Access DENIED."
                        ),
                    }
                )
            else:
                alerts.append(
                    {
                        "level": "warning",
                        "title": "Attack Scenario Active",
                        "message": (
                            f"{attack_info['attacker_tx']} is trying to impersonate {attack_info['target_tx']}. "
                            f"Decision ratio is {self._ratio_text(decision_ratio)} while the dashboard keeps watching."
                        ),
                    }
                )

        for tx_id in TRANSMITTER_IDS:
            state = device_states.get(tx_id, {})
            verdict = state.get("verdict")
            auth_result = state.get("auth_result")
            ratio = self._safe_float(state.get("decision_ratio", state.get("score_ratio")))
            is_attack_target = bool(attack_info) and tx_id == attack_info.get("target_tx")

            if auth_result == "DENY" and not is_attack_target:
                alerts.append(
                    {
                        "level": "critical",
                        "title": f"{tx_id} Authentication Failed",
                        "message": (
                            f"{verdict} (decision ratio={self._ratio_text(ratio)}). Access DENIED."
                        ),
                    }
                )
            elif state.get("status") in {"suspicious", "compromised"}:
                trust = state.get("trust_score")
                alerts.append(
                    {
                        "level": "warning",
                        "title": f"{tx_id} Low Trust Score",
                        "message": (
                            f"Trust score is {self._value_text(trust, 2)} with decision ratio "
                            f"{self._ratio_text(ratio)}. Monitoring closely."
                        ),
                    }
                )

        if not alerts:
            alerts.append(
                {
                    "level": "healthy",
                    "title": "All Devices Authenticated",
                    "message": "All active devices have valid RF fingerprints.",
                }
            )

        return alerts

    def _build_activity(self, device_states, attack_info):
        now = datetime.now().strftime("%H:%M:%S")
        rows = []

        if attack_info:
            target_state = device_states.get(attack_info["target_tx"], {})
            decision_ratio = self._safe_float(target_state.get("decision_ratio", target_state.get("score_ratio")))
            attack_detected = target_state.get("auth_result") == "DENY"
            rows.append(
                {
                    "timestamp": now,
                    "level": "critical" if attack_detected else "warning",
                    "message": (
                        f"ATTACK: {attack_info['attacker_tx']} -> {attack_info['target_tx']} | "
                        f"decision={self._ratio_text(decision_ratio)} | "
                        f"{'Access DENIED' if attack_detected else 'Monitoring in progress'}"
                    ),
                }
            )

        for tx_id in TRANSMITTER_IDS:
            state = device_states.get(tx_id, {})
            verdict = state.get("verdict", "Offline")
            auth_result = state.get("auth_result")

            if verdict == "Offline":
                continue

            score = self._safe_float(state.get("rf_score"))
            ratio = self._safe_float(state.get("decision_ratio", state.get("score_ratio")))
            recon_ratio = self._safe_float(state.get("recon_ratio"))
            profile_ratio = self._safe_float(state.get("profile_ratio"))
            trust = self._safe_float(state.get("trust_score"))

            score_text = self._value_text(score, 5)
            ratio_text = self._ratio_text(ratio)
            recon_ratio_text = self._ratio_text(recon_ratio)
            profile_ratio_text = self._ratio_text(profile_ratio)
            trust_text = self._value_text(trust, 2)

            if auth_result == "GRANT":
                level = "success"
                message = (
                    f"{tx_id}: GRANTED | RF score={score_text} | decision={ratio_text} "
                    f"(recon={recon_ratio_text}, profile={profile_ratio_text}) | trust={trust_text}"
                )
            elif auth_result == "DENY":
                level = "error"
                message = (
                    f"{tx_id}: DENIED | RF score={score_text} | decision={ratio_text} "
                    f"(recon={recon_ratio_text}, profile={profile_ratio_text}) | trust={trust_text}"
                )
            else:
                level = "info"
                message = (
                    f"{tx_id}: {verdict} | RF score={score_text} | decision={ratio_text} "
                    f"(recon={recon_ratio_text}, profile={profile_ratio_text})"
                )

            rows.append({"timestamp": now, "level": level, "message": message})

        if not rows:
            rows.append({"timestamp": now, "level": "info", "message": "Waiting for authentication requests..."})

        return rows[:20]


class DashboardServerThread(threading.Thread):
    """Background WSGI server for the dashboard."""

    def __init__(self):
        super().__init__(daemon=True)
        self._ready = threading.Event()
        self._server = None
        self.error = None

    def run(self):
        try:
            self._server = make_server(DASHBOARD_HOST, DASHBOARD_PORT, app)
            self._ready.set()
            self._server.serve_forever()
        except Exception as exc:
            self.error = exc
            self._ready.set()

    def wait_until_ready(self, timeout=5):
        self._ready.wait(timeout=timeout)
        if self.error:
            raise self.error
        if not self._ready.is_set():
            raise RuntimeError("Dashboard server did not become ready in time.")

    def stop(self):
        if self._server is not None:
            self._server.shutdown()


store = DashboardStore()
server_thread = None


def start_dashboard_server(open_browser=True):
    """Start the dashboard server and optionally open the login page."""

    global server_thread

    if server_thread and server_thread.is_alive():
        return get_dashboard_url()

    store.reset()
    server_thread = DashboardServerThread()
    server_thread.start()
    server_thread.wait_until_ready()

    set_dashboard_phase(
        "ready",
        "RF Authentication System Ready",
        "Model loaded. Waiting for authentication requests.",
    )

    browser_enabled = open_browser and os.environ.get("RF_OPEN_BROWSER", "1") != "0"
    if browser_enabled:
        threading.Timer(0.75, lambda: webbrowser.open(get_login_url())).start()

    return get_dashboard_url()


def stop_dashboard_server():
    """Stop the dashboard server."""

    global server_thread
    if server_thread and server_thread.is_alive():
        server_thread.stop()
    server_thread = None


def wait_for_login(timeout=None):
    return store.wait_for_login(timeout=timeout)


def set_dashboard_phase(phase, headline, detail="", extra=None):
    store.set_phase(phase, headline, detail, extra=extra)


def set_model_loaded_info(model_info, threshold_info):
    store.set_model_info(model_info, threshold_info)


def set_summary_report_path(report_path):
    store.set_summary_report_path(report_path)


def get_dashboard_snapshot():
    return store.snapshot()


def render_dashboard(device_states, cycle_num, attack_info, _mac_conflicts, elapsed_time):
    """Publish authentication results to dashboard."""

    store.publish_monitoring_state(device_states, cycle_num, attack_info, elapsed_time)


def _is_authenticated():
    return bool(session.get("authenticated")) and session.get("login_generation") == store.get_login_generation()


def _attempt_login(username, password):
    username = str(username or "").strip()
    password = str(password or "")

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["authenticated"] = True
        session["username"] = username
        session["login_generation"] = store.get_login_generation()
        store.set_logged_in(username)
        return True, username

    return False, username


@app.get("/")
def index():
    if _is_authenticated():
        return redirect(url_for("dashboard_page"))
    return redirect(url_for("login_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        session.clear()
        return render_template("login.html", error=None)

    ok, _username = _attempt_login(
        request.form.get("username", ""),
        request.form.get("password", ""),
    )
    if ok:
        return redirect(url_for("dashboard_page"))
    return render_template("login.html", error="Invalid username or password.")


@app.get("/dashboard")
def dashboard_page():
    if not _is_authenticated():
        return redirect(url_for("login_page"))
    return render_template("dashboard.html")


@app.get("/api/status")
def api_status():
    """API endpoint for dashboard status updates."""

    authenticated = _is_authenticated()
    return jsonify(
        {
            "authenticated": authenticated,
            "username": session.get("username") if authenticated else None,
            "state": store.snapshot(),
        }
    )


@app.get("/results/assets/<path:filename>")
def result_asset(filename):
    """Serve generated graph images to authenticated dashboard users."""
    if not _is_authenticated():
        abort(401)
    return send_from_directory(RESULTS_DIR, filename)


@app.post("/api/login")
def api_login():
    payload = request.get_json(silent=True) or {}
    ok, username = _attempt_login(payload.get("username", ""), payload.get("password", ""))
    if ok:
        return jsonify({"ok": True, "username": username})

    return jsonify({"ok": False, "message": "Invalid username or password."}), 401


def save_attack_report(device_states, attack_info, mac_conflicts=None, cycle_num=0, elapsed_time=0):
    """Save an attack detection report."""

    del mac_conflicts
    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORTS_DIR, f"attack_report_{timestamp}.txt")

    lines = [
        "=" * 68,
        "  RF IMPERSONATION ATTACK REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Monitoring Cycle: {cycle_num}",
        f"  System Uptime: {elapsed_time}s",
        "=" * 68,
        "",
    ] 

    if attack_info:
        target_state = device_states.get(attack_info["target_tx"], {})
        decision_ratio = target_state.get("decision_ratio")
        recon_ratio = target_state.get("recon_ratio")
        profile_ratio = target_state.get("profile_ratio")
        profile_distance = target_state.get("maha_distance")
        lines += [
            "ATTACK DETAILS:",
            f"  Attacker RF Source    : {attack_info['attacker_tx']}",
            f"  Claimed Identity      : {attack_info['target_tx']}",
            f"  Attack Started Cycle  : {attack_info.get('attack_cycle', cycle_num)}",
            f"  Detection Method      : Reconstruction + profile-distance verification",
            f"  Decision              : {target_state.get('auth_result', 'PENDING')}",
            f"  Final Verdict         : {target_state.get('verdict', 'Pending analysis')}",
            f"  Decision Ratio        : {f'{decision_ratio:.2f}x' if decision_ratio is not None else 'N/A'}",
            f"  Reconstruction Ratio  : {f'{recon_ratio:.2f}x' if recon_ratio is not None else 'N/A'}",
            f"  Profile Ratio         : {f'{profile_ratio:.2f}x' if profile_ratio is not None else 'N/A'}",
            f"  Profile Distance      : {f'{profile_distance:.4f}' if profile_distance is not None else 'N/A'}",
            "",
        ]

    lines.append("DEVICE AUTHENTICATION STATUS:")
    lines.append(
        f"{'TX':<6} {'Status':<14} {'Auth Result':<12} {'RF Score':<12} "
        f"{'Decision':<10} {'Recon':<10} {'Profile':<10} {'Trust':<8}"
    )
    lines.append("-" * 100)

    for tx_id in TRANSMITTER_IDS:
        state = device_states.get(tx_id, {})
        score = state.get("rf_score")
        ratio = state.get("decision_ratio", state.get("score_ratio"))
        recon_ratio = state.get("recon_ratio")
        profile_ratio = state.get("profile_ratio")
        trust = state.get("trust_score")
        auth_result = state.get("auth_result", "pending")

        lines.append(
            f"{tx_id:<6} {state.get('verdict', state.get('status', 'offline')):<14} "
            f"{auth_result:<12} "
            f"{(f'{score:.6f}' if score is not None else 'N/A'):<12} "
            f"{(f'{ratio:.2f}x' if ratio is not None else 'N/A'):<10} "
            f"{(f'{recon_ratio:.2f}x' if recon_ratio is not None else 'N/A'):<10} "
            f"{(f'{profile_ratio:.2f}x' if profile_ratio is not None else 'N/A'):<10} "
            f"{(f'{trust:.2f}' if trust is not None else 'N/A'):<8}"
        )

    lines += [
        "",
        "CONCLUSION:",
        "  The dashboard compares both the reconstruction score and the",
        "  learned profile distance. The highest ratio becomes the device",
        "  decision ratio, and any value above 1.00x is denied.",
        "",
        "=" * 68,
    ]

    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    store.set_report_path(report_file)
    return report_file

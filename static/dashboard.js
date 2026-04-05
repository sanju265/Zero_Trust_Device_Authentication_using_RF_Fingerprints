const phaseBadge = document.getElementById("phaseBadge");
const cycleValue = document.getElementById("cycleValue");
const updatedValue = document.getElementById("updatedValue");
const heroCopy = document.getElementById("heroCopy");
const headlineValue = document.getElementById("headlineValue");
const detailValue = document.getElementById("detailValue");
const uptimeValue = document.getElementById("uptimeValue");
const refreshValue = document.getElementById("refreshValue");
const sessionPhase = document.getElementById("sessionPhase");
const sessionReport = document.getElementById("sessionReport");

const statTotal = document.getElementById("statTotal");
const statLegitimate = document.getElementById("statLegitimate");
const statFlagged = document.getElementById("statFlagged");
const statBlocked = document.getElementById("statBlocked");
const statSuccess = document.getElementById("statSuccess");
const statOffline = document.getElementById("statOffline");

const deviceTableBody = document.getElementById("deviceTableBody");
const alertsList = document.getElementById("alertsList");
const attackCard = document.getElementById("attackCard");
const reportPath = document.getElementById("reportPath");
const metricsPath = document.getElementById("metricsPath");
const summaryPath = document.getElementById("summaryPath");
const activityList = document.getElementById("activityList");


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}


function formatPhase(phase) {
    return String(phase || "booting").replaceAll("_", " ");
}


function formatTime(value) {
    if (!value) {
        return "--";
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleTimeString();
}


function formatNumber(value, digits = 4) {
    if (value === null || value === undefined) {
        return "--";
    }

    return Number(value).toFixed(digits);
}


function formatPercent(value, digits = 2) {
    if (value === null || value === undefined) {
        return "--";
    }

    return `${(Number(value) * 100).toFixed(digits)}%`;
}


function formatRatio(value) {
    if (value === null || value === undefined) {
        return "--";
    }

    return `${Number(value).toFixed(2)}x`;
}


function ratioClass(ratio) {
    const numeric = Number(ratio || 0);
    if (numeric > 1.0) {
        return "danger";
    }
    if (numeric > 0.85) {
        return "warning";
    }
    return "";
}


function verdictClass(device) {
    if (!device) {
        return "";
    }

    if (device.auth_result === "DENY" || device.verdict === "Impersonation detected" || device.status === "compromised") {
        return "verdict-bad";
    }
    if (device.status === "suspicious" || device.verdict === "Suspicious" || Number(device.score_ratio || 0) > 0.85) {
        return "verdict-warning";
    }
    if (device.verdict === "Offline") {
        return "verdict-offline";
    }
    return "verdict-good";
}


function rowClass(device) {
    if (!device) {
        return "";
    }

    const classes = [];
    if (device.auth_result === "DENY" || device.status === "compromised") {
        classes.push("row-danger");
    } else if (device.status === "suspicious" || device.verdict === "Suspicious" || Number(device.score_ratio || 0) > 0.85) {
        classes.push("row-warning");
    } else if (device.auth_result === "GRANT") {
        classes.push("row-healthy");
    }

    if (device.is_refreshing && device.auth_result === "GRANT") {
        classes.push("row-refreshing");
    }

    return classes.join(" ");
}


function deviceHint(device) {
    if (device.attack_role === "target") {
        return "Claim under attack";
    }
    if (device.attack_role === "attacker") {
        return "Attack RF source";
    }
    if (device.status === "compromised" || device.verdict === "Compromised") {
        return "Compromised trust state";
    }
    if (device.status === "suspicious" || device.verdict === "Suspicious") {
        return "Suspicious RF behavior";
    }
    if (device.verdict === "Offline") {
        return "No live packets in this cycle";
    }
    return "Live RF verification";
}


function renderSummary(state) {
    phaseBadge.textContent = formatPhase(state.phase);
    cycleValue.textContent = String(state.cycle || 0);
    updatedValue.textContent = formatTime(state.last_updated);
    headlineValue.textContent = state.headline || "Starting localhost dashboard";
    detailValue.textContent = state.detail || "Preparing the browser UI.";
    uptimeValue.textContent = `${state.elapsed_time || 0}s`;
    refreshValue.textContent = formatTime(state.last_updated);
    sessionPhase.textContent = formatPhase(state.phase);
    if (state.summary_report_path) {
        sessionReport.textContent = "Summary saved";
    } else if (state.report_path) {
        sessionReport.textContent = "Attack report saved";
    } else if (state.phase === "monitoring") {
        sessionReport.textContent = "Monitoring live";
    } else if (state.metrics_report_path) {
        sessionReport.textContent = "Metrics ready";
    } else {
        sessionReport.textContent = "Awaiting activity";
    }

    heroCopy.textContent = state.phase === "monitoring"
        ? "Live testing is active. The dashboard compares each current RF score and profile-distance check against the saved thresholds, then updates the Bayesian trust score over time."
        : "Offline training is already complete. Sign in and watch only the live testing status here.";
}


function renderStats(state) {
    const stats = state.stats || {};
    statTotal.textContent = String(stats.total_devices || 0);
    statLegitimate.textContent = String(stats.legitimate || 0);
    statFlagged.textContent = String(stats.flagged || 0);
    statBlocked.textContent = String(stats.blocked_auth || 0);
    statSuccess.textContent = String(stats.successful_auth || 0);
    statOffline.textContent = String(stats.offline || 0);
}


function renderDevices(devices) {
    if (!devices || devices.length === 0) {
        deviceTableBody.innerHTML = `
            <tr>
                <td colspan="8" class="empty-state">Waiting for live threshold comparisons.</td>
            </tr>
        `;
        return;
    }

    deviceTableBody.innerHTML = devices.map((device) => {
        const hasRatio = device.score_ratio !== null && device.score_ratio !== undefined;
        const ratio = hasRatio ? Number(device.score_ratio) : 0;
        const fillWidth = hasRatio ? Math.max(8, Math.min(ratio * 65, 100)) : 0;
        const thresholdValue = device.recon_threshold ?? device.threshold;
        const lastRefresh = formatTime(device.last_refresh || device.last_activity);

        return `
            <tr class="${rowClass(device)}">
                <td>
                    <div class="device-name">
                        <strong>${escapeHtml(device.device_name || device.tx_id)}</strong>
                        <small>${escapeHtml(deviceHint(device))}</small>
                    </div>
                </td>
                <td>${escapeHtml(device.tx_id)}</td>
                <td><code>${escapeHtml(device.registered_mac || "--")}</code></td>
                <td>
                    <div class="threshold-stack">
                        <strong>${formatNumber(thresholdValue, 6)}</strong>
                        <small class="score-meta">Profile ${formatNumber(device.profile_threshold, 3)}</small>
                    </div>
                </td>
                <td>
                    <div class="score-stack">
                        <strong>${formatNumber(device.rf_score, 6)}</strong>
                        <small class="score-meta">Recon ${formatNumber(device.recon_error, 6)} | Dist ${formatNumber(device.maha_distance, 3)}</small>
                    </div>
                </td>
                <td>
                    <div class="ratio-stack">
                        <strong class="ratio-value ${ratioClass(device.score_ratio)}">${formatRatio(device.score_ratio)}</strong>
                        <small class="ratio-meta">Recon ${formatRatio(device.recon_ratio)} | Profile ${formatRatio(device.profile_ratio)}</small>
                        <div class="ratio-track">
                            <div class="ratio-fill ${ratioClass(device.score_ratio)}" style="width:${fillWidth}%"></div>
                        </div>
                    </div>
                </td>
                <td><span class="verdict-pill ${verdictClass(device)}">${escapeHtml(device.verdict || "--")}</span></td>
                <td>${escapeHtml(lastRefresh)}</td>
            </tr>
        `;
    }).join("");
}


function renderAlerts(alerts) {
    if (!alerts || alerts.length === 0) {
        alertsList.innerHTML = `
            <div class="message-card healthy">
                <strong>No alerts</strong>
                <p>Live monitoring is active and no RF mismatches are visible.</p>
            </div>
        `;
        return;
    }

    alertsList.innerHTML = alerts.map((alert) => `
        <div class="message-card ${escapeHtml(alert.level || "healthy")}">
            <strong>${escapeHtml(alert.title)}</strong>
            <p>${escapeHtml(alert.message)}</p>
        </div>
    `).join("");
}


function renderAttack(state) {
    const attack = state.attack_info;
    reportPath.textContent = state.report_path || "No attack report saved yet.";
    metricsPath.textContent = state.metrics_report_path || "Metrics report available from offline training.";
    summaryPath.textContent = state.summary_report_path || "The final session report will appear here after monitoring stops.";

    if (!attack) {
        attackCard.className = "attack-card";
        attackCard.innerHTML = "<p>No active RF identity injection detected.</p>";
        return;
    }

    const target = (state.device_states || []).find((device) => device.tx_id === attack.target_tx);
    const detected = Boolean(target) && target.auth_result === "DENY";
    attackCard.className = `attack-card live ${detected ? "critical" : "warning"}`;

    attackCard.innerHTML = `
        <strong>${detected ? "Impersonation detected" : "RF identity injection active"}</strong>
        <p>RF source <code>${escapeHtml(attack.attacker_tx)}</code> is being checked under claim <code>${escapeHtml(attack.target_tx)}</code>.</p>
        <p>Decision ratio: <strong>${formatRatio(target?.score_ratio)}</strong> (recon ${formatRatio(target?.recon_ratio)}, profile ${formatRatio(target?.profile_ratio)}).</p>
        <p>${detected ? "The claim failed verification and access was denied." : "The dashboard is still watching this claim against the saved thresholds."}</p>
        <p>Started at cycle ${escapeHtml(attack.attack_cycle)}</p>
    `;
}


function renderActivity(activity) {
    if (!activity || activity.length === 0) {
        activityList.innerHTML = `
            <div class="activity-row">
                <span>--:--:--</span>
                <p>Waiting for activity.</p>
            </div>
        `;
        return;
    }

    activityList.innerHTML = activity.map((item) => `
        <div class="activity-row ${escapeHtml(item.level || "info")}">
            <span>${escapeHtml(item.timestamp || "--:--:--")}</span>
            <p>${escapeHtml(item.message || "")}</p>
        </div>
    `).join("");
}


function renderState(payload) {
    const state = payload.state || {};
    if (!payload.authenticated) {
        window.location.replace("/login");
        return;
    }

    renderSummary(state);
    renderStats(state);
    renderDevices(state.device_states || []);
    renderAlerts(state.alerts || []);
    renderAttack(state);
    renderActivity(state.activity || []);
}


async function loadStatus() {
    try {
        const response = await fetch("/api/status", { cache: "no-store" });
        const payload = await response.json();
        renderState(payload);
    } catch (_error) {
        headlineValue.textContent = "Unable to reach localhost dashboard";
        detailValue.textContent = "Keep the Python process running and refresh this page.";
    }
}


loadStatus();
window.setInterval(loadStatus, 2000);

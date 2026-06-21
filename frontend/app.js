/* ===================================================
   TrafficAgent — Frontend Application Logic v2
   =================================================== */

const API_BASE = "http://localhost:8000";

// ===== Navigation =====
const sidebar      = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const navItems     = document.querySelectorAll(".nav-item[data-view]");
const views        = document.querySelectorAll(".view");
const pageTitle    = document.getElementById("pageTitle");

const VIEW_TITLES = {
    dashboard: "Dashboard",
    demo:      "Scenario Demo",
    chat:      "Agent Chat",
    alerts:    "Live Alerts",
};

function switchView(viewId) {
    navItems.forEach(n => n.classList.remove("active"));
    views.forEach(v => v.classList.remove("active"));

    const navEl = document.querySelector(`[data-view="${viewId}"]`);
    if (navEl) navEl.classList.add("active");

    const viewEl = document.getElementById(`${viewId}View`);
    if (viewEl) viewEl.classList.add("active");

    pageTitle.textContent = VIEW_TITLES[viewId] || "Dashboard";
    sidebar.classList.remove("open");
}

navItems.forEach(item => {
    item.addEventListener("click", e => {
        e.preventDefault();
        switchView(item.dataset.view);
    });
});

sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));

document.getElementById("openFullChat")?.addEventListener("click", e => {
    e.preventDefault();
    switchView("chat");
});

// ===== Health Check =====
const serverStatus = document.getElementById("serverStatus");
const apiBadge     = document.getElementById("apiBadge");

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health/`);
        if (res.ok) {
            serverStatus.innerHTML = `<div class="status-dot online"></div><span>Server online</span>`;
            apiBadge.classList.remove("offline");
            apiBadge.querySelector("span").textContent = "API";
            return true;
        }
    } catch { /* fall through */ }
    serverStatus.innerHTML = `<div class="status-dot offline"></div><span>Server offline</span>`;
    apiBadge.classList.add("offline");
    apiBadge.querySelector("span").textContent = "OFFLINE";
    return false;
}

checkHealth();
setInterval(checkHealth, 15_000);

// ===== Database Stats =====
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats/`);
        if (!res.ok) return;
        const d = await res.json();
        document.getElementById("statDuckdb").textContent =
            d.duckdb_rows != null ? d.duckdb_rows.toLocaleString() : "—";
        document.getElementById("statMilvus").textContent =
            d.milvus_events != null ? d.milvus_events.toLocaleString() : "—";
        document.getElementById("statKuzu").textContent =
            d.kuzu_edges != null ? d.kuzu_edges.toLocaleString() : "—";
        document.getElementById("statProfiles").textContent =
            d.milvus_profiles != null ? d.milvus_profiles.toLocaleString() : "—";
    } catch { /* stats are non-critical */ }
}

loadStats();
setInterval(loadStats, 30_000);

// ===== Model Management =====
const modelFileInput     = document.getElementById("modelFileInput");
const activeModelName    = document.getElementById("activeModelName");
const modelStatusChip    = document.getElementById("modelStatusChip");
const modelProgress      = document.getElementById("modelProgress");
const modelProgressFill  = document.getElementById("modelProgressFill");
const modelProgressLabel = document.getElementById("modelProgressLabel");
const modelUploadHint    = document.getElementById("modelUploadHint");

async function loadActiveModel() {
    try {
        const res = await fetch(`${API_BASE}/models/`);
        if (!res.ok) return;
        const data = await res.json();
        activeModelName.textContent = data.active;
        modelStatusChip.className   = "chip done";
        modelStatusChip.textContent = "Active";
    } catch {
        activeModelName.textContent = "yolov8n.pt (default)";
    }
}

modelFileInput.addEventListener("change", async () => {
    const file = modelFileInput.files[0];
    if (!file) return;
    if (!file.name.endsWith(".pt")) {
        modelUploadHint.textContent = "Only .pt files are accepted.";
        return;
    }

    modelStatusChip.className   = "chip processing";
    modelStatusChip.textContent = "Uploading";
    modelUploadHint.textContent = file.name;
    modelProgress.style.display = "block";
    modelProgressFill.style.width = "0%";
    modelProgressLabel.textContent = "Uploading...";

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener("progress", e => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            modelProgressFill.style.width = `${pct}%`;
            modelProgressLabel.textContent = `${pct}%`;
        }
    });
    xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            modelProgressFill.style.width = "100%";
            modelProgressLabel.textContent = "Upload complete";
            modelStatusChip.className   = "chip done";
            modelStatusChip.textContent = "Active";
            activeModelName.textContent = file.name;
            setTimeout(() => { modelProgress.style.display = "none"; }, 2000);
        } else {
            modelProgressLabel.textContent = "Upload failed";
            modelStatusChip.className   = "chip error";
            modelStatusChip.textContent = "Failed";
        }
        modelFileInput.value = "";
    });
    xhr.addEventListener("error", () => {
        modelProgressLabel.textContent = "Server error";
        modelStatusChip.className   = "chip error";
        modelStatusChip.textContent = "Failed";
        modelFileInput.value = "";
    });

    xhr.open("POST", `${API_BASE}/upload_model/`);
    xhr.send(formData);
});

loadActiveModel();

// ===== Video Upload & Video Select =====
const videoFileInput    = document.getElementById("videoFileInput");
const videoUploadStatus = document.getElementById("videoUploadStatus");
const videoUploadLabel  = document.getElementById("videoUploadLabel");
const physicsVideoSelect = document.getElementById("physicsVideoSelect");
const runPhysicsBtn      = document.getElementById("runPhysicsBtn");

function selectVideo(value) {
    physicsVideoSelect.value = value;
    runPhysicsBtn.disabled   = !value;
    if (value) localStorage.setItem("trafficagent_video", value);
}

videoFileInput.addEventListener("change", () => {
    const file = videoFileInput.files[0];
    if (!file) return;

    videoUploadStatus.style.display = "block";
    videoUploadLabel.textContent    = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener("progress", e => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            videoUploadLabel.textContent = `Uploading ${file.name}… ${pct}%`;
        }
    });
    xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            videoUploadLabel.textContent = `${file.name} saved`;
            loadVideoList().then(() => selectVideo(file.name));
            setTimeout(() => { videoUploadStatus.style.display = "none"; }, 3000);
        } else {
            videoUploadLabel.textContent = "Upload failed";
        }
        videoFileInput.value = "";
    });
    xhr.addEventListener("error", () => {
        videoUploadLabel.textContent = "Server error";
        videoFileInput.value = "";
    });
    xhr.open("POST", `${API_BASE}/upload_video/`);
    xhr.send(formData);
});

async function loadVideoList() {
    try {
        const res = await fetch(`${API_BASE}/calibrate/videos`);
        if (!res.ok) return;
        const data = await res.json();
        physicsVideoSelect.innerHTML = data.videos.length
            ? `<option value="">Select a video...</option>` +
              data.videos.map(v => `<option value="${v}">${v}</option>`).join("")
            : `<option value="">No videos found</option>`;

        const saved = localStorage.getItem("trafficagent_video");
        if (saved) {
            physicsVideoSelect.value = saved;
            if (physicsVideoSelect.value === saved) {
                runPhysicsBtn.disabled = false;
            } else {
                localStorage.removeItem("trafficagent_video");
                runPhysicsBtn.disabled = !physicsVideoSelect.value;
            }
        } else {
            runPhysicsBtn.disabled = !physicsVideoSelect.value;
        }
    } catch {
        physicsVideoSelect.innerHTML = `<option value="">Could not load videos</option>`;
    }
}

physicsVideoSelect.addEventListener("change", () => selectVideo(physicsVideoSelect.value));

// ===== Physics Pipeline =====
const physicsStatusChip    = document.getElementById("physicsStatusChip");
const physicsProgress      = document.getElementById("physicsProgress");
const physicsProgressFill  = document.getElementById("physicsProgressFill");
const physicsProgressLabel = document.getElementById("physicsProgressLabel");

let physicsJobId  = null;
let physicsPoller = null;

runPhysicsBtn.addEventListener("click", async () => {
    const videoPath = physicsVideoSelect.value;
    if (!videoPath) return;

    const runPhysics = document.getElementById("chkPhysics").checked;
    const runVlm     = document.getElementById("chkVlm").checked;
    if (!runPhysics && !runVlm) {
        alert("Select at least one of Physics or VLM.");
        return;
    }

    runPhysicsBtn.disabled         = true;
    physicsVideoSelect.disabled    = true;
    setPhysicsChip("processing", "Running");
    physicsProgress.style.display  = "block";
    physicsProgressFill.style.width = "0%";
    physicsProgressLabel.textContent = "Starting...";

    try {
        const res = await fetch(`${API_BASE}/run_physics/`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ video_path: videoPath, run_physics: runPhysics, run_vlm: runVlm }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => null);
            setPhysicsChip("error", "Failed");
            physicsProgressLabel.textContent = err?.detail || "Error starting job";
            physicsVideoSelect.disabled      = false;
            runPhysicsBtn.disabled           = false;
            return;
        }
        const job = await res.json();
        physicsJobId  = job.job_id;
        physicsPoller = setInterval(pollPhysicsJob, 2000);
    } catch {
        setPhysicsChip("error", "Server error");
        physicsVideoSelect.disabled = false;
        runPhysicsBtn.disabled      = false;
    }
});

async function pollPhysicsJob() {
    if (!physicsJobId) return;
    try {
        const res = await fetch(`${API_BASE}/job/${physicsJobId}`);
        if (!res.ok) return;
        const job = await res.json();

        if (job.frames_processed != null && job.total_frames) {
            const pct = Math.round((job.frames_processed / job.total_frames) * 100);
            physicsProgressFill.style.width  = `${pct}%`;
            physicsProgressLabel.textContent = `${job.frames_processed} / ${job.total_frames} frames (${pct}%)`;
        }

        if (job.status === "done") {
            clearInterval(physicsPoller);
            physicsPoller = null;
            physicsProgressFill.style.width  = "100%";
            physicsProgressLabel.textContent = "Complete — databases populated";
            setPhysicsChip("done", "Done");
            physicsVideoSelect.disabled = false;
            runPhysicsBtn.disabled      = false;
            loadStats();
        } else if (job.status === "failed") {
            clearInterval(physicsPoller);
            physicsPoller = null;
            setPhysicsChip("error", "Failed");
            physicsProgressLabel.textContent = job.error || "Pipeline error";
            physicsVideoSelect.disabled      = false;
            runPhysicsBtn.disabled           = false;
        }
    } catch { /* keep polling */ }
}

function setPhysicsChip(type, text) {
    physicsStatusChip.className   = `chip ${type}`;
    physicsStatusChip.textContent = text;
}

loadVideoList();

// Zone editor nav — pass selected video as URL param
const zoneEditorNav = document.getElementById("zoneEditorNav");
if (zoneEditorNav) {
    zoneEditorNav.addEventListener("click", e => {
        e.preventDefault();
        const video = physicsVideoSelect.value;
        window.location.href = video ? `/zone-ui?video=${encodeURIComponent(video)}` : "/zone-ui";
    });
}

// ===== Shared Chat API call =====
async function queryAgent(message) {
    const res = await fetch(`${API_BASE}/chat/`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query: message }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `Server error (${res.status})`);
    }
    return res.json();   // { query, summary, reasoning_steps }
}

// ===== Quick Chat (Dashboard) =====
const quickChatForm = document.getElementById("quickChatForm");
const quickInput    = document.getElementById("quickInput");
const quickSendBtn  = document.getElementById("quickSendBtn");
const quickResponse = document.getElementById("quickResponse");

quickChatForm.addEventListener("submit", async e => {
    e.preventDefault();
    const msg = quickInput.value.trim();
    if (!msg) return;

    quickSendBtn.disabled    = true;
    quickInput.disabled      = true;
    quickResponse.textContent = "Thinking...";
    quickResponse.classList.add("visible");

    try {
        const data = await queryAgent(msg);
        quickResponse.innerHTML = renderMarkdown(data.summary);
    } catch (err) {
        quickResponse.textContent = `Error: ${err.message}`;
    }

    quickSendBtn.disabled = false;
    quickInput.disabled   = false;
    quickInput.value      = "";
});

// ===== Full Chat View =====
const chatMessages = document.getElementById("chatMessages");
const chatForm     = document.getElementById("chatForm");
const chatInput    = document.getElementById("chatInput");
const chatSendBtn  = document.getElementById("chatSendBtn");

let isWaiting = false;

chatForm.addEventListener("submit", e => {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg || isWaiting) return;
    sendChatMessage(msg);
    chatInput.value = "";
});

document.querySelectorAll(".example-btn").forEach(btn => {
    btn.addEventListener("click", () => sendChatMessage(btn.dataset.query));
});

async function sendChatMessage(message) {
    const welcome = chatMessages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    addChatBubble("user", message);

    const typingEl = showTyping();
    isWaiting      = true;
    chatSendBtn.disabled = true;

    try {
        const data = await queryAgent(message);

        typingEl.remove();
        addChatBubble("agent", data.summary, data.reasoning_steps || [], data);
    } catch (err) {
        typingEl.remove();
        addChatBubble("agent", `Error: ${err.message}`);
    }

    isWaiting            = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
}

function addChatBubble(role, text, steps = [], data = {}) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;

    const label = document.createElement("div");
    label.className   = "bubble-label";
    label.textContent = role === "user" ? "You" : "Agent";
    bubble.appendChild(label);

    if (role === "agent" && steps.length > 0) {
        const trace = document.createElement("div");
        trace.className = "bubble-trace";
        trace.innerHTML = steps.map(s =>
            `<span class="bubble-step">${s.step}. ${s.tool}</span>`
        ).join(" → ");
        bubble.appendChild(trace);
    }

    const content = document.createElement("div");
    content.className = "bubble-text";
    if (role === "agent") {
        content.innerHTML = renderMarkdown(text);
    } else {
        content.textContent = text;
    }
    bubble.appendChild(content);

    if (role === "agent") {
        if (data.contradictions && data.contradictions.length > 0) {
            const note = document.createElement("div");
            note.className = "contradiction-note";
            note.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <span>${escapeHtml(data.contradictions[0])}</span>`;
            bubble.appendChild(note);
        }
        const meta = document.createElement("div");
        meta.innerHTML = buildMetaBar(data);
        bubble.appendChild(meta);
    }

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    const el = document.createElement("div");
    el.className = "typing-indicator";
    el.innerHTML = "<span></span><span></span><span></span>";
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
}

// ===== Scenario Demo View =====
const demoQueryDisplay = document.getElementById("demoQueryDisplay");
const demoStatusChip   = document.getElementById("demoStatusChip");
const toolTrace        = document.getElementById("toolTrace");
const toolSteps        = document.getElementById("toolSteps");
const demoAnswerBox    = document.getElementById("demoAnswerBox");

let activeDemoBtn = null;

document.querySelectorAll(".demo-query-btn").forEach(btn => {
    btn.addEventListener("click", () => runDemoQuery(btn.dataset.query, btn));
});

async function runDemoQuery(query, btnEl) {
    if (activeDemoBtn) activeDemoBtn.classList.remove("active");
    if (btnEl) { btnEl.classList.add("active"); activeDemoBtn = btnEl; }

    demoQueryDisplay.textContent = query;
    demoStatusChip.className     = "chip processing";
    demoStatusChip.textContent   = "Running";
    toolTrace.style.display      = "none";
    toolSteps.innerHTML          = "";
    demoAnswerBox.innerHTML      = `<div class="demo-answer-placeholder"><p>Agent is reasoning...</p></div>`;

    try {
        const data = await queryAgent(query);
        const steps = data.reasoning_steps || [];

        if (steps.length > 0) {
            toolTrace.style.display = "block";
            toolSteps.innerHTML     = steps.map(s => renderToolStep(s)).join("");
        }

        let html = `<div class="demo-answer-text">${renderMarkdown(data.summary)}</div>`;

        if (data.contradictions && data.contradictions.length > 0) {
            html += `<div class="contradiction-note">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <span>${escapeHtml(data.contradictions[0])}</span>
            </div>`;
        }

        html += buildMetaBar(data);
        demoAnswerBox.innerHTML    = html;
        demoStatusChip.className   = "chip done";
        demoStatusChip.textContent = `Done · ${steps.length} tool${steps.length !== 1 ? "s" : ""}`;
    } catch (err) {
        demoAnswerBox.innerHTML    = `<div class="demo-answer-text" style="color:var(--error)">Error: ${escapeHtml(err.message)}</div>`;
        demoStatusChip.className   = "chip error";
        demoStatusChip.textContent = "Failed";
    }
}

function buildMetaBar(data) {
    const parts = [];
    if (data.route)   parts.push(escapeHtml(data.route.replace("_", " ")));
    if (data.reasoning_steps?.length) parts.push(`${data.reasoning_steps.length} tools`);
    if (data.session_id) parts.push(`session ${escapeHtml(data.session_id.slice(0, 8))}`);
    if (!parts.length) return "";
    return `<div class="response-meta">${parts.join(" &nbsp;·&nbsp; ")}</div>`;
}

function renderToolStep(step) {
    const excerpt = step.output_excerpt
        ? `<span class="step-excerpt"> — ${escapeHtml(step.output_excerpt.slice(0, 80))}</span>`
        : "";
    return `
        <div class="tool-step">
            <span class="step-num">${step.step}</span>
            <span class="step-tool">${escapeHtml(step.tool)}</span>
            ${excerpt}
        </div>`;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

/**
 * Minimal markdown → HTML renderer.
 * Handles: headings, bold, inline code, code blocks, horizontal rules,
 * bullet lists, numbered lists, and blockquotes.
 * No external dependency — safe against XSS via escapeHtml on raw text.
 */
function renderMarkdown(raw) {
    const lines = String(raw).split("\n");
    const out   = [];
    let inList = null;   // "ul" | "ol" | null
    let inCode = false;
    let codeBuf = [];

    function closeList() {
        if (inList) { out.push(`</${inList}>`); inList = null; }
    }

    function flushCode() {
        out.push(`<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
        codeBuf = [];
        inCode  = false;
    }

    function inlineFormat(text) {
        return escapeHtml(text)
            // **bold**
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            // *italic*
            .replace(/\*(.+?)\*/g, "<em>$1</em>")
            // `code`
            .replace(/`([^`]+)`/g, "<code>$1</code>");
    }

    for (const raw_line of lines) {
        const line = raw_line;

        // Fenced code block toggle
        if (line.startsWith("```")) {
            if (inCode) { flushCode(); continue; }
            closeList();
            inCode = true;
            continue;
        }
        if (inCode) { codeBuf.push(line); continue; }

        // Horizontal rule
        if (/^---+$/.test(line.trim())) {
            closeList();
            out.push("<hr>");
            continue;
        }

        // Headings
        const h = line.match(/^(#{1,4})\s+(.*)/);
        if (h) {
            closeList();
            const level = Math.min(h[1].length + 2, 6); // h3–h6 to stay below page h1/h2
            out.push(`<h${level}>${inlineFormat(h[2])}</h${level}>`);
            continue;
        }

        // Blockquote
        if (line.startsWith("> ")) {
            closeList();
            out.push(`<blockquote>${inlineFormat(line.slice(2))}</blockquote>`);
            continue;
        }

        // Unordered list
        const ul = line.match(/^[-*]\s+(.*)/);
        if (ul) {
            if (inList !== "ul") { closeList(); out.push("<ul>"); inList = "ul"; }
            out.push(`<li>${inlineFormat(ul[1])}</li>`);
            continue;
        }

        // Ordered list
        const ol = line.match(/^\d+\.\s+(.*)/);
        if (ol) {
            if (inList !== "ol") { closeList(); out.push("<ol>"); inList = "ol"; }
            out.push(`<li>${inlineFormat(ol[1])}</li>`);
            continue;
        }

        // Blank line — close list, paragraph break
        if (line.trim() === "") {
            closeList();
            out.push("<br>");
            continue;
        }

        // Plain paragraph
        closeList();
        out.push(`<p>${inlineFormat(line)}</p>`);
    }

    closeList();
    if (inCode) flushCode();

    return out.join("\n");
}

// ===== Live Alerts (SSE) =====
const alertsList    = document.getElementById("alertsList");
const connectSseBtn = document.getElementById("connectSseBtn");
const clearAlertsBtn = document.getElementById("clearAlertsBtn");
const sseDot        = document.getElementById("sseDot");
const sseLabel      = document.getElementById("sseLabel");
const alertBadge    = document.getElementById("alertBadge");

let sseSource       = null;
let unreadAlerts    = 0;

connectSseBtn.addEventListener("click", () => {
    if (sseSource) {
        sseSource.close();
        sseSource = null;
        setSseState(false);
        connectSseBtn.textContent = "Connect";
    } else {
        sseSource = new EventSource(`${API_BASE}/alerts/stream`);
        sseSource.onopen    = () => { setSseState(true); connectSseBtn.textContent = "Disconnect"; };
        sseSource.onerror   = () => { setSseState(false); };
        sseSource.onmessage = e => handleAlert(JSON.parse(e.data));
    }
});

clearAlertsBtn.addEventListener("click", () => {
    alertsList.innerHTML = `
        <div class="alerts-empty">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
            </svg>
            <p>Cleared.</p>
        </div>`;
    unreadAlerts = 0;
    alertBadge.style.display = "none";
});

function setSseState(connected) {
    sseDot.className = `sse-dot ${connected ? "connected" : "disconnected"}`;
    sseLabel.textContent = connected ? "Connected" : "Disconnected";
}

function handleAlert(alert) {
    // Remove empty state
    const empty = alertsList.querySelector(".alerts-empty");
    if (empty) empty.remove();

    const severity = (alert.severity || "info").toLowerCase();
    const card = document.createElement("div");
    card.className = `alert-card severity-${severity}`;

    const ts = alert.timestamp != null
        ? new Date(alert.timestamp * 1000).toISOString().substr(11, 8)
        : "–";

    card.innerHTML = `
        <div class="alert-severity-dot"></div>
        <div class="alert-body">
            <div class="alert-type">${escapeHtml(alert.alert_type || "ALERT")}</div>
            <div class="alert-message">${escapeHtml(alert.message || "")}</div>
            <div class="alert-meta">
                Vehicle #${alert.track_id ?? "?"} &nbsp;·&nbsp;
                t=${ts} &nbsp;·&nbsp;
                frame ${alert.frame_id ?? "?"}
            </div>
        </div>`;

    alertsList.prepend(card);

    // Update badge if not on alerts view
    const alertsView = document.getElementById("alertsView");
    if (!alertsView.classList.contains("active")) {
        unreadAlerts++;
        alertBadge.textContent   = unreadAlerts > 99 ? "99+" : unreadAlerts;
        alertBadge.style.display = "flex";
    }
}

// When switching to alerts view, clear the badge
navItems.forEach(item => {
    if (item.dataset.view === "alerts") {
        item.addEventListener("click", () => {
            unreadAlerts = 0;
            alertBadge.style.display = "none";
        });
    }
});

// Load alert history on startup
(async () => {
    try {
        const res = await fetch(`${API_BASE}/alerts/history?limit=50`);
        if (!res.ok) return;
        const data = await res.json();
        data.alerts.reverse().forEach(a => handleAlert(a));
    } catch { /* non-critical */ }
})();

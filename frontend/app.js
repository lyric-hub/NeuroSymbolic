/* ===================================================
   TrafficAgent — Frontend Application Logic
   =================================================== */

const API_BASE = "http://localhost:8000";

// ===== DOM Elements =====
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");
const pageTitle = document.getElementById("pageTitle");

// Chat
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");

// Quick chat
const quickChatForm = document.getElementById("quickChatForm");
const quickInput = document.getElementById("quickInput");
const quickSendBtn = document.getElementById("quickSendBtn");
const quickResponse = document.getElementById("quickResponse");
const openFullChat = document.getElementById("openFullChat");

// Status
const serverStatus = document.getElementById("serverStatus");
const apiBadge = document.getElementById("apiBadge");

// ===== Navigation =====
navItems.forEach((item) => {
    item.addEventListener("click", (e) => {
        // Only intercept items that switch an in-page view (have data-view).
        // Items with a real href (like /calibrate-ui) navigate normally.
        if (!item.dataset.view) return;
        e.preventDefault();
        switchView(item.dataset.view);
    });
});

openFullChat.addEventListener("click", (e) => {
    e.preventDefault();
    switchView("chat");
});

function switchView(viewId) {
    navItems.forEach((n) => n.classList.remove("active"));
    views.forEach((v) => v.classList.remove("active"));

    document.querySelector(`[data-view="${viewId}"]`).classList.add("active");
    document.getElementById(`${viewId}View`).classList.add("active");

    const titles = { dashboard: "Dashboard", chat: "Agent Chat" };
    pageTitle.textContent = titles[viewId] || "Dashboard";

    // Close mobile sidebar
    sidebar.classList.remove("open");
}

// Mobile sidebar toggle
sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});

// ===== Health Check =====
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health/`);
        if (res.ok) {
            serverStatus.innerHTML = `<div class="status-dot online"></div><span>Server online</span>`;
            apiBadge.classList.remove("offline");
            apiBadge.querySelector("span").textContent = "API";
            return true;
        }
    } catch {
        // fall through
    }
    serverStatus.innerHTML = `<div class="status-dot offline"></div><span>Server offline</span>`;
    apiBadge.classList.add("offline");
    apiBadge.querySelector("span").textContent = "OFFLINE";
    return false;
}

// Check on load and periodically
checkHealth();
setInterval(checkHealth, 15000);

// ===== Inline Video Upload (inside Run Pipeline card) =====
const videoFileInput    = document.getElementById("videoFileInput");
const videoUploadStatus = document.getElementById("videoUploadStatus");
const videoUploadLabel  = document.getElementById("videoUploadLabel");

videoFileInput.addEventListener("change", () => {
    const file = videoFileInput.files[0];
    if (!file) return;

    videoUploadStatus.style.display = "block";
    videoUploadLabel.textContent = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            videoUploadLabel.textContent = `Uploading ${file.name}… ${pct}%`;
        }
    });
    xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            videoUploadLabel.textContent = `${file.name} saved`;
            loadVideoList().then(() => {
                // Auto-select the just-uploaded file
                physicsVideoSelect.value = file.name;
                runPhysicsBtn.disabled = false;
            });
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

// ===== Chat =====
let isWaiting = false;

chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg || isWaiting) return;
    sendChatMessage(msg);
    chatInput.value = "";
});

// Example query buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        const query = btn.dataset.query;
        sendChatMessage(query);
    });
});

async function sendChatMessage(message) {
    // Hide welcome screen on first message
    const welcome = chatMessages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    // Add user bubble
    addChatBubble("user", message);

    // Show typing indicator
    const typingEl = showTyping();
    isWaiting = true;
    chatSendBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/chat/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: message }),
        });

        typingEl.remove();

        if (res.ok) {
            const data = await res.json();
            addChatBubble("agent", data.summary);
        } else {
            const errData = await res.json().catch(() => null);
            const errMsg = errData?.detail || `Server error (${res.status})`;
            addChatBubble("agent", `⚠️ ${errMsg}`);
        }
    } catch {
        typingEl.remove();
        addChatBubble("agent", "⚠️ Could not reach the server. Is the API running?");
    }

    isWaiting = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
}

function addChatBubble(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;

    const label = document.createElement("div");
    label.className = "bubble-label";
    label.textContent = role === "user" ? "You" : "Agent";

    const content = document.createElement("div");
    content.textContent = text;

    bubble.appendChild(label);
    bubble.appendChild(content);
    chatMessages.appendChild(bubble);

    // Scroll to bottom
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
        modelStatusChip.className = "chip done";
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

    modelStatusChip.className = "chip processing";
    modelStatusChip.textContent = "Uploading";
    modelUploadHint.textContent = file.name;
    modelProgress.style.display = "block";
    modelProgressFill.style.width = "0%";
    modelProgressLabel.textContent = "Uploading...";

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener("progress", (e) => {
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
            modelStatusChip.className = "chip done";
            modelStatusChip.textContent = "Active";
            activeModelName.textContent = file.name;
            setTimeout(() => { modelProgress.style.display = "none"; }, 2000);
        } else {
            modelProgressLabel.textContent = "Upload failed";
            modelStatusChip.className = "chip error";
            modelStatusChip.textContent = "Failed";
        }
        modelFileInput.value = "";
    });
    xhr.addEventListener("error", () => {
        modelProgressLabel.textContent = "Server error";
        modelStatusChip.className = "chip error";
        modelStatusChip.textContent = "Failed";
        modelFileInput.value = "";
    });

    xhr.open("POST", `${API_BASE}/upload_model/`);
    xhr.send(formData);
});

loadActiveModel();

// ===== Physics Processing =====
const physicsVideoSelect  = document.getElementById("physicsVideoSelect");
const runPhysicsBtn       = document.getElementById("runPhysicsBtn");
const physicsStatusChip   = document.getElementById("physicsStatusChip");
const physicsProgress     = document.getElementById("physicsProgress");
const physicsProgressFill = document.getElementById("physicsProgressFill");
const physicsProgressLabel = document.getElementById("physicsProgressLabel");

let physicsJobId     = null;
let physicsPoller    = null;

async function loadVideoList() {
    try {
        const res = await fetch(`${API_BASE}/calibrate/videos`);
        if (!res.ok) return;
        const data = await res.json();
        physicsVideoSelect.innerHTML = data.videos.length
            ? `<option value="">Select a video...</option>` +
              data.videos.map(v => `<option value="${v}">${v}</option>`).join("")
            : `<option value="">No videos found</option>`;
        runPhysicsBtn.disabled = data.videos.length === 0;
    } catch {
        physicsVideoSelect.innerHTML = `<option value="">Could not load videos</option>`;
    }
}

physicsVideoSelect.addEventListener("change", () => {
    runPhysicsBtn.disabled = !physicsVideoSelect.value;
});

runPhysicsBtn.addEventListener("click", async () => {
    const videoPath = physicsVideoSelect.value;
    if (!videoPath) return;

    const runPhysics = document.getElementById("chkPhysics").checked;
    const runVlm    = document.getElementById("chkVlm").checked;
    if (!runPhysics && !runVlm) {
        alert("Select at least one of Physics or VLM.");
        return;
    }

    runPhysicsBtn.disabled = true;
    physicsVideoSelect.disabled = true;
    setPhysicsChip("processing", "Running");
    physicsProgress.style.display = "block";
    physicsProgressFill.style.width = "0%";
    physicsProgressLabel.textContent = "Starting...";

    try {
        const res = await fetch(`${API_BASE}/run_physics/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video_path: videoPath, run_physics: runPhysics, run_vlm: runVlm }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => null);
            setPhysicsChip("error", "Failed");
            physicsProgressLabel.textContent = err?.detail || "Error starting job";
            physicsVideoSelect.disabled = false;
            runPhysicsBtn.disabled = false;
            return;
        }
        const job = await res.json();
        physicsJobId = job.job_id;
        physicsPoller = setInterval(pollPhysicsJob, 2000);
    } catch {
        setPhysicsChip("error", "Server error");
        physicsVideoSelect.disabled = false;
        runPhysicsBtn.disabled = false;
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
            physicsProgressFill.style.width = `${pct}%`;
            physicsProgressLabel.textContent = `${job.frames_processed} / ${job.total_frames} frames (${pct}%)`;
        }

        if (job.status === "done") {
            clearInterval(physicsPoller);
            physicsPoller = null;
            physicsProgressFill.style.width = "100%";
            physicsProgressLabel.textContent = "Complete — databases populated";
            setPhysicsChip("done", "Done");
            physicsVideoSelect.disabled = false;
            runPhysicsBtn.disabled = false;
        } else if (job.status === "failed") {
            clearInterval(physicsPoller);
            physicsPoller = null;
            setPhysicsChip("error", "Failed");
            physicsProgressLabel.textContent = job.error || "Pipeline error";
            physicsVideoSelect.disabled = false;
            runPhysicsBtn.disabled = false;
        }
    } catch { /* network hiccup — keep polling */ }
}

function setPhysicsChip(type, text) {
    physicsStatusChip.className = `chip ${type}`;
    physicsStatusChip.textContent = text;
}

loadVideoList();

// ===== Quick Chat (Dashboard) =====
quickChatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const msg = quickInput.value.trim();
    if (!msg) return;

    quickSendBtn.disabled = true;
    quickInput.disabled = true;
    quickResponse.textContent = "Thinking...";
    quickResponse.classList.add("visible");

    try {
        const res = await fetch(`${API_BASE}/chat/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: msg }),
        });

        if (res.ok) {
            const data = await res.json();
            quickResponse.textContent = data.summary;
        } else {
            quickResponse.textContent = "⚠️ Failed to get response from agent.";
        }
    } catch {
        quickResponse.textContent = "⚠️ Server offline. Start the API with: uvicorn api:app --reload";
    }

    quickSendBtn.disabled = false;
    quickInput.disabled = false;
    quickInput.value = "";
});

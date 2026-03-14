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

// Upload
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadProgress = document.getElementById("uploadProgress");
const progressCircle = document.getElementById("progressCircle");
const progressText = document.getElementById("progressText");
const progressLabel = document.getElementById("progressLabel");
const uploadStatusChip = document.getElementById("uploadStatusChip");
const uploadInfo = document.getElementById("uploadInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

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

// ===== File Upload =====
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) handleFileUpload(file);
});

async function handleFileUpload(file) {
    // Validate file type
    const validTypes = [".mp4", ".avi", ".mov"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!validTypes.includes(ext)) {
        showUploadStatus("error", "Unsupported file type");
        return;
    }

    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    uploadInfo.style.display = "flex";

    // Show progress
    uploadProgress.classList.add("active");
    showUploadStatus("processing", "Uploading");

    // Simulate upload progress (XHR for real progress)
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            updateProgress(pct);
        }
    });

    xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            updateProgress(100);
            progressLabel.textContent = "Processing in background...";
            showUploadStatus("processing", "Processing");

            setTimeout(() => {
                uploadProgress.classList.remove("active");
                showUploadStatus("done", "Complete");
            }, 2000);
        } else {
            uploadProgress.classList.remove("active");
            showUploadStatus("error", "Upload failed");
        }
    });

    xhr.addEventListener("error", () => {
        uploadProgress.classList.remove("active");
        showUploadStatus("error", "Server error");
    });

    xhr.open("POST", `${API_BASE}/upload_video/`);
    xhr.send(formData);
}

function updateProgress(pct) {
    const circumference = 2 * Math.PI * 42; // r=42 from SVG
    const offset = circumference - (pct / 100) * circumference;
    progressCircle.style.strokeDashoffset = offset;
    progressText.textContent = `${pct}%`;
}

function showUploadStatus(type, text) {
    uploadStatusChip.className = `chip ${type}`;
    uploadStatusChip.textContent = text;
}

function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

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

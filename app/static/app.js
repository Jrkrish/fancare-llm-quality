// FanCare chat UI: streams answers from /chat/stream and keeps the session in localStorage.
const $ = (id) => document.querySelector(`[data-testid="${id}"]`);
const messages = $("messages");
const errorBanner = $("error-banner");

function newSessionId() {
  return "web-" + Math.random().toString(36).slice(2, 12);
}

let sessionId = localStorage.getItem("fancare-session") || newSessionId();
localStorage.setItem("fancare-session", sessionId);
$("session-id").textContent = sessionId;
$("api-key").value = localStorage.getItem("fancare-api-key") || $("api-key").value;
$("api-key").addEventListener("change", (e) => localStorage.setItem("fancare-api-key", e.target.value));

const headers = () => ({ "X-API-Key": $("api-key").value });

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.dataset.testid = `message-${role}`;
  el.dataset.chunks = "0";
  el.textContent = text;
  messages.appendChild(el);
  return el;
}

function renderCitations(el, citations) {
  if (!citations || !citations.length) return;
  const box = document.createElement("div");
  box.className = "citations";
  box.dataset.testid = "citations";
  for (const c of citations) {
    const chip = document.createElement("span");
    chip.className = "citation" + (c.retrieved ? "" : " unverified");
    chip.dataset.testid = "citation";
    chip.textContent = c.chunk_id;
    chip.title = c.passage || "Passage not found";
    box.appendChild(chip);
  }
  el.appendChild(box);
}

function showError(text) {
  errorBanner.textContent = text;
  errorBanner.hidden = false;
}

function parseEvents(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    const type = (part.match(/^event: (.*)$/m) || [])[1];
    const data = (part.match(/^data: (.*)$/m) || [])[1];
    if (type && data) onEvent(type, JSON.parse(data));
  }
  return rest;
}

async function send(text) {
  errorBanner.hidden = true;
  addMessage("user", text);
  const bot = addMessage("assistant", "");
  bot.classList.add("streaming");
  $("send-button").disabled = true;
  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${resp.status})`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer = parseEvents(buffer + decoder.decode(value, { stream: true }), (type, data) => {
        if (type === "delta") {
          bot.textContent += data.text;
          bot.dataset.chunks = String(Number(bot.dataset.chunks) + 1);
        } else if (type === "done") {
          bot.textContent = data.answer;
          renderCitations(bot, data.citations);
          bot.dataset.traceId = data.trace_id;
        } else if (type === "error") {
          throw new Error(data.detail || "The assistant is unavailable");
        }
      });
    }
  } catch (err) {
    bot.remove();
    showError(`Sorry, something went wrong: ${err.message}. Please try again.`);
  } finally {
    bot.classList.remove("streaming");
    $("send-button").disabled = false;
  }
}

async function loadHistory() {
  try {
    const resp = await fetch(`/sessions/${sessionId}/history`, { headers: headers() });
    if (!resp.ok) return;
    const { turns } = await resp.json();
    for (const t of turns) {
      addMessage("user", t.user);
      addMessage("assistant", t.assistant);
    }
  } catch (_) { /* history is a convenience; ignore failures */ }
}

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("message-input").value.trim();
  if (!text) return;
  $("message-input").value = "";
  send(text);
});

$("new-session").addEventListener("click", () => {
  sessionId = newSessionId();
  localStorage.setItem("fancare-session", sessionId);
  $("session-id").textContent = sessionId;
  messages.innerHTML = "";
  errorBanner.hidden = true;
});

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("upload-input").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  const out = $("upload-result");
  out.hidden = false;
  out.textContent = "Reading document...";
  const resp = await fetch("/upload", { method: "POST", headers: headers(), body: form });
  const body = await resp.json().catch(() => ({}));
  out.dataset.status = resp.ok ? body.status : "error";
  out.textContent = resp.ok ? JSON.stringify(body, null, 2) : `Upload failed: ${body.detail || resp.status}`;
});

loadHistory();

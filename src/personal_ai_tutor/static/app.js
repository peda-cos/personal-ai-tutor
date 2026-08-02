"use strict";

const $ = (selector) => document.querySelector(selector);

const state = {
  token: sessionStorage.getItem("tutor.learnerToken") || "",
  learnerId: sessionStorage.getItem("tutor.learnerId") || "",
  voiceRef: sessionStorage.getItem("tutor.voiceRef") || "",
  sessionId: "",
  parentKey: "",
};

function setStatus(text) {
  $("#voice-status").textContent = text;
}

function showTab(name) {
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
});

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  return headers;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const type = response.headers.get("content-type") || "";
  let data;
  if (type.includes("application/json")) {
    data = await response.json();
  } else if (type.includes("application/pdf")) {
    data = await response.blob();
  } else {
    data = await response.text();
  }
  if (!response.ok) {
    const message = data && data.error ? data.error.message : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

function appendMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  $("#transcript").appendChild(node);
  $("#transcript").scrollTop = $("#transcript").scrollHeight;
}

function masteryPercent(turn) {
  const value = turn.mastery_after ?? turn.mastery_before;
  return value === null || value === undefined ? null : Math.round(value * 100);
}

function renderVisual(visual) {
  const host = $("#visual");
  host.replaceChildren();
  if (!visual || visual.kind !== "grid") return;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 320 320");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Coordinate grid with plotted points");
  const min = -5;
  const max = 5;
  const scale = 30;
  const offset = 160;
  const line = (x1, y1, x2, y2, color, width) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", "line");
    node.setAttribute("x1", x1); node.setAttribute("y1", y1);
    node.setAttribute("x2", x2); node.setAttribute("y2", y2);
    node.setAttribute("stroke", color); node.setAttribute("stroke-width", width);
    svg.appendChild(node);
  };
  for (let value = min; value <= max; value += 1) {
    line(offset + value * scale, 20, offset + value * scale, 300, "#e4eaf3", 1);
    line(20, offset - value * scale, 300, offset - value * scale, "#e4eaf3", 1);
  }
  line(20, offset, 300, offset, "#65758b", 1.5);
  line(offset, 20, offset, 300, "#65758b", 1.5);
  (visual.points || []).forEach(([x, y]) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", offset + x * scale);
    dot.setAttribute("cy", offset - y * scale);
    dot.setAttribute("r", 5);
    dot.setAttribute("fill", "#1f5fbf");
    svg.appendChild(dot);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", offset + x * scale + 8);
    label.setAttribute("y", offset - y * scale - 8);
    label.setAttribute("font-size", "11");
    label.setAttribute("fill", "#172033");
    label.textContent = `(${x}, ${y})`;
    svg.appendChild(label);
  });
  host.appendChild(svg);
}

function applyTurn(turn) {
  state.sessionId = turn.session_id;
  $("#concept-title").textContent = turn.concept_title || "Algebra";
  const percent = masteryPercent(turn);
  $("#mastery-label").textContent = percent === null ? "Mastery: not started" : `Mastery: ${percent}%`;
  $("#mastery-bar").style.width = `${percent || 0}%`;
  $("#problem").textContent = turn.item_prompt || "";
  $("#problem").classList.toggle("hidden", !turn.item_prompt);
  renderVisual(turn.visual);
  if (turn.message) appendMessage("tutor", turn.message);
  if (turn.safety_blocked) appendMessage("system", "A safety filter handled this turn.");
}

async function startSession() {
  if (!state.token) return;
  try {
    const turn = await request("/api/session", {
      method: "POST",
      headers: authHeaders(),
      body: {},
    });
    appendMessage("system", "New local session. The tutor will ask before it tells.");
    applyTurn(turn);
  } catch (error) {
    appendMessage("system", error.message);
  }
}

$("#create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const displayName = $("#display-name").value.trim();
  const ageBand = $("#age-band").value;
  const consent = $("#consent").checked;
  try {
    const data = await request("/api/learners", {
      method: "POST",
      body: { display_name: displayName, age_band: ageBand, consent },
    });
    state.token = data.learner_token;
    state.learnerId = data.learner_id;
    state.voiceRef = data.voice_ref;
    sessionStorage.setItem("tutor.learnerToken", state.token);
    sessionStorage.setItem("tutor.learnerId", state.learnerId);
    sessionStorage.setItem("tutor.voiceRef", state.voiceRef);
    $("#learner-token").textContent = data.learner_token;
    $("#parent-key").textContent = data.parent_key;
    $("#credentials").classList.remove("hidden");
    appendMessage("system", "Learner created. The token above is the only login credential.");
    await startSession();
  } catch (error) {
    appendMessage("system", error.message);
  }
});

async function sendText(text) {
  if (!state.token || !state.sessionId) {
    appendMessage("system", "Create a learner and start a session first.");
    return;
  }
  if (!text.trim()) return;
  appendMessage("learner", text);
  try {
    const turn = await request("/api/chat", {
      method: "POST",
      headers: authHeaders(),
      body: { session_id: state.sessionId, text },
    });
    applyTurn(turn);
  } catch (error) {
    appendMessage("system", error.message);
  }
}

$("#send").addEventListener("click", async () => {
  const input = $("#answer");
  const text = input.value;
  input.value = "";
  await sendText(text);
});

$("#answer").addEventListener("keydown", async (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    const input = $("#answer");
    const text = input.value;
    input.value = "";
    await sendText(text);
  }
});

$("#hint").addEventListener("click", () => sendText("I'm stuck"));
$("#skip").addEventListener("click", () => sendText("next problem"));

$("#photo").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  if (!state.sessionId) {
    appendMessage("system", "Start a session before uploading a photo.");
    return;
  }
  const reader = new FileReader();
  reader.onload = async () => {
    const result = String(reader.result || "");
    const encoded = result.includes(",") ? result.split(",")[1] : result;
    appendMessage("learner", `Photo submitted: ${file.name}`);
    try {
      const turn = await request("/api/photo", {
        method: "POST",
        headers: authHeaders(),
        body: {
          session_id: state.sessionId,
          image_b64: encoded,
          media_type: file.type || "image/png",
        },
      });
      applyTurn(turn);
    } catch (error) {
      appendMessage("system", error.message);
    }
  };
  reader.readAsDataURL(file);
  event.target.value = "";
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;

function stopSpeech() {
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  setStatus("");
}

$("#stop-speech").addEventListener("click", stopSpeech);

async function speak(text) {
  if (!window.speechSynthesis || !text) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.98;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

$("#mic").addEventListener("click", () => {
  if (!state.sessionId) {
    appendMessage("system", "Start a session before speaking.");
    return;
  }
  stopSpeech();
  if (!SpeechRecognition) {
    const fallback = window.prompt("Speech recognition is unavailable in this browser. Type what you would say:");
    if (fallback) sendVoice(fallback);
    return;
  }
  if (!recognizer) {
    recognizer = new SpeechRecognition();
    recognizer.lang = "en-US";
    recognizer.interimResults = false;
    recognizer.maxAlternatives = 1;
    recognizer.onstart = () => setStatus("Listening...");
    recognizer.onerror = (event) => setStatus(`Voice input unavailable: ${event.error}`);
    recognizer.onend = () => setStatus("");
    recognizer.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      sendVoice(transcript);
    };
  }
  recognizer.start();
});

async function sendVoice(transcript) {
  appendMessage("learner", transcript);
  try {
    const turn = await request("/api/voice", {
      method: "POST",
      headers: authHeaders({ "X-Speaker-Ref": state.voiceRef }),
      body: { session_id: state.sessionId, transcript, barge_in: false },
    });
    applyTurn(turn);
    if (turn.tts_b64) {
      speak(turn.message);
    }
  } catch (error) {
    appendMessage("system", error.message);
  }
}

function statCard(label, value, note) {
  const card = document.createElement("div");
  card.className = "stat";
  const strong = document.createElement("strong");
  strong.textContent = value;
  const span = document.createElement("span");
  span.textContent = label;
  card.append(strong, span);
  if (note) {
    const small = document.createElement("p");
    small.className = "small muted";
    small.textContent = note;
    card.appendChild(small);
  }
  return card;
}

async function loadProgress(parentKey) {
  const data = await request("/api/progress", {
    headers: { "X-Parent-Key": parentKey },
  });
  const host = $("#progress-result");
  host.replaceChildren();
  const heading = document.createElement("div");
  heading.className = "stack";
  const name = document.createElement("h3");
  name.textContent = data.learner.display_name;
  const meta = document.createElement("p");
  meta.className = "muted";
  meta.textContent = `Age band: ${data.learner.age_band}. Interactions: ${data.interaction_count}.`;
  heading.append(name, meta);
  host.appendChild(heading);

  if (data.mastery && data.mastery.length) {
    const table = document.createElement("table");
    const head = document.createElement("tr");
    ["Concept", "Mastery", "Attempts"].forEach((text) => {
      const th = document.createElement("th");
      th.textContent = text;
      head.appendChild(th);
    });
    table.appendChild(head);
    data.mastery.forEach((row) => {
      const tr = document.createElement("tr");
      const title = document.createElement("td");
      title.textContent = row.title;
      const mastery = document.createElement("td");
      mastery.textContent = `${Math.round(row.mastery * 100)}%`;
      const attempts = document.createElement("td");
      attempts.textContent = String(row.attempts);
      tr.append(title, mastery, attempts);
      table.appendChild(tr);
    });
    host.appendChild(table);
  } else {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No mastery records yet.";
    host.appendChild(empty);
  }

  const next = document.createElement("p");
  next.textContent = `Next up: ${(data.next_concepts || []).map((item) => item.title).join(", ") || "review queue"}`;
  host.appendChild(next);

  if (data.safety_events && data.safety_events.length) {
    const safety = document.createElement("div");
    safety.className = "callout";
    const strong = document.createElement("strong");
    strong.textContent = "Safety events";
    safety.appendChild(strong);
    data.safety_events.forEach((event) => {
      const line = document.createElement("p");
      line.textContent = `${event.ts.slice(0, 10)} - ${event.category} (${event.blocked ? "blocked" : "redacted"})`;
      safety.appendChild(line);
    });
    host.appendChild(safety);
  }

  const retention = document.createElement("p");
  retention.className = "small muted";
  retention.textContent = "Retention: interactions/semantic/safety 365 days; audit 730 days; inactive learners purged after 365 days.";
  host.appendChild(retention);
  $("#progress-actions").classList.remove("hidden");
}

$("#progress-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  state.parentKey = $("#progress-key").value.trim();
  try {
    await loadProgress(state.parentKey);
  } catch (error) {
    $("#progress-result").textContent = error.message;
    $("#progress-actions").classList.add("hidden");
  }
});

$("#download-report").addEventListener("click", async () => {
  try {
    const blob = await request("/api/report.pdf?week=0", {
      headers: { "X-Parent-Key": state.parentKey },
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "weekly-report.pdf";
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(error.message);
  }
});

$("#export-data").addEventListener("click", async () => {
  try {
    const data = await request("/api/export", {
      headers: { "X-Parent-Key": state.parentKey },
    });
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "learner-export.json";
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(error.message);
  }
});

$("#delete-data").addEventListener("click", async () => {
  if (!window.confirm("Delete all learner data? This cannot be undone.")) return;
  try {
    await request("/api/memory", {
      method: "DELETE",
      headers: { "X-Parent-Key": state.parentKey },
    });
    $("#progress-result").textContent = "Learner data deleted.";
    $("#progress-actions").classList.add("hidden");
    sessionStorage.clear();
    state.token = "";
    state.sessionId = "";
    $("#concept-title").textContent = "No session yet";
  } catch (error) {
    alert(error.message);
  }
});

$("#load-study").addEventListener("click", async () => {
  const host = $("#study-result");
  host.textContent = "Loading the deterministic simulation...";
  try {
    const data = await request("/api/study");
    host.replaceChildren();
    const banner = document.createElement("p");
    banner.className = "callout";
    banner.textContent = "Simulated learners only. This harness measures the sequencing mechanism, not human learning.";
    host.appendChild(banner);
    const grid = document.createElement("div");
    grid.className = "stat-grid";
    grid.append(
      statCard("Adaptive gain", data.adaptive.gain.toFixed(1), `pre ${data.adaptive.pre_mean.toFixed(1)} / post ${data.adaptive.post_mean.toFixed(1)}`),
      statCard("Baseline gain", data.baseline.gain.toFixed(1), `pre ${data.baseline.pre_mean.toFixed(1)} / post ${data.baseline.post_mean.toFixed(1)}`),
      statCard("Delta (95% CI)", `${data.delta.toFixed(1)}`, `[${data.delta_ci95[0].toFixed(1)}, ${data.delta_ci95[1].toFixed(1)}]`),
    );
    host.appendChild(grid);
    const detail = document.createElement("p");
    detail.textContent = `n = ${data.learners_per_arm} per arm, ${data.sessions} sessions, ${data.pre_post_items}-item pre/post, Cohen's d = ${data.cohens_d.toFixed(2)}.`;
    host.appendChild(detail);
    const note = document.createElement("p");
    note.className = "small muted";
    note.textContent = data.protocol;
    host.appendChild(note);
  } catch (error) {
    host.textContent = error.message;
  }
});

if (state.token) {
  appendMessage("system", "Learner token found in this tab. Starting a local session.");
  startSession();
}

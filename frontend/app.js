/**
 * LUNA AI — Liquid-Glass Frontend Controller
 * Uses relative paths (/api/... and ws(s)://host/ws/chat) so it works seamlessly
 * on Windows 10 localhost:8000 AND proxied live preview environments.
 */
(() => {
  const state = {
    sessionId: localStorage.getItem("luna_session_id") || `sess_${Date.now().toString(36)}`,
    settings: null,
    activeLangProfile: "fa_profile",
    ws: null,
    busy: false,
  };
  localStorage.setItem("luna_session_id", state.sessionId);

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const PROVENANCE_LABELS = {
    memory: "🧠 حافظه بلندمدت داخلی",
    internet: "🌐 اینترنت (مرز داده غیرمطمئن)",
    tool: "🛠️ نتیجه قطعی ابزار",
    local_model: "💻 مدل محلی qwen3:0.6b",
    hybrid_teacher: "✨ معلم آنلاین رایگان + تقطیر دانش",
    arena_dual: "⚖️ مقایسه دوگانه Arena",
    confirmation: "🔐 منتظر تأیید کاربر",
    security: "🛡️ سپر امنیتی لونا",
    system: "⚙️ سیستم",
    general: "💬 پاسخ عمومی",
  };

  function toast(msg) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    $("#toastContainer").appendChild(el);
    setTimeout(() => el.remove(), 3400);
  }

  async function api(path, options = {}) {
    const opts = {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    };
    if (opts.body && typeof opts.body !== "string" && !(opts.body instanceof FormData)) {
      opts.body = JSON.stringify(opts.body);
    }
    if (opts.body instanceof FormData) {
      delete opts.headers["Content-Type"];
    }
    const resp = await fetch(path, opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatRichText(raw) {
    let html = escapeHtml(raw);
    html = html.replace(/```([\s\S]*?)```/g, '<pre style="direction:ltr;text-align:left;background:rgba(2,6,23,0.65);padding:8px;border-radius:8px;overflow-x:auto;margin:6px 0;"><code>$1</code></pre>');
    html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color:#38bdf8;">$1</a>');
    return html;
  }

  // --- Health & Status Polling ---
  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      const netPill = $("#pillInternet");
      netPill.className = `status-pill ${h.internet_online ? "online" : "offline"}`;
      netPill.querySelector(".label").textContent = h.internet_online ? "شبکه: آنلاین" : "شبکه: آفلاین";

      const ol = h.ollama || {};
      const olPill = $("#pillOllama");
      olPill.className = `status-pill ${ol.online && ol.model_available ? "online" : ol.online ? "warn" : "offline"}`;
      olPill.querySelector(".label").textContent = ol.online
        ? `Ollama: ${ol.active_model}${ol.model_available ? "" : " (نصب نشده)"}`
        : "Ollama: آفلاین";

      if (h.ram) {
        const procTxt = h.hardware?.processor?.processor_summary || "";
        $("#pillRam .label").textContent = `RAM: ${h.ram.used_gb}/${h.ram.total_gb}GB${procTxt ? " • " + procTxt : ""}`;
      }
      if (h.hardware) {
        document.body.classList.toggle("lite-glass", Boolean(h.hardware.ui_lite_glass));
      }
      $("#selectMode").value = h.operating_mode;
      $("#selectLang").value = h.preferred_language;
    } catch {
      $("#pillOllama").className = "status-pill offline";
      $("#pillOllama .label").textContent = "سرور در دسترس نیست";
    }
  }

  // --- WebSocket Real-Time Stream & Tool Status ---
  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/chat`;
    try {
      const ws = new WebSocket(wsUrl);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "status") {
          setBusy(true, msg.message || "در حال پردازش...");
        } else if (msg.type === "tool_start") {
          setBusy(true, `⚙️ در حال اجرای ابزار مجاز: ${msg.tool} ...`);
        } else if (msg.type === "tool_end") {
          setBusy(true, `✓ ابزار ${msg.tool} کامل شد؛ در حال تدوین پاسخ...`);
        } else if (msg.type === "cancelled") {
          setBusy(false);
          toast("عملیات متوقف شد.");
        } else if (msg.type === "final") {
          setBusy(false);
          renderAssistantTurn(msg.data);
          refreshSidebarData();
        } else if (msg.type === "error") {
          setBusy(false);
          toast(`خطا: ${msg.error}`);
        }
      };
      ws.onclose = () => {
        state.ws = null;
      };
      state.ws = ws;
    } catch {
      state.ws = null;
    }
  }

  function setBusy(flag, text = "در حال پردازش...") {
    state.busy = flag;
    $("#btnSend").disabled = flag;
    const bar = $("#liveActivityBar");
    if (flag) {
      bar.classList.remove("hidden");
      $("#liveActivityText").textContent = text;
    } else {
      bar.classList.add("hidden");
    }
  }

  // --- Render Messages ---
  function appendUserBubble(text) {
    const welcome = $(".welcome-card");
    if (welcome) welcome.remove();

    const row = document.createElement("div");
    row.className = "msg-row user";
    row.innerHTML = `<div class="msg-bubble" dir="auto">${formatRichText(text)}</div>`;
    $("#chatMessages").appendChild(row);
    scrollToBottom();
  }

  function renderAssistantTurn(data) {
    const welcome = $(".welcome-card");
    if (welcome) welcome.remove();

    const prov = data.provenance || "general";
    const provLabel = PROVENANCE_LABELS[prov] || prov;
    const row = document.createElement("div");
    row.className = "msg-row assistant";

    let toolsHtml = "";
    if (Array.isArray(data.executed_tools) && data.executed_tools.length > 0) {
      const items = data.executed_tools
        .map(
          (t) =>
            `<div>• <code>${escapeHtml(t.tool)}</code> — وضعیت: <strong>${t.ok ? "موفق" : "ناموفق"}</strong> (${t.latency_ms || 0}ms)${
              t.backup_created ? ` | پشتیبان: <code>${escapeHtml(t.backup_created)}</code>` : ""
            }</div>`
        )
        .join("");
      toolsHtml = `
        <details class="tool-trace-box">
          <summary>ابزارهای اجراشده (${data.executed_tools.length})</summary>
          <div style="margin-top:6px;">${items}</div>
          <pre>${escapeHtml(JSON.stringify(data.executed_tools, null, 2))}</pre>
        </details>`;
    }

    let arenaHtml = "";
    if (data.arena_comparison) {
      const ac = data.arena_comparison;
      arenaHtml = `
        <div class="arena-dual-grid">
          <div class="arena-col">
            <h4>💻 مدل محلی (${escapeHtml(ac.local_model)} • ${ac.local_latency_ms}ms)</h4>
            <div dir="auto">${formatRichText(ac.local_output)}</div>
          </div>
          <div class="arena-col">
            <h4>✨ معلم آنلاین (${escapeHtml(ac.teacher_provider)} • ${ac.teacher_latency_ms}ms)</h4>
            <div dir="auto">${formatRichText(ac.teacher_output)}</div>
          </div>
        </div>`;
    }

    let confirmHtml = "";
    if (data.pending_confirmation && data.pending_confirmation.token) {
      const pc = data.pending_confirmation;
      confirmHtml = `
        <div class="confirm-glass-card" data-token="${escapeHtml(pc.token)}">
          <strong>🔐 درخواست مجوز عملیات حساس:</strong>
          <div>${escapeHtml(pc.summary_fa || pc.summary_en)}</div>
          <div class="confirm-actions">
            <button class="glass-btn primary compact btn-approve" data-token="${escapeHtml(pc.token)}">تأیید و اجرا</button>
            <button class="glass-btn danger compact btn-reject" data-token="${escapeHtml(pc.token)}">رد و لغو</button>
          </div>
        </div>`;
    }

    const latencyNote = data.latency_ms ? `<span>⏱ ${data.latency_ms}ms</span>` : "";

    row.innerHTML = `
      <div class="msg-meta-bar">
        <span class="provenance-badge ${escapeHtml(prov)}">${escapeHtml(provLabel)}</span>
        ${latencyNote}
      </div>
      <div class="msg-bubble" dir="auto">${formatRichText(data.reply || "")}${arenaHtml}${toolsHtml}${confirmHtml}</div>
    `;

    row.querySelectorAll(".btn-approve").forEach((btn) =>
      btn.addEventListener("click", () => handleConfirm(btn.dataset.token, true, btn.closest(".confirm-glass-card")))
    );
    row.querySelectorAll(".btn-reject").forEach((btn) =>
      btn.addEventListener("click", () => handleConfirm(btn.dataset.token, false, btn.closest(".confirm-glass-card")))
    );

    $("#chatMessages").appendChild(row);
    scrollToBottom();
  }

  async function handleConfirm(token, approved, cardEl) {
    try {
      const res = await api("/api/confirm", {
        method: "POST",
        body: { token, approved, language: $("#selectLang").value === "en" ? "en" : "fa" },
      });
      if (cardEl) {
        cardEl.innerHTML = `<em>${approved ? "✓ تأیید شد" : "✕ لغو شد"}</em>`;
      }
      renderAssistantTurn(res);
      refreshSidebarData();
    } catch (err) {
      toast(`خطا در ثبت تصمیم: ${err.message}`);
    }
  }

  function scrollToBottom() {
    const box = $("#chatMessages");
    box.scrollTop = box.scrollHeight;
  }

  // --- Send Message ---
  async function sendMessage(text) {
    const msg = (text || "").trim();
    if (!msg || state.busy) return;
    $("#chatInput").value = "";
    appendUserBubble(msg);
    setBusy(true, "در حال مسیریابی و پردازش...");

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ action: "chat", message: msg, session_id: state.sessionId }));
      return;
    }

    try {
      const res = await api("/api/chat", {
        method: "POST",
        body: { message: msg, session_id: state.sessionId },
      });
      setBusy(false);
      renderAssistantTurn(res);
      refreshSidebarData();
    } catch (err) {
      setBusy(false);
      toast(`خطای ارتباط: ${err.message}`);
    }
  }

  // --- Sessions, Files, Memories, Settings, Exemplars, Audit ---
  async function loadSessions() {
    try {
      const data = await api("/api/sessions");
      const box = $("#sessionList");
      box.innerHTML = "";
      (data.sessions || []).forEach((s) => {
        const div = document.createElement("div");
        div.className = `list-item ${s.session_id === state.sessionId ? "active" : ""}`;
        div.innerHTML = `
          <span class="list-item-title">${escapeHtml(s.title || s.session_id)} (${s.message_count || 0})</span>
          <button class="glass-btn danger tiny btn-del-sess" title="حذف">✕</button>
        `;
        div.addEventListener("click", (e) => {
          if (e.target.classList.contains("btn-del-sess")) return;
          switchSession(s.session_id);
        });
        div.querySelector(".btn-del-sess").addEventListener("click", async (e) => {
          e.stopPropagation();
          await api(`/api/sessions/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
          if (s.session_id === state.sessionId) {
            state.sessionId = `sess_${Date.now().toString(36)}`;
            localStorage.setItem("luna_session_id", state.sessionId);
            $("#chatMessages").innerHTML = "";
          }
          loadSessions();
        });
        box.appendChild(div);
      });
    } catch {}
  }

  async function switchSession(sessionId) {
    state.sessionId = sessionId;
    localStorage.setItem("luna_session_id", sessionId);
    const data = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    const box = $("#chatMessages");
    box.innerHTML = "";
    (data.messages || []).forEach((m) => {
      if (m.role === "user") {
        appendUserBubble(m.content);
      } else {
        renderAssistantTurn({
          reply: m.content,
          provenance: m.provenance,
          executed_tools: [],
          arena_comparison: m.meta ? m.meta.arena_comparison : null,
          pending_confirmation: m.meta ? m.meta.pending_confirmation : null,
        });
      }
    });
    loadSessions();
  }

  async function loadFiles() {
    try {
      const data = await api("/api/files");
      const box = $("#fileList");
      box.innerHTML = "";
      const files = data.files || [];
      if (!files.length) {
        box.innerHTML = `<div class="tiny-note">هنوز فایلی در فضای کاری نیست.</div>`;
        return;
      }
      files.forEach((f) => {
        const div = document.createElement("div");
        div.className = "list-item";
        div.innerHTML = `
          <span class="list-item-title" dir="ltr">${escapeHtml(f.path)}</span>
          <button class="glass-btn secondary tiny btn-view-file">مشاهده</button>
        `;
        div.addEventListener("click", (e) => {
          if (e.target.classList.contains("btn-view-file")) return;
          $("#chatInput").value = `فایل ${f.path} را بخوان`;
          $("#chatInput").focus();
        });
        div.querySelector(".btn-view-file").addEventListener("click", async (e) => {
          e.stopPropagation();
          try {
            const res = await api("/api/files/inspect", { method: "POST", body: { path: f.path } });
            openModal(`پیش‌نمایش امن: ${f.path}`, JSON.stringify(res, null, 2));
          } catch (err) {
            toast(err.message);
          }
        });
        box.appendChild(div);
      });
    } catch {}
  }

  async function loadMemories(query = "") {
    try {
      const url = query ? `/api/memories?q=${encodeURIComponent(query)}` : "/api/memories?include_inactive=true";
      const data = await api(url);
      const box = $("#memoryList");
      box.innerHTML = "";
      const mems = data.memories || [];
      if (!mems.length) {
        box.innerHTML = `<div class="tiny-note">موردی در حافظه بلندمدت ثبت نشده است.</div>`;
        return;
      }
      mems.forEach((m) => {
        const card = document.createElement("div");
        card.className = `mem-card ${m.is_active ? "" : "inactive"}`;
        card.innerHTML = `
          <div class="mem-card-head">
            <span>[${escapeHtml(m.category)}] ${escapeHtml(m.title)}</span>
            <span>★${m.importance}</span>
          </div>
          <div class="mem-card-body" dir="auto">${escapeHtml(m.content)}</div>
          <div class="mem-card-foot">
            <span>استفاده: ${m.use_count} بار</span>
            <div class="btn-row">
              <button class="glass-btn secondary tiny btn-edit-mem">ویرایش</button>
              <button class="glass-btn secondary tiny btn-toggle-mem">${m.is_active ? "غیرفعال" : "فعال"}</button>
              <button class="glass-btn danger tiny btn-del-mem">فراموش</button>
            </div>
          </div>
        `;
        card.querySelector(".btn-edit-mem").addEventListener("click", async () => {
          const updated = prompt("ویرایش متن حافظه:", m.content);
          if (updated && updated.trim() !== m.content) {
            try {
              await api(`/api/memories/${m.id}`, { method: "PATCH", body: { content: updated.trim() } });
              toast("حافظه اصلاح شد.");
              loadMemories(query);
            } catch (err) {
              toast(err.message);
            }
          }
        });
        card.querySelector(".btn-toggle-mem").addEventListener("click", async () => {
          await api(`/api/memories/${m.id}`, { method: "PATCH", body: { is_active: !m.is_active } });
          loadMemories(query);
        });
        card.querySelector(".btn-del-mem").addEventListener("click", async () => {
          await api(`/api/memories/${m.id}`, { method: "DELETE" });
          toast("از حافظه بلندمدت پاک شد.");
          loadMemories(query);
        });
        box.appendChild(card);
      });
    } catch {}
  }

  async function loadSettingsIntoLab() {
    try {
      const data = await api("/api/settings");
      state.settings = data.settings;
      const sel = $("#paramActiveModel");
      sel.innerHTML = "";
      const models = data.installed_models && data.installed_models.length ? data.installed_models : [state.settings.active_model];
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === state.settings.active_model) opt.selected = true;
        sel.appendChild(opt);
      });
      populateProfileInputs();

      const hw = state.settings.hardware || {};
      $("#hwAccelMode").value = hw.accelerator_mode || "hybrid_cpu_gpu";
      $("#hwCpuThreads").value = hw.cpu_threads ?? 0;
      $("#hwGpuLayers").value = hw.gpu_layers ?? 14;
      $("#hwLiteGlass").checked = hw.ui_lite_glass !== false;
      document.body.classList.toggle("lite-glass", $("#hwLiteGlass").checked);

      const t = state.settings.teacher || {};
      $("#teacherEnabledToggle").checked = Boolean(t.enabled);
      $("#teacherProviderSelect").value = t.provider || "pollinations_free";
      $("#customTeacherUrl").value = t.custom_base_url || "";
      $("#customTeacherUrlWrap").classList.toggle("hidden", t.provider !== "custom_free_openai");
    } catch {}
  }

  function populateProfileInputs() {
    if (!state.settings) return;
    const p = state.settings[state.activeLangProfile] || {};
    $("#paramTemp").value = p.temperature ?? 0.42;
    $("#valTemp").textContent = p.temperature ?? 0.42;
    $("#paramTopP").value = p.top_p ?? 0.82;
    $("#valTopP").textContent = p.top_p ?? 0.82;
    $("#paramPresPen").value = p.presence_penalty ?? 1.45;
    $("#valPresPen").textContent = p.presence_penalty ?? 1.45;
    $("#paramRepPen").value = p.repeat_penalty ?? 1.05;
    $("#valRepPen").textContent = p.repeat_penalty ?? 1.05;
    $("#paramTopK").value = p.top_k ?? 20;
    $("#paramMaxTok").value = p.max_tokens ?? 320;
    $("#paramNumCtx").value = p.num_ctx ?? 2048;
    $("#paramSeed").value = p.seed ?? "";
    $("#paramThink").checked = Boolean(p.think);
  }

  async function loadExemplars() {
    try {
      const data = await api("/api/teacher/exemplars");
      const list = data.exemplars || [];
      $("#exemplarCount").textContent = list.length;
      const box = $("#exemplarList");
      box.innerHTML = "";
      list.slice(0, 20).forEach((ex) => {
        const d = document.createElement("div");
        d.className = "mem-card";
        d.innerHTML = `
          <div class="mem-card-head"><span>${escapeHtml(ex.teacher_source)}</span><span>${escapeHtml(ex.language)}</span></div>
          <div class="mem-card-body" dir="auto"><strong>سؤال:</strong> ${escapeHtml(ex.user_prompt.slice(0, 90))}<br/><strong>پاسخ:</strong> ${escapeHtml(ex.ideal_response.slice(0, 120))}...</div>
        `;
        box.appendChild(d);
      });
    } catch {}
  }

  async function loadAudit() {
    try {
      const data = await api("/api/audit");
      const box = $("#auditList");
      box.innerHTML = "";
      (data.logs || []).forEach((l) => {
        const d = document.createElement("div");
        d.className = "mem-card";
        d.innerHTML = `
          <div class="mem-card-head"><span>${escapeHtml(l.event_type)}</span><span>${escapeHtml(l.status)}</span></div>
          <div class="tiny-note" dir="ltr">${escapeHtml(l.timestamp.slice(0, 19))} • ${escapeHtml(JSON.stringify(l.details))}</div>
        `;
        box.appendChild(d);
      });
    } catch {}
  }

  function openModal(title, bodyText) {
    $("#modalTitle").textContent = title;
    $("#modalBody").textContent = bodyText;
    $("#glassModal").classList.remove("hidden");
  }

  function refreshSidebarData() {
    loadSessions();
    loadFiles();
    loadMemories($("#memorySearchInput").value.trim());
    loadExemplars();
    refreshHealth();
  }

  // --- Event Listeners ---
  $("#chatForm").addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage($("#chatInput").value);
  });

  $("#chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage($("#chatInput").value);
    }
  });

  $("#btnCancelTurn").addEventListener("click", async () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ action: "cancel", session_id: state.sessionId }));
    } else {
      await api("/api/chat/cancel", { method: "POST", body: { session_id: state.sessionId } });
    }
    setBusy(false);
  });

  $$(".starter-card").forEach((btn) =>
    btn.addEventListener("click", () => sendMessage(btn.dataset.prompt))
  );

  $$(".chip-btn").forEach((btn) =>
    btn.addEventListener("click", () => {
      if (btn.dataset.send) {
        sendMessage(btn.dataset.send);
      } else if (btn.dataset.prefix) {
        $("#chatInput").value = btn.dataset.prefix;
        $("#chatInput").focus();
      }
    })
  );

  $("#btnNewSession").addEventListener("click", () => {
    state.sessionId = `sess_${Date.now().toString(36)}`;
    localStorage.setItem("luna_session_id", state.sessionId);
    $("#chatMessages").innerHTML = "";
    loadSessions();
    toast("گفتگوی جدید آغاز شد.");
  });

  $("#btnClearSession").addEventListener("click", async () => {
    await api(`/api/sessions/${encodeURIComponent(state.sessionId)}/clear`, { method: "POST" });
    $("#chatMessages").innerHTML = "";
    toast("تاریخچه گفتگوی جاری پاک شد.");
    loadSessions();
  });

  $("#fileUploadInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await api("/api/files/upload", { method: "POST", body: fd });
      toast(`فایل «${res.filename}» آپلود شد.`);
      loadFiles();
    } catch (err) {
      toast(`خطای آپلود: ${err.message}`);
    }
    e.target.value = "";
  });

  $("#selectMode").addEventListener("change", async (e) => {
    await api("/api/settings", { method: "PATCH", body: { operating_mode: e.target.value } });
    toast(`حالت اجرا تغییر یافت: ${e.target.value}`);
    refreshHealth();
  });

  $("#selectLang").addEventListener("change", async (e) => {
    await api("/api/settings", { method: "PATCH", body: { preferred_language: e.target.value } });
    toast("ترجیح زبان ذخیره شد.");
  });

  $("#btnEnvModal").addEventListener("click", async () => {
    const rep = await api("/api/env");
    openModal("گزارش واقعی بررسی محیط سیستم (Stage 1 Diagnostics)", JSON.stringify(rep, null, 2));
  });

  $("#btnCloseModal").addEventListener("click", () => $("#glassModal").classList.add("hidden"));
  $("#glassModal").addEventListener("click", (e) => {
    if (e.target.id === "glassModal") $("#glassModal").classList.add("hidden");
  });

  // Inspector Tabs
  $$(".tab-btn").forEach((btn) =>
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => b.classList.remove("active"));
      $$(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      $(`#${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "tabAudit") loadAudit();
    })
  );

  // Memory Controls
  $("#btnSearchMemory").addEventListener("click", () => loadMemories($("#memorySearchInput").value.trim()));
  $("#btnAddMemory").addEventListener("click", async () => {
    const content = $("#memContentInput").value.trim();
    if (!content) return;
    try {
      await api("/api/memories", {
        method: "POST",
        body: {
          title: $("#memTitleInput").value.trim() || undefined,
          category: $("#memCategoryInput").value,
          importance: Number($("#memImportanceInput").value),
          content,
        },
      });
      $("#memContentInput").value = "";
      $("#memTitleInput").value = "";
      toast("در حافظه بلندمدت ذخیره شد.");
      loadMemories();
    } catch (err) {
      toast(err.message);
    }
  });

  // Model Params Controls
  $$(".subtab-btn").forEach((btn) =>
    btn.addEventListener("click", () => {
      $$(".subtab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeLangProfile = btn.dataset.langprofile;
      populateProfileInputs();
    })
  );

  [
    ["#paramTemp", "#valTemp"],
    ["#paramTopP", "#valTopP"],
    ["#paramPresPen", "#valPresPen"],
    ["#paramRepPen", "#valRepPen"],
  ].forEach(([inp, lbl]) => {
    $(inp).addEventListener("input", (e) => ($(lbl).textContent = e.target.value));
  });

  $("#btnSaveParams").addEventListener("click", async () => {
    const seedRaw = $("#paramSeed").value.trim();
    const profilePatch = {
      temperature: Number($("#paramTemp").value),
      top_p: Number($("#paramTopP").value),
      presence_penalty: Number($("#paramPresPen").value),
      repeat_penalty: Number($("#paramRepPen").value),
      top_k: Number($("#paramTopK").value),
      max_tokens: Number($("#paramMaxTok").value),
      num_ctx: Number($("#paramNumCtx").value),
      seed: seedRaw ? Number(seedRaw) : null,
      think: $("#paramThink").checked,
    };
    const hwPatch = {
      accelerator_mode: $("#hwAccelMode").value,
      cpu_threads: Number($("#hwCpuThreads").value),
      gpu_layers: Number($("#hwGpuLayers").value),
      ui_lite_glass: $("#hwLiteGlass").checked,
    };
    try {
      const res = await api("/api/settings", {
        method: "PATCH",
        body: {
          active_model: $("#paramActiveModel").value,
          [state.activeLangProfile]: profilePatch,
          hardware: hwPatch,
        },
      });
      state.settings = res.settings;
      document.body.classList.toggle("lite-glass", hwPatch.ui_lite_glass);
      toast("پارامترهای مدل و تقسیم بار سخت‌افزار ذخیره شد.");
    } catch (err) {
      toast(err.message);
    }
  });

  // Teacher & Distillation Controls
  $("#teacherProviderSelect").addEventListener("change", (e) => {
    $("#customTeacherUrlWrap").classList.toggle("hidden", e.target.value !== "custom_free_openai");
  });

  $("#btnSaveTeacherCfg").addEventListener("click", async () => {
    await api("/api/settings", {
      method: "PATCH",
      body: {
        teacher: {
          enabled: $("#teacherEnabledToggle").checked,
          provider: $("#teacherProviderSelect").value,
          custom_base_url: $("#customTeacherUrl").value.trim(),
        },
      },
    });
    toast("تنظیمات معلم آنلاین ثبت شد.");
    refreshHealth();
  });

  $("#btnProbeLMArena").addEventListener("click", async () => {
    $("#lmarenaProbeStatus").textContent = "در حال بررسی با مرورگر امن Playwright...";
    const res = await api("/api/teacher/probe-lmarena", { method: "POST" });
    $("#lmarenaProbeStatus").textContent = res.ok ? res.content : `⚠️ ${res.error}`;
  });

  $("#btnDistillSave").addEventListener("click", async () => {
    const user_prompt = $("#distillPromptInput").value.trim();
    const ideal_response = $("#distillResponseInput").value.trim();
    if (!user_prompt || !ideal_response) return;
    try {
      await api("/api/teacher/distill", {
        method: "POST",
        body: { user_prompt, ideal_response, language: "fa", source: "manual_distill_studio" },
      });
      $("#distillPromptInput").value = "";
      $("#distillResponseInput").value = "";
      toast("پاسخ طلایی در حافظه آموزشی مدل محلی ذخیره شد.");
      loadExemplars();
    } catch (err) {
      toast(err.message);
    }
  });

  $("#btnRefreshAudit").addEventListener("click", loadAudit);
  $("#btnToggleSidebar").addEventListener("click", () => $("#leftSidebar").classList.toggle("open"));
  $("#btnToggleInspector").addEventListener("click", () => $("#rightInspector").classList.toggle("open"));

  // Boot
  connectWebSocket();
  refreshHealth();
  loadSessions();
  loadFiles();
  loadMemories();
  loadSettingsIntoLab();
  loadExemplars();
  setInterval(refreshHealth, 40000);
})();

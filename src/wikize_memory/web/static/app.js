(() => {
  const tokenKey = "memory-viva-access";
  const state = {
    token: sessionStorage.getItem(tokenKey) || "",
    pending: [],
    approved: [],
    query: "",
    pollTimer: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const settingsDialog = $("#settings-dialog");
  const statusBox = $("#job-status");
  const statusTitle = $("#status-title");
  const statusMessage = $("#status-message");
  const submitButton = $("#submit-button");

  async function request(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...options, headers });
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const message = typeof payload === "object" && payload !== null ? payload.detail || "A solicitação falhou." : payload;
      throw new Error(message);
    }
    return payload;
  }

  function setStatus(title, message, error = false) {
    statusBox.hidden = false;
    statusBox.classList.toggle("is-error", error);
    statusBox.querySelector(".status-spinner").hidden = !["queued", "downloading", "analyzing", "saving"].includes(title);
    statusTitle.textContent = title;
    statusMessage.textContent = message;
  }

  function clearStatus() {
    statusBox.hidden = true;
    statusBox.classList.remove("is-error");
  }

  function setConnected(connected, message = "") {
    const tokenMessage = $("#token-message");
    tokenMessage.textContent = message;
    tokenMessage.style.color = connected ? "#46765d" : "#9b594c";
    if (connected) {
      setTimeout(() => settingsDialog.close(), 250);
      loadPrivateData();
    }
  }

  function openSettings() {
    $("#access-token").value = state.token;
    $("#token-message").textContent = "";
    settingsDialog.showModal();
  }

  $("#settings-open").addEventListener("click", openSettings);
  $("#save-token").addEventListener("click", async () => {
    const candidate = $("#access-token").value.trim();
    if (!candidate) {
      setConnected(false, "Cole a chave configurada no servidor.");
      return;
    }
    const prior = state.token;
    state.token = candidate;
    try {
      await request("/api/summary");
      sessionStorage.setItem(tokenKey, candidate);
      setConnected(true, "Conectado ao seu acervo.");
    } catch (error) {
      state.token = prior;
      setConnected(false, error.message || "Não foi possível validar a chave.");
    }
  });
  $("#clear-token").addEventListener("click", () => {
    state.token = "";
    sessionStorage.removeItem(tokenKey);
    $("#access-token").value = "";
    setConnected(false, "Chave removida desta sessão.");
  });

  async function loadPrivateData() {
    if (!state.token) return;
    try {
      const [summary, pending] = await Promise.all([
        request("/api/summary"),
        request("/api/memories?status=pending&limit=300"),
      ]);
      state.pending = pending;
      $("#pending-badge").textContent = String(summary.pending);
      renderMemories(state.pending, $("#new-memories"), { review: true });
      $("#new-memory-count").textContent = String(state.pending.length);
    } catch (error) {
      if (/autoriz|chave|accesso/i.test(error.message)) {
        state.token = "";
        sessionStorage.removeItem(tokenKey);
        openSettings();
      }
    }
  }

  async function loadApproved() {
    try {
      const suffix = state.query ? `?query=${encodeURIComponent(state.query)}&limit=300` : "?limit=300";
      state.approved = await request(`/api/memories${suffix}`);
      renderMemories(state.approved, $("#approved-memories"), { review: false });
      $("#empty-library").hidden = state.approved.length > 0;
    } catch (error) {
      setStatus("Acesso necessário", error.message, true);
    }
  }

  function addText(parent, tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    parent.append(node);
    return node;
  }

  function renderMemories(memories, target, { review }) {
    target.replaceChildren();
    if (!memories.length) return;
    for (const memory of memories) {
      const card = document.createElement("article");
      card.className = "memory-card";
      const content = document.createElement("div");
      content.className = "memory-card-content";
      const topline = document.createElement("div");
      topline.className = "memory-topline";
      addText(topline, "strong", "memory-subject", memory.subject);
      const status = addText(topline, "span", `memory-status ${memory.status}`, {
        pending: "pendente",
        approved: "aprovada",
        rejected: "rejeitada",
      }[memory.status] || memory.status);
      status.setAttribute("aria-label", `Status: ${status.textContent}`);
      content.append(topline);
      addText(content, "p", "memory-fact", memory.fact);

      if (memory.tags?.length) {
        const tags = document.createElement("div");
        tags.className = "memory-meta";
        for (const tag of memory.tags.slice(0, 10)) addText(tags, "span", "memory-tag", tag);
        content.append(tags);
      }

      const meta = document.createElement("div");
      meta.className = "memory-source";
      if (memory.source_url) {
        const link = document.createElement("a");
        link.href = memory.source_url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = memory.article_title || "Fonte da Wikipédia";
        meta.append("Fonte: ", link);
      } else {
        meta.textContent = `Fonte: ${memory.source}`;
      }
      content.append(meta);
      card.append(content);

      if (review) {
        const actions = document.createElement("div");
        actions.className = "memory-actions";
        const reject = addText(actions, "button", "action-button reject", "Descartar");
        reject.type = "button";
        reject.addEventListener("click", () => updateMemory(memory.id, "rejected"));
        const approve = addText(actions, "button", "action-button approve", "Aprovar");
        approve.type = "button";
        approve.addEventListener("click", () => updateMemory(memory.id, "approved"));
        card.append(actions);
      }
      target.append(card);
    }
  }

  async function updateMemory(id, status) {
    try {
      await request(`/api/memories/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify({ status }),
      });
      await Promise.all([loadPrivateData(), loadApproved()]);
    } catch (error) {
      setStatus("Não foi possível atualizar", error.message, true);
    }
  }

  async function pollJob(jobId) {
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
      const busy = ["queued", "downloading", "analyzing", "saving"].includes(job.status);
      if (busy) {
        setStatus(job.status, job.message);
        state.pollTimer = setTimeout(() => pollJob(jobId), 2500);
        return;
      }
      submitButton.disabled = false;
      submitButton.querySelector("span:first-child").textContent = "Extrair memórias";
      if (job.status === "failed") {
        setStatus("Análise não concluída", job.error || job.message, true);
        return;
      }
      clearStatus();
      showResult(job.result);
      await Promise.all([loadPrivateData(), loadApproved()]);
    } catch (error) {
      submitButton.disabled = false;
      setStatus("Erro ao acompanhar a tarefa", error.message, true);
    }
  }

  function showResult(result) {
    if (!result?.analysis || !result?.article) return;
    const article = result.article;
    const analysis = result.analysis;
    $("#result-summary").textContent = analysis.abstractive_summary || analysis.extractive_summary;
    $("#source-link").href = article.canonical_url;
    $("#source-link").textContent = `Abrir ${article.title} ↗`;

    const claims = $("#central-claims");
    claims.replaceChildren();
    for (const claim of (analysis.central_claims || []).slice(0, 4)) addText(claims, "div", "claim-item", claim);

    const topics = $("#topics-list");
    topics.replaceChildren();
    for (const topic of (analysis.topics || []).slice(0, 12)) addText(topics, "span", "chip", topic);

    const entities = $("#entities-list");
    entities.replaceChildren();
    for (const entity of (analysis.entities || []).slice(0, 12)) {
      const chip = document.createElement("span");
      chip.className = "entity-chip";
      chip.append(document.createTextNode(entity.text));
      addText(chip, "small", "", entity.label);
      entities.append(chip);
    }

    $("#results-title").textContent = article.title;
    $("#results-section").hidden = false;
    $("#results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("#harvest-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.token) {
      openSettings();
      setStatus("Conecte seu acervo", "A chave de acesso é necessária para iniciar uma análise.", true);
      return;
    }
    const title = $("#article-title").value.trim();
    if (!title) return;
    submitButton.disabled = true;
    submitButton.querySelector("span:first-child").textContent = "Preparando…";
    clearStatus();
    try {
      const job = await request("/api/harvest", {
        method: "POST",
        body: JSON.stringify({
          title,
          abstractive: $("#abstractive").checked,
          embeddings: $("#embeddings").checked,
          max_paragraphs: 8,
        }),
      });
      setStatus(job.status, job.message);
      pollJob(job.id);
    } catch (error) {
      submitButton.disabled = false;
      submitButton.querySelector("span:first-child").textContent = "Extrair memórias";
      setStatus("Não foi possível iniciar", error.message, true);
      if (/não autorizado|chave|token/i.test(error.message)) openSettings();
    }
  });

  let searchTimer;
  $("#memory-search").addEventListener("input", (event) => {
    state.query = event.target.value.trim();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadApproved, 250);
  });

  $("#export-button").addEventListener("click", async () => {
    try {
      const pack = await request("/api/export");
      downloadJson(pack, "memorias-wikipedia.json");
    } catch (error) {
      setStatus("Exportação indisponível", error.message, true);
    }
  });

  function downloadJson(payload, filename) {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  $("#import-button").addEventListener("click", () => {
    if (!state.token) {
      openSettings();
      return;
    }
    $("#import-file").click();
  });

  $("#import-file").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.size > 5_000_000) {
      setStatus("Arquivo grande demais", "O pacote JSON deve ter até 5 MB.", true);
      return;
    }
    try {
      const pack = JSON.parse(await file.text());
      const result = await request("/api/import", { method: "POST", body: JSON.stringify(pack) });
      setStatus("Importação concluída", `${result.imported} memórias importadas; ${result.duplicates} duplicatas ignoradas.`);
      await Promise.all([loadPrivateData(), loadApproved()]);
    } catch (error) {
      setStatus("Não foi possível importar", error.message || "Verifique se o arquivo é um pacote JSON válido.", true);
    }
  });

  $("#load-more").addEventListener("click", loadApproved);

  loadApproved();
  if (state.token) loadPrivateData();
  else {
    setTimeout(openSettings, 400);
  }
})();

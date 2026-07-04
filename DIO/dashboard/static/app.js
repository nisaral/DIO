const MANAGER = window.DIO_MANAGER_URL || "http://127.0.0.1:8085";
let slopeChart = null;

async function fetchMetrics() {
  const res = await fetch(`${MANAGER}/debug/metrics`);
  if (!res.ok) throw new Error(`metrics ${res.status}`);
  return res.json();
}

function renderWorkers(data) {
  const el = document.getElementById("workers");
  const workers = data.workers || [];
  if (!workers.length) {
    el.innerHTML = "<p>No workers registered.</p>";
    return;
  }
  el.innerHTML = workers.map((w) => {
    const nlms = w.nlms || {};
    const vram = w.free_vram_mb ?? "?";
    const hot = vram < 4096 ? "hot" : "";
    return `<div class="worker-row">
      <span><strong>${w.id}</strong> · pending ${w.pending}</span>
      <span class="badge ${hot}">VRAM ${vram} MB · slope ${(nlms.fast_slope || 0).toFixed(2)}</span>
    </div>`;
  }).join("");
}

function renderDecision(data) {
  const d = data.last_decision;
  document.getElementById("decision").textContent = d
    ? JSON.stringify(d, null, 2)
    : "No routing decisions yet.";
}

function renderSlopeChart(workers) {
  const ctx = document.getElementById("slope-chart");
  const labels = workers.map((w) => w.id);
  const fast = workers.map((w) => (w.nlms && w.nlms.fast_slope) || 0);
  const slow = workers.map((w) => (w.nlms && w.nlms.slow_slope) || 0);
  if (!slopeChart) {
    slopeChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Fast slope", data: fast, backgroundColor: "#3dd68c" },
          { label: "Slow slope", data: slow, backgroundColor: "#4a9eff" },
        ],
      },
      options: { responsive: true, scales: { y: { beginAtZero: true } } },
    });
  } else {
    slopeChart.data.labels = labels;
    slopeChart.data.datasets[0].data = fast;
    slopeChart.data.datasets[1].data = slow;
    slopeChart.update();
  }
}

function appendLog(decisions) {
  const log = document.getElementById("routing-log");
  const tail = (decisions || []).slice(-8).reverse();
  log.innerHTML = tail.map((d) =>
    `<div class="line">${d.worker_id}: total=${d.total_ms.toFixed(0)}ms (exec=${d.exec_ms.toFixed(0)} wait=${d.wait_ms.toFixed(0)})</div>`
  ).join("") || "<div class='line'>—</div>";
}

async function poll() {
  try {
    const data = await fetchMetrics();
    document.getElementById("status").textContent = `Strategy: ${data.strategy || "?"} · ${new Date().toLocaleTimeString()}`;
    renderWorkers(data);
    renderDecision(data);
    renderSlopeChart(data.workers || []);
    appendLog(data.decisions);
  } catch (e) {
    document.getElementById("status").textContent = `Offline: ${e.message}`;
  }
}

document.getElementById("btn-vram").onclick = async () => {
  const worker_id = document.getElementById("chaos-worker").value || "w_0";
  await fetch(`${MANAGER}/debug/chaos/vram`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ worker_id, free_vram_mb: 1200 }),
  });
  poll();
};

document.getElementById("btn-burst").onclick = async () => {
  const reqs = Array.from({ length: 10 }, (_, i) =>
    fetch(`${MANAGER}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: `Burst ping ${i}`, model_id: "llama", tier: "small" }),
    })
  );
  await Promise.allSettled(reqs);
  poll();
};

document.getElementById("btn-send").onclick = async () => {
  const prompt = document.getElementById("prompt").value;
  const res = await fetch(`${MANAGER}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-DIO-Tier": "small" },
    body: JSON.stringify({
      model: "llama-3.2-3b",
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const text = await res.text();
  document.getElementById("response").textContent = text;
  poll();
};

setInterval(poll, 2000);
poll();
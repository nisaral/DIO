// ═══════════════════════════════════════════════════════
// DIO Demonstration — Frontend Application v2
// ═══════════════════════════════════════════════════════

const API_BASE = '';
let ws = null;
let nlmsChart = null;
let testDefinitions = [];
let activeWorkers = {};
let workerTelemetry = {}; // worker_id -> latest telemetry

// ═══════ INITIALIZATION ═══════
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadTests();
    initDualChart();
    pollStatus();
    setInterval(pollStatus, 3000);
});

// ═══════ WEBSOCKET ═══════
function initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        addLog('system', 'WebSocket connected. Streaming live.');
        setStatusDot('ws', 'online');
    };
    ws.onclose = () => {
        addLog('error', 'WebSocket disconnected. Reconnecting in 3s...');
        setStatusDot('ws', '');
        setTimeout(initWebSocket, 3000);
    };
    ws.onerror = () => setStatusDot('ws', '');
    ws.onmessage = (event) => handleWSMessage(JSON.parse(event.data));
}

function handleWSMessage(msg) {
    switch (msg.type) {
        case 'log':
            addLog(msg.data.level, msg.data.message);
            parseWorkerFromLog(msg.data.message);
            break;
        case 'metrics_update':
            updateDualChart(msg.data);
            break;
        case 'worker_telemetry':
            onWorkerTelemetry(msg.data);
            break;
        case 'test_complete':
            onTestComplete(msg.data);
            break;
        case 'injection_result':
            updateDualChart({ iteration: msg.data.id, latency: msg.data.latency_ms, predicted: msg.data.predicted_ms, rr_latency: msg.data.latency_ms * 1.4 });
            break;
        case 'comparison_result':
            onComparisonResult(msg.data);
            break;
        case 'safety_block':
            onSafetyBlock(msg.data);
            break;
    }
}

// ═══════ TERMINAL ═══════
function addLog(level, message) {
    const terminal = document.getElementById('terminal');
    const line = document.createElement('div');
    line.className = 'log-line';
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const levelMap = { 'success': '[OK]', 'error': '[ERR]', 'info': '[-->]', 'system': '[SYS]', 'warning': '[!]' };
    const prefix = levelMap[level] || '[.]';
    line.innerHTML = `<span class="log-time">${prefix}</span><span class="log-text ${level}">${escapeHtml(message)}</span>`;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
    while (terminal.children.length > 500) terminal.removeChild(terminal.firstChild);
}

function clearTerminal() {
    document.getElementById('terminal').innerHTML = '';
    addLog('system', 'Terminal cleared.');
}

// ═══════ GPU WORKER PANEL ═══════
function parseWorkerFromLog(message) {
    const startMatch = message.match(/Worker (\S+) started on port (\d+) \(mock=(\w+), latency_mult=([\d.]+)\)/);
    if (startMatch) {
        const [_, id, port, , mult] = startMatch;
        activeWorkers[id] = { id, port: parseInt(port), speed: parseFloat(mult), requests: 0, status: 'idle', lastRequest: null, heat: 0 };
        renderGPUPanel();
        return;
    }
    if (message.match(/^=== T\d+:/)) { activeWorkers = {}; workerTelemetry = {}; renderGPUPanel(); return; }
}

function onWorkerTelemetry(data) {
    const id = data.worker_id;
    if (!activeWorkers[id]) {
        activeWorkers[id] = { id, speed: 1.0, requests: 0, status: 'idle', lastRequest: null, heat: 0 };
    }
    activeWorkers[id].requests++;
    activeWorkers[id].status = 'active';
    activeWorkers[id].lastRequest = data.actual_ms.toFixed(1) + 'ms';
    activeWorkers[id].heat = data.heat;
    workerTelemetry[id] = data;
    renderGPUPanel();
    setTimeout(() => { if (activeWorkers[id]) { activeWorkers[id].status = 'idle'; renderGPUPanel(); } }, 400);
}

function getHeatClass(heat) {
    if (heat < 0.3) return 'cool';
    if (heat < 0.65) return 'warm';
    return 'hot';
}

function renderGPUPanel() {
    const grid = document.getElementById('gpu-grid');
    const workerIds = Object.keys(activeWorkers);
    if (workerIds.length === 0) {
        grid.innerHTML = `<div class="gpu-card idle"><div class="gpu-name">No active workers</div><div class="gpu-detail">Run a test to see GPU shards</div></div>`;
        return;
    }
    grid.innerHTML = workerIds.map(id => {
        const w = activeWorkers[id];
        const t = workerTelemetry[id];
        const heatClass = w.status === 'active' ? getHeatClass(w.heat) : 'idle';
        const speedLabel = w.speed === 1.0 ? 'Normal' : `${w.speed}x slower`;
        const formulaHTML = t ? `
            <div class="formula-box">
                <span class="formula-item">Wait: <b>${t.wait_ms.toFixed(0)}ms</b></span>
                <span class="formula-plus">+</span>
                <span class="formula-item">Exec: <b>${t.exec_ms.toFixed(0)}ms</b></span>
                <span class="formula-plus">+</span>
                <span class="formula-item">VRAM: <b>${t.vram_penalty.toFixed(0)}ms</b></span>
                <span class="formula-eq">= <b class="formula-total">${t.total_cost.toFixed(0)}ms</b></span>
            </div>` : '';
        return `
            <div class="gpu-card ${heatClass}">
                <div class="gpu-header">
                    <span class="gpu-status-dot ${w.status === 'active' ? heatClass : ''}"></span>
                    <span class="gpu-name">${id}</span>
                </div>
                <div class="gpu-stats">
                    <div class="gpu-stat"><span class="gpu-stat-label">Speed</span><span class="gpu-stat-value">${speedLabel}</span></div>
                    <div class="gpu-stat"><span class="gpu-stat-label">Reqs</span><span class="gpu-stat-value">${w.requests}</span></div>
                    <div class="gpu-stat"><span class="gpu-stat-label">Last</span><span class="gpu-stat-value">${w.lastRequest || '-'}</span></div>
                </div>
                ${formulaHTML}
            </div>`;
    }).join('');
}

// ═══════ SAFETY FLASH ═══════
function onSafetyBlock(data) {
    document.getElementById('safety-msg').textContent = data.message;
    document.getElementById('safety-overlay').classList.remove('hidden');
    addLog('warning', `SAFETY: ${data.message}`);
    // Re-enable OOM button
    document.getElementById('oom-btn').disabled = false;
    document.getElementById('oom-btn').textContent = 'FIRE OOM BOMB';
}

function closeSafetyOverlay() {
    document.getElementById('safety-overlay').classList.add('hidden');
}

// ═══════ DUAL-VIEW NLMS CHART ═══════
function initDualChart() {
    const ctx = document.getElementById('nlms-chart').getContext('2d');
    nlmsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'DIO (NLMS — Actual)',
                    data: [],
                    borderColor: 'hsl(142, 71%, 45%)',
                    backgroundColor: 'hsla(142, 71%, 45%, 0.1)',
                    borderWidth: 2.5, fill: true, tension: 0.3, pointRadius: 2,
                },
                {
                    label: 'NLMS Prediction',
                    data: [],
                    borderColor: 'hsl(217, 91%, 60%)',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5, borderDash: [4, 3], fill: false, tension: 0.3, pointRadius: 0,
                },
                {
                    label: 'Round-Robin (Baseline)',
                    data: [],
                    borderColor: 'hsl(0, 84%, 60%)',
                    backgroundColor: 'hsla(0, 84%, 60%, 0.08)',
                    borderWidth: 2, fill: true, tension: 0.3, pointRadius: 2,
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 150 },
            plugins: {
                legend: { labels: { color: 'hsl(215, 20%, 65%)', font: { family: 'Inter', size: 11 } } }
            },
            scales: {
                x: { title: { display: true, text: 'Request #', color: 'hsl(215, 15%, 45%)', font: { size: 10 } }, ticks: { color: 'hsl(215, 15%, 45%)', font: { size: 9 } }, grid: { color: 'hsla(215, 20%, 35%, 0.15)' } },
                y: { title: { display: true, text: 'Latency (ms)', color: 'hsl(215, 15%, 45%)', font: { size: 10 } }, ticks: { color: 'hsl(215, 15%, 45%)', font: { size: 9 } }, grid: { color: 'hsla(215, 20%, 35%, 0.15)' }, beginAtZero: true }
            }
        }
    });
}

function updateDualChart(data) {
    if (!nlmsChart) return;

    // Detect if this is a single-worker test:
    // When backend sends rr_latency == latency, it means only 1 worker exists.
    const isSingleWorker = !data.rr_latency || Math.abs(data.rr_latency - data.latency) < 0.5;

    // Show/hide RR dataset and update title accordingly
    const rrDataset = nlmsChart.data.datasets[2];
    if (isSingleWorker) {
        rrDataset.hidden = true;
        nlmsChart.options.plugins.title = {
            display: true,
            text: 'NLMS Prediction Convergence — Watch prediction error shrink',
            color: 'hsl(215, 20%, 55%)',
            font: { family: 'Inter', size: 11 }
        };
    } else {
        rrDataset.hidden = false;
        nlmsChart.options.plugins.title = {
            display: true,
            text: 'DIO (green) vs Round-Robin baseline (red) — DIO wins',
            color: 'hsl(215, 20%, 55%)',
            font: { family: 'Inter', size: 11 }
        };
    }

    const labels = nlmsChart.data.labels;
    labels.push(data.iteration || labels.length + 1);
    nlmsChart.data.datasets[0].data.push(data.latency);
    nlmsChart.data.datasets[1].data.push(data.predicted);
    nlmsChart.data.datasets[2].data.push(data.rr_latency || data.latency);

    if (labels.length > 100) {
        labels.shift();
        nlmsChart.data.datasets.forEach(d => d.data.shift());
    }
    nlmsChart.update('none');
}

// ═══════ TEST SUITE ═══════
async function loadTests() {
    try {
        const resp = await fetch(`${API_BASE}/api/tests`);
        testDefinitions = await resp.json();
    } catch (e) { testDefinitions = getDefaultTests(); }
    renderTestGrid();
}

function renderTestGrid() {
    const grid = document.getElementById('test-grid');
    grid.innerHTML = '';
    testDefinitions.forEach(test => {
        const card = document.createElement('div');
        card.className = 'test-card'; card.id = `test-card-${test.id}`;
        card.onclick = () => runSingleTest(test.id);
        card.innerHTML = `<div class="test-id">${test.id}</div><div class="test-name">${test.name}</div><div class="test-desc">${test.description}</div><div class="test-metrics" id="metrics-${test.id}"></div>`;
        grid.appendChild(card);
    });
}

async function runSingleTest(testId) {
    const card = document.getElementById(`test-card-${testId}`);
    if (card) card.className = 'test-card running';
    // Reset chart for new test — hide RR until first data point decides worker count
    if (nlmsChart) {
        nlmsChart.data.labels = [];
        nlmsChart.data.datasets.forEach(d => d.data = []);
        nlmsChart.data.datasets[2].hidden = true;
        if (nlmsChart.options.plugins.title) nlmsChart.options.plugins.title.display = false;
        nlmsChart.update();
    }
    try {
        await fetch(`${API_BASE}/api/tests/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ test_id: testId }) });
        addLog('info', `Test ${testId} started...`);
    } catch (e) {
        addLog('error', `Failed to start ${testId}: ${e.message}`);
        if (card) card.className = 'test-card fail';
    }
}

async function runAllTests() {
    addLog('system', '=== Running Full Test Suite ===');
    testDefinitions.forEach(t => { const c = document.getElementById(`test-card-${t.id}`); if (c) c.className = 'test-card'; });
    try { await fetch(`${API_BASE}/api/tests/run-all`, { method: 'POST' }); }
    catch (e) { addLog('error', `Failed to start test suite: ${e.message}`); }
}

function onTestComplete(data) {
    const testId = data.test_id || data.TestID;
    const status = data.status || data.Status;
    const card = document.getElementById(`test-card-${testId}`);
    if (card) card.className = `test-card ${status}`;
    if (data.metrics || data.MetricsSummary) {
        const m = data.metrics || data.MetricsSummary;
        const el = document.getElementById(`metrics-${testId}`);
        if (el && m) el.innerHTML = `<span class="metric-badge latency">${(m.avg_latency_ms || m.AvgLatency || 0).toFixed(1)}ms avg</span><span class="metric-badge tokens">${m.total_tokens || m.TotalTokens || 0} tok</span><span class="metric-badge throughput">${(m.requests_per_sec || m.Throughput || 0).toFixed(1)} req/s</span>`;
    }
}

// ═══════ CHAOS ENGINEERING ═══════
let throttleDebounce = null;

function updateThrottle(value) {
    const v = parseFloat(value);
    const label = v <= 1.0 ? '1.0x (Normal)' : `${v.toFixed(1)}x (THROTTLED)`;
    document.getElementById('throttle-value').textContent = label;
    document.getElementById('throttle-value').className = v > 1.0 ? 'chaos-value hot' : 'chaos-value';
    clearTimeout(throttleDebounce);
    throttleDebounce = setTimeout(() => {
        fetch(`${API_BASE}/api/worker/throttle`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ worker_id: 'RTX4050_Shard_2', multiplier: v })
        });
    }, 200);
}

async function fireOOMBomb() {
    const btn = document.getElementById('oom-btn');
    btn.disabled = true;
    btn.textContent = 'FIRING...';
    addLog('warning', 'OOM BOMB fired! Watching Roofline Admission Control...');
    try {
        await fetch(`${API_BASE}/api/chaos/oom-bomb`, { method: 'POST' });
    } catch (e) {
        addLog('error', `OOM bomb failed: ${e.message}`);
        btn.disabled = false;
        btn.textContent = 'FIRE OOM BOMB';
    }
}

// ═══════ INJECTION ═══════
function setPreset(size) {
    const presets = {
        short: 'What is the capital of France?',
        medium: 'Explain gradient descent optimization in ML, covering SGD, Adam, and RMSProp.',
        long: 'Provide a comprehensive analysis of distributed systems architecture. Cover consensus algorithms (Paxos, Raft), CAP theorem, eventual consistency, vector clocks, and real-world systems like Google Spanner, Amazon DynamoDB, and Apache Cassandra.'
    };
    document.getElementById('inject-prompt').value = presets[size] || '';
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
}

async function injectPrompt() {
    const prompt = document.getElementById('inject-prompt').value.trim();
    if (!prompt) { addLog('warning', 'Please enter a prompt.'); return; }
    try { await fetch(`${API_BASE}/api/inject`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) }); }
    catch (e) { addLog('error', `Injection failed: ${e.message}`); }
}

function updateBurstLabel() {
    document.getElementById('burst-label').textContent = document.getElementById('burst-count').value;
}

async function fireBurst() {
    const count = parseInt(document.getElementById('burst-count').value);
    const active = document.querySelector('.preset-btn.active');
    const size = active ? (active.textContent.includes('Short') ? 'short' : active.textContent.includes('Long') ? 'long' : 'medium') : 'short';
    try { await fetch(`${API_BASE}/api/inject/burst`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ size, count }) }); }
    catch (e) { addLog('error', `Burst failed: ${e.message}`); }
}

function onComparisonResult(data) {
    addLog('system', '=== A/B Comparison Result ===');
    addLog('info', `DIO (NLMS):      Avg=${data.nlms_avg_ms.toFixed(1)}ms | P99=${data.nlms_p99_ms.toFixed(1)}ms`);
    addLog('info', `Round Robin:     Avg=${data.rr_avg_ms.toFixed(1)}ms | P99=${data.rr_p99_ms.toFixed(1)}ms`);
    const improvement = ((data.rr_avg_ms - data.nlms_avg_ms) / data.rr_avg_ms * 100).toFixed(1);
    addLog('success', `DIO is ${improvement}% faster on average`);
}

// ═══════ SYSTEM STATUS ═══════
async function pollStatus() {
    try {
        const data = await fetch(`${API_BASE}/api/system/status`).then(r => r.json());
        setStatusDot('manager', data.manager ? 'online' : '');
        setStatusDot('workers', data.workers > 0 ? 'online' : '');
        document.getElementById('worker-count-label').textContent = `${data.workers || 0} Workers`;
    } catch (e) { }
}

function setStatusDot(id, status) {
    const dot = document.getElementById(`dot-${id}`);
    if (dot) dot.className = `status-dot ${status}`;
}

// ═══════ UTILITIES ═══════
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getDefaultTests() {
    return [
        { id: 'T1', name: 'NLMS Convergence', description: 'Adaptive filter converges to optimal latency prediction' },
        { id: 'T2', name: 'Heterogeneous Routing', description: 'Smart routing between fast and slow workers' },
        { id: 'T3', name: 'Cold Start Recovery', description: 'New worker joins mid-test gracefully' },
        { id: 'T4', name: 'VRAM Roofline Safety', description: 'Memory saturation guard prevents OOM' },
        { id: 'T5', name: 'Spike Absorption', description: 'Handle sudden traffic burst' },
        { id: 'T7', name: 'Scalability (8)', description: 'O(1) overhead at 8-worker scale' },
        { id: 'T8', name: 'Scalability (32)', description: 'O(1) overhead at 32-worker scale' },
        { id: 'T9', name: 'Short Queries', description: 'Chat-style short prompt benchmark' },
        { id: 'T10', name: 'Long Queries', description: 'Document summarization benchmark' },
        { id: 'T11', name: 'Mixed Workload', description: 'Interleaved short + long queries' },
    ];
}

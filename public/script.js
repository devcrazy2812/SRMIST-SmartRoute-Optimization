/* ═══════════════════════════════════════════════════════
   SmartRoute SRMIST — Agentic AI v4 Application Logic
   Created by: devcrazy AKA Abhay Goyal
   ═══════════════════════════════════════════════════════ */

let map, tileLayer;
let nodeData = {}, markers = {}, routeLines = [];
let currentPath = [], pickMode = 'START';
let routeCount = 0;
let activityLog = [];

const TILES = {
    light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    dark:  'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
};

// ── Init ───────────────────────────────────────────────
window.addEventListener('load', async () => {
    initMap();
    await loadNodes();
    wireUI();
    updateTrafficStatus();
    loadStats();
});

function initMap() {
    const isDark = document.documentElement.dataset.theme === 'dark';
    map = L.map('map', { zoomControl: true }).setView([12.825, 80.046], 16);
    tileLayer = L.tileLayer(TILES[isDark ? 'dark' : 'light'], {
        attribution: '&copy; OpenStreetMap & CartoDB', maxZoom: 19,
    }).addTo(map);
}

async function loadNodes() {
    try {
        const res = await fetch('/api/nodes');
        const data = await res.json();
        const startSel = document.getElementById('startNode');
        const endSel   = document.getElementById('endNode');
        const prefSel  = document.getElementById('preference');

        Object.entries(data.nodes).forEach(([id, name]) => {
            startSel.innerHTML += `<option value="${id}">${name}</option>`;
            endSel.innerHTML   += `<option value="${id}">${name}</option>`;
        });

        data.preferences.forEach(p => {
            const label = p.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            prefSel.innerHTML += `<option value="${p}">${label}</option>`;
        });

        // Draw edge connections (faint)
        data.map_data.edges.forEach(edge => {
            const a = data.map_data.nodes.find(n => n.id === edge.from);
            const b = data.map_data.nodes.find(n => n.id === edge.to);
            if (a && b) {
                L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {
                    color: '#94a3b8', weight: 1, opacity: 0.3, dashArray: '4,6'
                }).addTo(map);
            }
        });

        // Plot markers
        data.map_data.nodes.forEach(n => {
            nodeData[n.id] = { lat: n.lat, lon: n.lon, name: n.name };
            const icon = makeIcon('#6c63ff', 10);
            const m = L.marker([n.lat, n.lon], { icon }).addTo(map);
            m.bindTooltip(n.name, { direction: 'top', offset: [0, -8], className: 'tip-label' });
            m.on('click', () => onMarkerClick(n.id));
            markers[n.id] = m;
        });
    } catch (e) {
        console.error('Failed to load nodes:', e);
    }
}

// ── Markers ────────────────────────────────────────────
function makeIcon(color, size) {
    return L.divIcon({
        className: '',
        html: `<div style="width:${size}px;height:${size}px;background:${color};
                border-radius:50%;box-shadow:0 0 ${size}px ${color}80;
                border:2px solid #fff;"></div>`,
        iconSize: [size, size],
    });
}

function resetAllMarkers() {
    Object.values(markers).forEach(m => m.setIcon(makeIcon('#6c63ff', 10)));
}

function onMarkerClick(id) {
    if (pickMode === 'START') {
        resetAllMarkers(); clearRoute();
        document.getElementById('startNode').value = id;
        markers[id].setIcon(makeIcon('#6c63ff', 14));
        pickMode = 'END';
    } else if (pickMode === 'END') {
        if (id === document.getElementById('startNode').value) return;
        document.getElementById('endNode').value = id;
        markers[id].setIcon(makeIcon('#e53e3e', 14));
        pickMode = 'DONE';
    } else {
        clearRoute(); resetAllMarkers();
        document.getElementById('startNode').value = id;
        document.getElementById('endNode').value = '';
        markers[id].setIcon(makeIcon('#6c63ff', 14));
        pickMode = 'END';
    }
}

// ── Wire UI ────────────────────────────────────────────
function wireUI() {
    document.getElementById('startNode').addEventListener('change', () => { pickMode = 'END'; clearRoute(); });
    document.getElementById('endNode').addEventListener('change', () => { pickMode = 'DONE'; clearRoute(); });
    document.getElementById('computeBtn').addEventListener('click', computeRoute);
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Neural Core drawer
    document.getElementById('openQTable').addEventListener('click', () => {
        document.getElementById('drawer').classList.add('open');
        loadDrawerQTable();
    });
    document.getElementById('closeDrawer').addEventListener('click', () => {
        document.getElementById('drawer').classList.remove('open');
    });

    // Emergency Replan
    document.getElementById('emergencyBtn').addEventListener('click', () => {
        if (currentPath.length < 2) { alert('No active route. Generate a route first!'); return; }
        const pref = document.getElementById('preference').value || 'fastest';
        const s = document.getElementById('startNode').value;
        const e = document.getElementById('endNode').value;
        if (s && e) { computeRoute(); addActivity('⚡ Emergency replan triggered'); }
    });

    // FAB button
    document.getElementById('fabBtn').addEventListener('click', () => {
        switchTab('srmist');
    });
}

// ── Tab Switching ──────────────────────────────────────
const VIEW_MAP = {
    planner:   'plannerView',
    srmist:    'srmistView',
    dashboard: 'dashboardView',
};

function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    const tabEl = document.querySelector(`.tab[data-tab="${name}"]`);
    if (tabEl) tabEl.classList.add('active');

    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const viewId = VIEW_MAP[name];
    if (viewId) document.getElementById(viewId).classList.add('active');

    if (name === 'dashboard') loadDashboard();
    if (name === 'srmist') loadSrmistStats();
    if (name === 'planner') setTimeout(() => map.invalidateSize(), 100);
}

// ── Quick Route (from SRMIST tab) ──────────────────────
function quickRoute(from, to) {
    switchTab('planner');
    setTimeout(() => {
        document.getElementById('startNode').value = from;
        document.getElementById('endNode').value = to;
        document.getElementById('preference').value = 'fastest';
        resetAllMarkers();
        if (markers[from]) markers[from].setIcon(makeIcon('#6c63ff', 14));
        if (markers[to])   markers[to].setIcon(makeIcon('#e53e3e', 14));
        pickMode = 'DONE';
        computeRoute();
    }, 200);
}

// ── SRMIST Tab Stats ───────────────────────────────────
async function loadSrmistStats() {
    try {
        const res = await fetch('/api/stats');
        const s = await res.json();
        document.getElementById('srmFeedback').textContent = s.total_entries;
    } catch {}
}

// ── Theme Toggle ───────────────────────────────────────
function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.dataset.theme === 'dark';
    if (isDark) {
        html.removeAttribute('data-theme');
        document.getElementById('themeToggle').textContent = '🌙 Dark Mode';
    } else {
        html.dataset.theme = 'dark';
        document.getElementById('themeToggle').textContent = '☀️ Light Mode';
    }
    map.removeLayer(tileLayer);
    tileLayer = L.tileLayer(TILES[isDark ? 'light' : 'dark'], {
        attribution: '&copy; OpenStreetMap & CartoDB', maxZoom: 19,
    }).addTo(map);
}

// ── Traffic Status ─────────────────────────────────────
function updateTrafficStatus() {
    const hour = new Date().getHours();
    const el = document.getElementById('trafficInfo');
    const sub = document.getElementById('trafficSub');
    if (hour >= 8 && hour <= 10) {
        el.innerHTML = '<span class="traffic-icon">🔴</span><span>Heavy (Morning Rush)</span>';
        sub.textContent = 'Traffic multiplier: 2.5×';
    } else if (hour >= 16 && hour <= 18) {
        el.innerHTML = '<span class="traffic-icon">🟠</span><span>Moderate (Evening Rush)</span>';
        sub.textContent = 'Traffic multiplier: 2.0×';
    } else if (hour >= 12 && hour <= 13) {
        el.innerHTML = '<span class="traffic-icon">🟡</span><span>Moderate (Lunch Hour)</span>';
        sub.textContent = 'Traffic multiplier: 1.5×';
    } else {
        el.innerHTML = '<span class="traffic-icon">🟢</span><span>Normal Flow</span>';
        sub.textContent = 'Peak hours: 8-10 AM, 4-6 PM';
    }
}

// ── Route Computation ──────────────────────────────────
async function computeRoute() {
    const s = document.getElementById('startNode').value;
    const e = document.getElementById('endNode').value;
    const p = document.getElementById('preference').value;
    if (!s || !e || !p) return alert('Please select Origin, Destination, and Strategy.');
    if (s === e) return alert('Origin and Destination must be different.');

    const btn = document.getElementById('computeBtn');
    btn.disabled = true;

    const term = document.getElementById('terminal');
    const line = document.getElementById('termLine');
    term.classList.remove('hidden');

    const steps = [
        '> Loading campus graph (16 nodes, 27 edges)...',
        '> Computing A* with Haversine heuristic...',
        '> Evaluating 4-factor cost: dist + time + traffic + RL...',
        '> Applying Q-Learning Bellman bias...',
        '> ✅ Route optimized.',
    ];
    for (const msg of steps) { line.textContent = msg; await sleep(350); }

    try {
        const res = await fetch('/api/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ startNode: s, endNode: e, preference: p }),
        });
        if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }

        const data = await res.json();
        currentPath = data.path;
        routeCount++;
        addActivity(`Route: ${data.path_names[0]} → ${data.path_names[data.path_names.length-1]} [${p}] Cost: ${data.total_cost.toFixed(1)}`);
        showResult(data);
        drawRoute(data.path);
        loadStats();
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        term.classList.add('hidden');
    }
}

function showResult(data) {
    const card = document.getElementById('resultCard');
    card.classList.remove('hidden');
    document.getElementById('fbMsg').classList.add('hidden');

    // Stats
    document.getElementById('rDist').textContent = data.total_distance_km.toFixed(1);
    document.getElementById('rTime').textContent = data.total_time_min;
    const tLvl = data.avg_traffic;
    document.getElementById('rTraffic').textContent = tLvl > 0.6 ? 'High' : tLvl > 0.3 ? 'Med' : 'Low';
    document.getElementById('rTraffic').style.color = tLvl > 0.6 ? 'var(--red)' : tLvl > 0.3 ? 'var(--amber)' : 'var(--green)';
    document.getElementById('rCost').textContent = data.total_cost.toFixed(1);

    // Explanation
    document.getElementById('explainText').textContent = data.explanation;

    // Path steps
    const steps = document.getElementById('pathSteps');
    steps.innerHTML = '';
    data.path_names.forEach(name => {
        steps.innerHTML += `<div class="step-item"><span class="step-dot"></span>${name}</div>`;
    });

    // Segment breakdown (right sidebar)
    const brCard = document.getElementById('breakdownCard');
    brCard.classList.remove('hidden');
    const segBody = document.getElementById('segBody');
    segBody.innerHTML = '';
    data.breakdown.forEach(s => {
        const qCls = s.q_value > 0 ? 'qp' : s.q_value < 0 ? 'qn' : '';
        segBody.innerHTML += `<tr>
            <td>${(s.from_name||s.from).split(' ').slice(0,2).join(' ')}</td>
            <td>${(s.to_name||s.to).split(' ').slice(0,2).join(' ')}</td>
            <td>${s.distance}</td>
            <td>${s.time}m</td>
            <td class="${qCls}">${s.q_value > 0 ? '+' : ''}${s.q_value.toFixed(2)}</td>
            <td>${s.base_cost.toFixed(1)}</td>
        </tr>`;
    });

    document.getElementById('fbGood').disabled = false;
    document.getElementById('fbBad').disabled  = false;
}

function drawRoute(ids) {
    clearRoute();
    const pts = ids.map(id => [nodeData[id].lat, nodeData[id].lon]);
    const glow = L.polyline(pts, { color: '#6c63ff', weight: 7, opacity: 0.2 }).addTo(map);
    const core = L.polyline(pts, { color: '#6c63ff', weight: 3, opacity: 0.85 }).addTo(map);
    routeLines.push(glow, core);
    map.fitBounds(glow.getBounds(), { padding: [60, 60] });
}

function clearRoute() {
    routeLines.forEach(l => map.removeLayer(l));
    routeLines = [];
}

// ── Feedback ───────────────────────────────────────────
async function sendFeedback(reward) {
    document.getElementById('fbGood').disabled = true;
    document.getElementById('fbBad').disabled  = true;

    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: currentPath, reward }),
        });
        const msg = document.getElementById('fbMsg');
        msg.textContent = reward > 0
            ? '✅ Positive reward applied — AI will favor this route.'
            : '⚠️ Negative reward applied — AI will avoid this route.';
        msg.style.color = reward > 0 ? 'var(--green)' : 'var(--red)';
        msg.classList.remove('hidden');
        addActivity(reward > 0 ? '👍 Positive feedback submitted' : '👎 Negative feedback submitted');
        loadStats();
    } catch {
        alert('Feedback failed.');
        document.getElementById('fbGood').disabled = false;
        document.getElementById('fbBad').disabled  = false;
    }
}

// ── Stats Loading ──────────────────────────────────────
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const s = await res.json();
        // Right sidebar
        document.getElementById('qEntries').textContent = s.total_entries;
    } catch {}
}

// ── Dashboard ──────────────────────────────────────────
async function loadDashboard() {
    try {
        const [statsRes, qtRes] = await Promise.all([
            fetch('/api/stats'), fetch('/api/qtable')
        ]);
        const stats = await statsRes.json();
        const qt = await qtRes.json();

        document.getElementById('dsRoutes').textContent = routeCount;
        document.getElementById('dsQEntries').textContent = stats.total_entries;
        document.getElementById('dsPositive').textContent = stats.positive_biases;
        document.getElementById('dsNegative').textContent = stats.negative_biases;
        document.getElementById('dsStates').textContent = stats.states_explored;

        // Q-Table
        const tbody = document.getElementById('dashQBody');
        tbody.innerHTML = '';
        const entries = Object.entries(qt);
        if (!entries.length) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--txt3)">No Q-data yet. Provide feedback to train the AI!</td></tr>';
        } else {
            entries.forEach(([state, actions]) => {
                Object.entries(actions).forEach(([action, v]) => {
                    const cls = v > 0 ? 'qp' : v < 0 ? 'qn' : '';
                    const sign = v > 0 ? '+' : '';
                    tbody.innerHTML += `<tr>
                        <td>${state.replace(/_/g, ' ')}</td>
                        <td>${action.replace(/_/g, ' ')}</td>
                        <td class="${cls}">${sign}${v.toFixed(4)}</td>
                    </tr>`;
                });
            });
        }

        // Activity feed
        const feed = document.getElementById('activityFeed');
        if (activityLog.length) {
            feed.innerHTML = activityLog.slice(-10).reverse()
                .map(a => `<div class="activity-item">${a}</div>`).join('');
        }
    } catch {}
}

// ── Drawer Q-Table ─────────────────────────────────────
async function loadDrawerQTable() {
    try {
        const [statsRes, qtRes] = await Promise.all([
            fetch('/api/stats'), fetch('/api/qtable')
        ]);
        const stats = await statsRes.json();
        const qt = await qtRes.json();

        document.getElementById('drawerStats').innerHTML = `
            <div class="rl-stat-item"><span class="val">${stats.total_entries}</span><span class="lbl">Q-Entries</span></div>
            <div class="rl-stat-item"><span class="val">${stats.states_explored}</span><span class="lbl">States</span></div>
            <div class="rl-stat-item"><span class="val" style="color:var(--green)">${stats.positive_biases}</span><span class="lbl">Positive</span></div>
            <div class="rl-stat-item"><span class="val" style="color:var(--red)">${stats.negative_biases}</span><span class="lbl">Negative</span></div>
        `;

        const tbody = document.getElementById('drawerQBody');
        tbody.innerHTML = '';
        const entries = Object.entries(qt);
        if (!entries.length) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--txt3)">Train the AI by providing feedback!</td></tr>';
        } else {
            entries.forEach(([state, actions]) => {
                Object.entries(actions).forEach(([action, v]) => {
                    const cls = v > 0 ? 'qp' : v < 0 ? 'qn' : '';
                    const sign = v > 0 ? '+' : '';
                    tbody.innerHTML += `<tr>
                        <td>${state.replace(/_/g, ' ')}</td>
                        <td>${action.replace(/_/g, ' ')}</td>
                        <td class="${cls}">${sign}${v.toFixed(4)}</td>
                    </tr>`;
                });
            });
        }
    } catch {
        document.getElementById('drawerQBody').innerHTML = '<tr><td colspan="3">Connection error</td></tr>';
    }
}

// ── Activity Log ───────────────────────────────────────
function addActivity(msg) {
    const time = new Date().toLocaleTimeString();
    activityLog.push(`<strong>${time}</strong> — ${msg}`);
}

// ── Utility ────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

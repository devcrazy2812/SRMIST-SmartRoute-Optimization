let currentPath = [];
let map;
let routeLayers = [];
let nodeData = {};

// Initialize application
async function loadConfig() {
    initMap();
    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();
        
        const startSelect = document.getElementById('startNode');
        const endSelect = document.getElementById('endNode');
        const prefSelect = document.getElementById('preference');
        
        // Populate Selects
        for (const [id, name] of Object.entries(data.nodes)) {
            startSelect.innerHTML += `<option value="${id}">${name}</option>`;
            endSelect.innerHTML += `<option value="${id}">${name}</option>`;
        }

        data.preferences.forEach(pref => {
            const displayPref = pref.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            prefSelect.innerHTML += `<option value="${pref}">${displayPref}</option>`;
        });

        // Store precise lat/lon map data from our simulated database
        const mapNodes = data.map_data.nodes;
        mapNodes.forEach(n => {
            nodeData[n.id] = { lat: n.lat, lon: n.lon, name: n.name };
            
            // Plot initial stationary markers on the map
            L.marker([n.lat, n.lon])
             .bindPopup(`<b>${n.name}</b>`)
             .addTo(map);
        });

    } catch (e) {
        console.error("Failed to fetch node config:", e);
    }
}

function initMap() {
    // Center roughly on SRM KTR coordinates
    map = L.map('map').setView([12.8210, 80.0420], 16);

    // Load CartoDB Dark Matter tiles for premium dark theme look
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);
}

document.getElementById('calculateBtn').addEventListener('click', async () => {
    const startNode = document.getElementById('startNode').value;
    const endNode = document.getElementById('endNode').value;
    const pref = document.getElementById('preference').value;

    if (!startNode || !endNode || !pref) return alert("Select all routing options.");
    if (startNode === endNode) return alert("Origin and Destination cannot be the same.");

    const btn = document.getElementById('calculateBtn');
    btn.innerHTML = 'Computing...'; btn.disabled = true;

    try {
        const response = await fetch('/api/route', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ startNode, endNode, preference: pref })
        });

        if (!response.ok) throw new Error((await response.json()).detail);

        const result = await response.json();
        currentPath = result.path; 
        
        displayResult(result);
        drawRouteOnMap(result.path);
        
    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        btn.innerHTML = 'Compute Neural Route'; btn.disabled = false;
    }
});

function displayResult(result) {
    document.getElementById('resultsSection').classList.remove('hidden');
    document.getElementById('feedbackResult').classList.add('hidden');
    
    document.getElementById('tripCost').innerText = `Cost Index: ${result.total_cost.toFixed(2)}`;
    document.getElementById('explanationText').innerText = result.explanation;

    const timeline = document.getElementById('routeTimeline');
    timeline.innerHTML = '';
    result.path_names.forEach((name, index) => {
        timeline.innerHTML += `
            <div class="timeline-item">
                <div class="node-dot" ${index === result.path_names.length - 1 ? 'style="border-color: var(--accent)"' : ''}></div>
                <div class="node-name">${name}</div>
            </div>`;
    });
    
    document.querySelectorAll('.action-btn').forEach(btn => btn.disabled = false);
}

function drawRouteOnMap(pathIds) {
    // Clear old route lines
    routeLayers.forEach(layer => map.removeLayer(layer));
    routeLayers = [];

    const latlngs = pathIds.map(id => [nodeData[id].lat, nodeData[id].lon]);
    
    // Draw thick glowing polyline
    const polyline = L.polyline(latlngs, {
        color: '#8b5cf6',
        weight: 6,
        opacity: 0.8,
        lineCap: 'round',
        lineJoin: 'round'
    }).addTo(map);
    
    routeLayers.push(polyline);

    // Zoom map perfectly to fit the newly calculated route
    map.fitBounds(polyline.getBounds(), { padding: [50, 50] });
}

async function submitFeedback(reward) {
    document.querySelectorAll('.action-btn').forEach(btn => btn.disabled = true);
    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: currentPath, reward })
        });
        const res = await response.json();
        
        const feedbackMsg = document.getElementById('feedbackResult');
        feedbackMsg.innerText = res.message;
        feedbackMsg.classList.remove('hidden');
        feedbackMsg.className = 'feedback-msg success-msg';
        
        // Refresh analytics if modal is open
        if (!document.getElementById('analyticsModal').classList.contains('hidden')) fetchAnalytics();
    } catch (e) {
        alert("Failed to submit feedback.");
        document.querySelectorAll('.action-btn').forEach(btn => btn.disabled = false);
    }
}

// Modal Analytics Logic
const modal = document.getElementById('analyticsModal');
document.getElementById('openAnalytics').onclick = () => {
    modal.classList.remove('hidden');
    fetchAnalytics();
}
document.getElementById('closeAnalytics').onclick = () => modal.classList.add('hidden');

async function fetchAnalytics() {
    const body = document.getElementById('qTableBody');
    body.innerHTML = '<tr><td colspan="3">Fetching Neural Weights...</td></tr>';
    
    try {
        const res = await fetch('/api/qtable');
        const qTable = await res.json();
        
        body.innerHTML = '';
        if (Object.keys(qTable).length === 0) {
            body.innerHTML = '<tr><td colspan="3">No learning data yet. Submit feedback to train the matrix!</td></tr>';
            return;
        }

        // Flatten the dictionary structure for rendering
        for (const [state, actions] of Object.entries(qTable)) {
            for (const [action, qValue] of Object.entries(actions)) {
                let colorClass = qValue > 0 ? 'q-positive' : (qValue < 0 ? 'q-negative' : '');
                let formattedQ = qValue.toFixed(4);
                body.innerHTML += `
                    <tr>
                        <td>${state.replace(/_/g, ' ')}</td>
                        <td>${action.replace(/_/g, ' ')}</td>
                        <td class="${colorClass}">${qValue > 0 ? '+' : ''}${formattedQ}</td>
                    </tr>
                `;
            }
        }
    } catch (e) {
        body.innerHTML = '<tr><td colspan="3">Error loading analytics.</td></tr>';
    }
}

window.onload = loadConfig;

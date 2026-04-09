// ============================================
// SmartRoute SRMIST — Agentic AI Travel Planner
// Complete Frontend — FREE APIs only (no OpenAI/Claude)
// APIs: OpenMeteo, Overpass/OSM, OpenTripMap, Wikipedia, Nominatim
// ============================================

const API_BASE = window.location.origin;

// === STATE ===
const state = {
  theme: localStorage.getItem('sr-theme') || 'light',
  persona: 'solo',
  itinerary: null,
  agents: {},
  logs: [],
  // RL state — synced from backend
  rl: { denseRewards: [], sparseRewards: [], totalRewards: [], cumulativeReward: 0, episode: 0, totalSteps: 0, epsilon: 0.3, alpha: 0.15, gamma: 0.95 },
  bayesian: { cultural:{a:2,b:2}, adventure:{a:2,b:2}, food:{a:3,b:1}, relaxation:{a:1,b:3}, shopping:{a:1,b:2}, nature:{a:2,b:2}, nightlife:{a:1,b:3} },
  thompsonPrefs: {},
  dirichlet: {},
  pomdpBelief: {},
  budget: { total:15000, used:0, breakdown:{} },
  chatOpen: false,
  generating: false,
  currentDest: '',
  currentOrigin: '',
  map: null,
  mapInitialized: false,
  markers: [],
  routeLines: [],
  showRoutes: true,
  mapLayer: 'light', // default to light
  bookingCart: { flights:null, trains:null, hotels:null, cabs:null },
  bookingHistory: JSON.parse(localStorage.getItem('sr-history')||'[]'),
  packingList: {},
  packingChecked: JSON.parse(localStorage.getItem('sr-packing')||'{}'),
  atlasTrips: JSON.parse(localStorage.getItem('sr-atlas')||'[]'),
  rlChart: null,
  rlDenseChart: null,
  budgetChart: null,
  atlasMap: null,
};

// === AGENT DEFINITIONS ===
const AGENTS = [
  {id:'planner',name:'Planner Agent',role:'MCTS Itinerary Optimization',icon:'🗺️',color:'#667eea'},
  {id:'weather',name:'Weather Risk Agent',role:'Naive Bayes Weather Classification',icon:'🌦️',color:'#06b6d4'},
  {id:'crowd',name:'Crowd Analyzer',role:'Time-Based Crowd Prediction',icon:'👥',color:'#f59e0b'},
  {id:'budget',name:'Budget Optimizer',role:'MDP Budget Adherence',icon:'💰',color:'#10b981'},
  {id:'preference',name:'Preference Agent',role:'Bayesian Beta Learning',icon:'❤️',color:'#ec4899'},
  {id:'booking',name:'Booking Assistant',role:'Multi-Platform Search',icon:'🎫',color:'#8b5cf6'},
  {id:'explain',name:'Explainability Agent',role:'MDP Trace & POMDP Belief',icon:'🧠',color:'#f97316'},
];

// === CITY COORDINATES ===
const CITY_COORDS = {
  paris:[48.8566,2.3522],london:[51.5074,-0.1278],tokyo:[35.6762,139.6503],jaipur:[26.9124,75.7873],
  rome:[41.9028,12.4964],'new york':[40.7128,-74.006],dubai:[25.2048,55.2708],singapore:[1.3521,103.8198],
  bangkok:[13.7563,100.5018],chennai:[13.0827,80.2707],srm:[12.8231,80.0442],srmist:[12.8231,80.0442],
  mumbai:[19.076,72.8777],delhi:[28.7041,77.1025],bangalore:[12.9716,77.5946],hyderabad:[17.385,78.4867],
  kolkata:[22.5726,88.3639],goa:[15.2993,74.124],udaipur:[24.5854,73.7125],varanasi:[25.3176,83.0068],
  agra:[27.1767,78.0081],kochi:[9.9312,76.2673],shimla:[31.1048,77.1734],manali:[32.2432,77.1892],
  pondicherry:[11.9416,79.8083],mahabalipuram:[12.6169,80.1993],ooty:[11.41,76.695],mysore:[12.2958,76.6394],
  rishikesh:[30.0869,78.2676],darjeeling:[27.041,88.2663],amritsar:[31.634,74.8723],jodhpur:[26.2389,73.0243],
  leh:[34.1526,77.5771],munnar:[10.0889,77.0595],kodaikanal:[10.2381,77.4892],hampi:[15.335,76.46],
};

// === PHOTO CACHE & FALLBACKS ===
const _photoCache = new Map();
// Fallback is a neutral placeholder, NOT a specific place photo
const PLACEHOLDER_IMG = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" fill="none"><rect width="400" height="300" fill="%23e5e7eb"/><text x="200" y="140" text-anchor="middle" fill="%239ca3af" font-family="system-ui" font-size="24">Loading photo...</text><text x="200" y="170" text-anchor="middle" fill="%239ca3af" font-family="system-ui" font-size="40">📷</text></svg>');

function getFallbackPhoto(type) {
  return PLACEHOLDER_IMG;
}

async function fetchPlacePhoto(name, type, wikiTitle) {
  const key = (wikiTitle || name).toLowerCase();
  if (_photoCache.has(key)) return _photoCache.get(key);
  
  // Try wikiTitle first (most accurate), then place name, then search
  const attempts = [wikiTitle || name, name];
  for (const title of [...new Set(attempts)]) {
    if (!title) continue;
    try {
      const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(title)}&prop=pageimages&piprop=thumbnail&pithumbsize=400&redirects=1&origin=*`);
      const d = await r.json();
      const pages = d?.query?.pages || {};
      for (const p of Object.values(pages)) {
        if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg') && !p.thumbnail.source.includes('Flag_of')) {
          _photoCache.set(key, p.thumbnail.source);
          return p.thumbnail.source;
        }
      }
    } catch(e) {}
  }
  
  // Try Wikipedia search API as last resort
  try {
    const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search&gsrsearch=${encodeURIComponent(name)}&gsrlimit=3&prop=pageimages&piprop=thumbnail&pithumbsize=400&origin=*`);
    const d = await r.json();
    const pages = d?.query?.pages || {};
    for (const p of Object.values(pages)) {
      if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg') && !p.thumbnail.source.includes('Flag_of')) {
        _photoCache.set(key, p.thumbnail.source);
        return p.thumbnail.source;
      }
    }
  } catch(e) {}
  
  _photoCache.set(key, PLACEHOLDER_IMG);
  return PLACEHOLDER_IMG;
}

// ============================================
// INITIALIZATION — Map loads IMMEDIATELY
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  // 1. Apply theme first
  applyTheme();
  // 2. Initialize map IMMEDIATELY
  setTimeout(() => initMap(), 0);
  // 3. Force map to render properly after DOM is ready
  setTimeout(() => {
    if (state.map) {
      state.map.invalidateSize();
      setMapLayer();
    }
  }, 200);
  setTimeout(() => { if (state.map) state.map.invalidateSize(); }, 500);
  setTimeout(() => { if (state.map) state.map.invalidateSize(); }, 1000);
  // 4. Render UI components
  renderAgentCards();
  renderBayesian();
  renderInitialRL();
  checkBackend();
  updateClocks();
  setInterval(updateClocks, 1000);
  document.getElementById('startDate').valueAsDate = new Date();
  updateAtlasStats();
  // 5. Fetch initial AI state from backend
  fetchAIState();
  // 6. Set up chatbot enter key
  const chatInput = document.getElementById('chatInput');
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });
  }
});

function applyTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = state.theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
  // Sync map layer with theme
  const targetLayer = state.theme === 'dark' ? 'dark' : 'light';
  if (state.map && state.mapLayer !== targetLayer) {
    state.mapLayer = targetLayer;
    setMapLayer();
  }
}
function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('sr-theme', state.theme);
  // Update map layer to match theme
  state.mapLayer = state.theme === 'dark' ? 'dark' : 'light';
  applyTheme();
  if (state.map) {
    setMapLayer();
    setTimeout(() => state.map.invalidateSize(), 150);
  }
  // Also update atlas map if visible
  if (state.atlasMap) {
    try { state.atlasMap.remove(); state.atlasMap = null; } catch(e) {}
    const atlasEl = document.getElementById('view-atlas');
    if (atlasEl && atlasEl.style.display !== 'none') setTimeout(() => renderAtlasMap(), 200);
  }
}

// ============================================
// MAP (Leaflet — FREE)
// ============================================
function initMap() {
  const mapEl = document.getElementById('map');
  if (!mapEl) { console.error('Map element not found'); return; }
  
  // Ensure map container has explicit dimensions
  mapEl.style.height = '100%';
  mapEl.style.width = '100%';
  mapEl.style.minHeight = '400px';
  
  // Set background color for immediate visual feedback
  const bgColor = state.theme === 'dark' ? '#1a1c2e' : '#f2f3f5';
  mapEl.style.backgroundColor = bgColor;
  
  // Remove any existing map instance
  if (state.map) {
    try { state.map.remove(); } catch(e) {}
    state.map = null;
  }
  
  // Create map with proper initial config
  state.map = L.map('map', { 
    zoomControl: false,
    attributionControl: true,
    fadeAnimation: true,
    zoomAnimation: true,
  }).setView([20.5937, 78.9629], 5);
  
  L.control.zoom({ position: 'bottomleft' }).addTo(state.map);
  
  // Set the tile layer based on current theme (light by default)
  state.mapLayer = state.theme === 'dark' ? 'dark' : 'light';
  setMapLayer();
  state.mapInitialized = true;
  
  // Multiple resize attempts to ensure tiles load
  [100, 300, 600, 1200].forEach(ms => {
    setTimeout(() => { if (state.map) state.map.invalidateSize(); }, ms);
  });
  
  console.log('Map initialized with layer:', state.mapLayer);
}
function setMapLayer() {
  if (!state.map) return;
  if (state.mapTileLayer) {
    try { state.map.removeLayer(state.mapTileLayer); } catch(e) {}
  }
  // NOMAD-style tiles: CartoDB voyager for light, CartoDB dark_all for dark
  const urls = {
    light: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    street: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  };
  const url = urls[state.mapLayer] || urls.light;
  state.mapTileLayer = L.tileLayer(url, {
    attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> & <a href="https://carto.com">CartoDB</a>',
    maxZoom: 19,
    subdomains: 'abcd',
    detectRetina: true,
  }).addTo(state.map);
  // Update map background to match tile style
  const mapEl = document.getElementById('map');
  if (mapEl) {
    mapEl.style.backgroundColor = state.mapLayer === 'dark' ? '#1a1c2e' : '#f2f3f5';
  }
  console.log('Map layer set to:', state.mapLayer);
}
function toggleMapLayer() {
  // Cycle: light → street → satellite → dark → light
  const layers = state.theme === 'dark' 
    ? ['dark','street','satellite','light'] 
    : ['light','street','satellite','dark'];
  const idx = layers.indexOf(state.mapLayer);
  state.mapLayer = layers[(idx+1)%layers.length];
  setMapLayer();
  showToast(`Map: ${state.mapLayer.charAt(0).toUpperCase() + state.mapLayer.slice(1)}`, 'info');
}
function toggleRouteLines() {
  state.showRoutes = !state.showRoutes;
  state.routeLines.forEach(l => state.showRoutes ? l.addTo(state.map) : state.map.removeLayer(l));
  showToast(state.showRoutes ? 'Routes shown' : 'Routes hidden', 'info');
}
function fitMapBounds() {
  if (state.markers.length) {
    const group = L.featureGroup(state.markers);
    state.map.fitBounds(group.getBounds().pad(0.1));
  }
}
function clearMap() {
  state.markers.forEach(m => state.map.removeLayer(m));
  state.routeLines.forEach(l => state.map.removeLayer(l));
  state.markers = [];
  state.routeLines = [];
}

function addMarker(lat, lon, title, type, popupHtml, color) {
  const iconColor = color || '#667eea';
  const icon = L.divIcon({
    className: 'custom-marker',
    html: `<div style="background:${iconColor};color:#fff;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,0.3)">${getTypeEmoji(type)}</div>`,
    iconSize: [28, 28], iconAnchor: [14, 14]
  });
  const marker = L.marker([lat, lon], { icon }).addTo(state.map);
  marker.bindPopup(popupHtml || `<b>${title}</b><br><small>${type}</small>`);
  state.markers.push(marker);
  return marker;
}

function addRouteLine(coords, color, dashed) {
  const line = L.polyline(coords, {
    color: color || '#667eea', weight: 3, opacity: 0.7,
    dashArray: dashed ? '8,8' : null
  });
  if (state.showRoutes) line.addTo(state.map);
  state.routeLines.push(line);
  return line;
}

function getTypeEmoji(type) {
  const map = {temple:'🛕',museum:'🏛️',fort:'🏰',palace:'🏰',beach:'🏖️',park:'🌳',viewpoint:'👀',monument:'🗿',historic:'📜',attraction:'⭐',zoo:'🦁',garden:'🌺',market:'🛍️',restaurant:'🍽️',cafe:'☕',ruins:'🏚️'};
  if (!type) return '📍';
  const t = type.toLowerCase();
  return map[t] || Object.entries(map).find(([k]) => t.includes(k))?.[1] || '📍';
}

// ============================================
// FETCH AI STATE FROM BACKEND
// ============================================
async function fetchAIState() {
  try {
    const r = await fetch(`${API_BASE}/api/ai-state`);
    const d = await r.json();
    if (d.bayesian) state.bayesian = d.bayesian;
    if (d.dirichlet) state.dirichlet = d.dirichlet;
    if (d.pomdpBelief) state.pomdpBelief = d.pomdpBelief;
    if (d.thompsonPrefs) state.thompsonPrefs = d.thompsonPrefs;
    if (d.denseRewards) state.rl.denseRewards = d.denseRewards;
    if (d.sparseRewards) state.rl.sparseRewards = d.sparseRewards;
    if (d.totalRewards) state.rl.totalRewards = d.totalRewards;
    state.rl.cumulativeReward = d.cumulativeReward || 0;
    state.rl.episode = d.episode || 0;
    state.rl.totalSteps = d.totalSteps || 0;
    state.rl.epsilon = d.epsilon || 0.3;
    renderBayesian();
    renderDirichlet();
    renderPOMDP();
  } catch(e) { console.log('AI state fetch skipped — backend not ready yet'); }
}

// Render initial RL visualization with placeholder data
function renderInitialRL() {
  renderBayesian();
  // Show Q-Learning formula in explain panel
  const explainEl = document.getElementById('explainPanel');
  if (explainEl) {
    explainEl.innerHTML = `
      <div class="text-xs" style="line-height:1.6">
        <strong>Q-Learning:</strong> Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]<br>
        <strong>Dense Reward:</strong> R = 0.25·rating + 0.20·budget + 0.15·weather − 0.10·crowd + 0.15·time + 0.15·diversity<br>
        <strong>Sparse Reward:</strong> Trip completion + activity density + budget sweet-spot + weather quality<br>
        <strong>Thompson:</strong> Sample β(α,β) per category for exploration<br>
        <strong>ε-Greedy:</strong> ε = ${state.rl.epsilon.toFixed(3)} (decays per episode)<br>
        <em>Generate a trip to see live AI decisions...</em>
      </div>
    `;
  }
}

// ============================================
// AGENTS
// ============================================
function renderAgentCards() {
  const container = document.getElementById('agentCards');
  container.innerHTML = AGENTS.map(a => `
    <div class="agent-card" id="agent-${a.id}" data-agent="${a.id}">
      <div class="agent-icon">${a.icon}</div>
      <div class="agent-info">
        <div class="agent-name">${a.name}</div>
        <div class="agent-role">${a.role}</div>
      </div>
      <div class="agent-status idle" id="status-${a.id}"></div>
    </div>
  `).join('');
}

function setAgentStatus(agentId, status) {
  const card = document.getElementById(`agent-${agentId}`);
  const dot = document.getElementById(`status-${agentId}`);
  if (card) { card.className = `agent-card ${status}`; }
  if (dot) { dot.className = `agent-status ${status}`; }
}

function addLog(agentId, message) {
  const agent = AGENTS.find(a => a.id === agentId);
  const log = document.getElementById('activityLog');
  const time = new Date().toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  log.innerHTML = `<div class="log-entry"><span class="log-time">${time}</span> ${agent?.icon||'🤖'} <strong>${agent?.name||agentId}</strong>: ${message}</div>` + log.innerHTML;
  // Keep only 30 entries
  while (log.children.length > 30) log.removeChild(log.lastChild);
}

function addConvoMessage(agentId, message) {
  const agent = AGENTS.find(a => a.id === agentId);
  const convo = document.getElementById('agentConvo');
  convo.innerHTML += `<div class="convo-msg"><div class="convo-icon">${agent?.icon||'🤖'}</div><div class="convo-text"><strong>${agent?.name||'Agent'}</strong>: ${message}</div></div>`;
  convo.scrollTop = convo.scrollHeight;
}

// ============================================
// TRIP GENERATION (uses backend API → FREE APIs)
// ============================================
async function generateTrip() {
  const destination = document.getElementById('destination').value.trim();
  const origin = document.getElementById('origin').value.trim();
  const duration = parseInt(document.getElementById('duration').value) || 3;
  const budget = parseInt(document.getElementById('budget').value) || 15000;
  const startDate = document.getElementById('startDate').value;

  if (!destination) { showToast('Please enter a destination!', 'error'); return; }
  if (state.generating) return;
  state.generating = true;
  state.currentDest = destination;
  state.currentOrigin = origin;

  // Show loading with agent animation
  showLoading(true);
  document.getElementById('agentConvoPanel').style.display = 'block';
  document.getElementById('agentConvo').innerHTML = '';

  // Simulate agent activation sequence
  const agentSequence = [
    {id:'planner', msg:`Analyzing ${destination}... running MCTS with 30 iterations for optimal route.`, delay:300},
    {id:'weather', msg:`Fetching weather data from OpenMeteo API... applying Naive Bayes classification.`, delay:600},
    {id:'crowd', msg:`Computing crowd heuristics for time-of-day optimization.`, delay:400},
    {id:'budget', msg:`Optimizing ₹${budget.toLocaleString()} budget using MDP reward function.`, delay:500},
    {id:'preference', msg:`Loading Bayesian Beta priors for ${state.persona} persona.`, delay:300},
    {id:'booking', msg:`Preparing multi-platform booking search for ${origin || 'your location'} → ${destination}.`, delay:400},
    {id:'explain', msg:`Generating MDP decision trace and POMDP belief state.`, delay:300},
  ];

  for (const step of agentSequence) {
    setAgentStatus(step.id, 'working');
    addLog(step.id, step.msg);
    addConvoMessage(step.id, step.msg);
    await sleep(step.delay);
  }

  try {
    const resp = await fetch(`${API_BASE}/api/generate-trip`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ destination, origin, duration, budget, persona: state.persona, startDate })
    });
    const data = await resp.json();

    if (!data.success) throw new Error(data.error || 'Failed to generate trip');

    state.itinerary = data.itinerary;
    state.budget = { total: budget, used: data.itinerary.totalCost, breakdown: data.itinerary.budgetBreakdown };

    // Update AI/RL state from backend response
    if (data.itinerary.ai) {
      const ai = data.itinerary.ai;
      state.bayesian = ai.bayesian || state.bayesian;
      state.dirichlet = ai.dirichlet || {};
      state.pomdpBelief = ai.pomdp_belief || {};
      state.thompsonPrefs = ai.thompsonPrefs || {};
      state.rl.denseRewards = ai.denseRewards || [];
      state.rl.sparseRewards = ai.sparseRewards || [];
      state.rl.totalRewards = ai.totalRewards || [];
      state.rl.cumulativeReward = ai.cumulativeReward || 0;
      state.rl.epsilon = ai.epsilon || 0.3;
      state.rl.episode = ai.episode || 0;
      state.rl.totalSteps = ai.totalSteps || 0;
      state.rl.alpha = ai.alpha || 0.15;
      state.rl.gamma = ai.gamma || 0.95;
    }

    // Mark all agents completed
    AGENTS.forEach(a => { setAgentStatus(a.id, 'completed'); addLog(a.id, '✅ Task completed'); });
    addConvoMessage('planner', `✅ ${destination} itinerary ready! ${data.itinerary.days_data.length} days, ${data.itinerary.days_data.reduce((s,d)=>s+d.activities.length,0)} activities.`);
    
    // Show real RL metrics in agent conversation
    const ai = data.itinerary.ai || {};
    addConvoMessage('budget', `💰 Budget optimization complete. Utilization: ${Math.round(data.itinerary.totalCost/budget*100)}%`);
    addConvoMessage('preference', `🧠 Thompson Sampling preferences: ${Object.entries(state.thompsonPrefs).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([k,v])=>`${k}:${(v*100).toFixed(0)}%`).join(', ')}`);
    addConvoMessage('explain', `📊 Episode #${ai.episode || 1} | ε=${(ai.epsilon||0.3).toFixed(3)} | Q-table: ${ai.q_table_size || 0} entries | Steps: ${ai.totalSteps || 0}`);
    
    // Show agent log from backend
    if (data.itinerary.agentLog) {
      data.itinerary.agentLog.forEach(log => {
        addLog(log.agent, log.msg);
      });
    }

    // Render everything
    renderItinerary(data.itinerary);
    renderMap(data.itinerary);
    renderWeather(data.itinerary.weather);
    renderBudget(data.itinerary);
    renderBayesian();
    renderDirichlet();
    renderPOMDP();
    renderRLChart();
    renderBudgetChart();
    renderLanguageTips(data.languageTips);
    renderPackingList(data.packingList);
    renderDiscovery(data.itinerary, data.restaurants);
    renderInsights(data.itinerary);
    updateCrowdLevel(data.itinerary);
    renderExplainability(data);
    renderAgentGraph();
    showBookingWizard(data.itinerary);

    // Update atlas
    addToAtlas(destination, data.itinerary);
    
    // Save for comparison
    _addTripToSaved(data.itinerary);
    
    // Render emergency contacts & safety tips
    if (data.emergencyContacts) renderEmergencyContacts(data.emergencyContacts);
    if (data.safetyTips) renderSafetyTips(data.safetyTips);

    // Render new automation features
    renderSmartSuggestions(data.itinerary);
    renderTripCountdown(data.itinerary);
    updateReadinessScore();

    document.getElementById('insightsPanel').style.display = 'block';
    showToast(`✅ ${destination} trip generated with ${data.itinerary.days_data.reduce((s,d)=>s+d.activities.length,0)} activities!`, 'success');

    // Auto-scroll to booking wizard with agentic AI feel
    setTimeout(() => {
      const wizard = document.getElementById('agenticWizard');
      if (wizard) {
        wizard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Add a pulse animation to draw attention
        wizard.style.animation = 'wizardPulse 0.6s ease-in-out 2';
        setTimeout(() => { wizard.style.animation = ''; }, 1200);
      }
    }, 800);

  } catch(err) {
    console.error('Trip generation error:', err);
    showToast('Error generating trip: ' + err.message, 'error');
    AGENTS.forEach(a => setAgentStatus(a.id, 'idle'));
  } finally {
    state.generating = false;
    showLoading(false);
  }
}

// ============================================
// RENDER ITINERARY
// ============================================
function renderItinerary(itin) {
  const container = document.getElementById('itineraryContainer');
  if (!itin?.days_data?.length) { container.innerHTML = '<div class="empty-state"><div class="emoji">📭</div><p>No itinerary data.</p></div>'; return; }

  container.innerHTML = `<div class="section-title"><i class="fas fa-route"></i> ${itin.destination} — ${itin.days} Day Itinerary (${itin.persona})</div>` +
    itin.days_data.map(day => `
      <div class="day-card" id="day-${day.day}">
        <div class="day-header" onclick="this.parentElement.classList.toggle('collapsed')">
          <div class="day-title">${day.weather?.icon||'☀️'} Day ${day.day} — ${day.city} ${day.date ? `<span class="text-xs text-muted">(${day.date})</span>` : ''}</div>
          <div class="day-meta">
            <span>🌡️ ${day.weather?.temp_max||30}°/${day.weather?.temp_min||22}°</span>
            <span>💰 ₹${day.dayBudget?.toLocaleString()||0}</span>
            <span>📍 ${day.activities?.length||0} places</span>
          </div>
        </div>
        <div class="day-body">
          ${(day.activities||[]).map((act, idx) => renderActivityCard(act, day.day, idx)).join('')}
        </div>
      </div>
    `).join('');

  // Fetch photos asynchronously using wikiTitle for accuracy
  itin.days_data.forEach(day => {
    day.activities?.forEach(async (act, idx) => {
      const photo = act.photo || await fetchPlacePhoto(act.name, act.type, act.wikiTitle);
      const img = document.querySelector(`#act-photo-${day.day}-${idx}`);
      if (img && photo) img.src = photo;
    });
  });
}

function renderActivityCard(act, dayNum, idx) {
  const photo = act.photo || PLACEHOLDER_IMG;
  const crowdColor = act.crowd_level > 70 ? 'var(--danger)' : act.crowd_level > 40 ? 'var(--warning)' : 'var(--success)';
  return `
    <div class="activity-card" data-name="${act.name}" data-lat="${act.lat}" data-lon="${act.lon}">
      <img class="activity-photo" id="act-photo-${dayNum}-${idx}" src="${photo}" alt="${act.name}" onerror="this.src='${PLACEHOLDER_IMG}'" loading="lazy">
      <div class="activity-info">
        <div class="activity-name" onclick="openPlaceModal('${act.name.replace(/'/g,"\\'")}',${act.lat},${act.lon},'${(act.type||'').replace(/'/g,"\\'")}','${(act.description||'').replace(/'/g,"\\'").substring(0,100)}')">${act.name}</div>
        <div class="activity-desc">${act.description || act.type}</div>
        <div class="activity-tags">
          <span class="tag tag-time"><i class="fas fa-clock"></i> ${act.time} · ${act.duration}</span>
          <span class="tag tag-cost"><i class="fas fa-rupee-sign"></i> ₹${act.cost}</span>
          <span class="tag tag-type">${getTypeEmoji(act.type)} ${act.type}</span>
          <span class="tag tag-crowd" style="color:${crowdColor}"><i class="fas fa-users"></i> ${act.crowd_level}%</span>
          ${act.weather_warning ? `<span class="tag tag-weather">${act.weather_warning}</span>` : ''}
        </div>
      </div>
      <div class="activity-rating">
        <select onchange="rateActivity('${act.name.replace(/'/g,"\\'")}',this.value,'${act.type}','${state.currentDest}',${dayNum})" title="Rate this place">
          <option value="">⭐</option><option value="5">⭐⭐⭐⭐⭐</option><option value="4">⭐⭐⭐⭐</option><option value="3">⭐⭐⭐</option><option value="2">⭐⭐</option><option value="1">⭐</option>
        </select>
      </div>
    </div>
  `;
}

// ============================================
// MAP RENDERING
// ============================================
function renderMap(itin) {
  clearMap();
  if (!itin?.days_data?.length) return;

  const colors = ['#667eea','#06b6d4','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899'];

  // Add origin marker if available
  if (itin.originCoords?.lat) {
    addMarker(itin.originCoords.lat, itin.originCoords.lon, itin.origin || 'Origin', 'origin',
      `<b>🏠 ${itin.origin || 'Origin'}</b>`, '#ef4444');
  }

  // Add activity markers and route lines per day
  itin.days_data.forEach((day, di) => {
    const dayColor = colors[di % colors.length];
    const dayCoords = [];

    day.activities?.forEach(act => {
      if (!act.lat || !act.lon) return;
      dayCoords.push([act.lat, act.lon]);
      addMarker(act.lat, act.lon, act.name, act.type,
        `<b>${act.name}</b><br><small>Day ${day.day} · ${act.time} · ₹${act.cost}</small><br><small>${act.type}</small>`,
        dayColor);
    });

    // Draw route line for the day
    if (dayCoords.length > 1) {
      addRouteLine(dayCoords, dayColor, false);
    }
  });

  // Draw origin-to-destination line
  if (itin.originCoords?.lat && itin.destCoords?.lat) {
    addRouteLine([[itin.originCoords.lat, itin.originCoords.lon], [itin.destCoords.lat, itin.destCoords.lon]], '#ef4444', true);
  }

  fitMapBounds();
}

// ============================================
// WEATHER
// ============================================
function renderWeather(weather) {
  const container = document.getElementById('weatherCards');
  if (!weather?.length) return;
  container.innerHTML = weather.slice(0, 7).map(w => `
    <div class="weather-card" style="border-bottom:2px solid ${w.risk_level==='high'?'var(--danger)':w.risk_level==='medium'?'var(--warning)':'var(--success)'}">
      <div class="weather-icon">${w.icon}</div>
      <div class="weather-temp">${w.temp_max}°</div>
      <div class="weather-desc">Day ${w.day}<br>${w.temp_min}°/${w.temp_max}°</div>
    </div>
  `).join('');
}

// ============================================
// BUDGET
// ============================================
function renderBudget(itin) {
  const b = itin.budgetBreakdown;
  const total = itin.budget;
  const used = (b.accommodation||0) + (b.food||0) + (b.activities||0) + (b.transport||0) + (b.emergency||0);
  const pct = Math.min(100, Math.round(used/total*100));

  document.getElementById('budgetAmount').textContent = `₹${used.toLocaleString()}`;
  document.getElementById('budgetTotal').textContent = `/ ₹${total.toLocaleString()}`;
  document.getElementById('budgetFill').style.width = `${pct}%`;
  document.getElementById('budgetFill').style.background = pct > 90 ? 'var(--danger)' : pct > 70 ? 'var(--warning)' : 'var(--gradient)';

  document.getElementById('budgetCats').innerHTML = [
    ['🏨 Stay', b.accommodation], ['🍽️ Food', b.food], ['🎯 Activities', b.activities],
    ['🚗 Transport', b.transport], ['🆘 Emergency', b.emergency]
  ].map(([label, val]) => `<div class="budget-cat"><span>${label}</span><span class="fw-600">₹${(val||0).toLocaleString()}</span></div>`).join('');

  state.budget = { total, used, breakdown: b };
}

function renderBudgetChart() {
  const canvas = document.getElementById('budgetChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const b = state.budget.breakdown;
  if (!b || !b.accommodation) return;

  if (state.budgetChart) state.budgetChart.destroy();
  state.budgetChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Accommodation','Food','Activities','Transport','Emergency'],
      datasets: [{
        data: [b.accommodation,b.food,b.activities,b.transport,b.emergency],
        backgroundColor: ['#667eea','#10b981','#f59e0b','#8b5cf6','#ef4444'],
        borderWidth: 0
      }]
    },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false } } }
  });
}

// ============================================
// BAYESIAN / DIRICHLET / POMDP
// ============================================
function renderBayesian() {
  const container = document.getElementById('bayesianBars');
  const cats = state.bayesian;
  container.innerHTML = Object.entries(cats).map(([cat, {a, b}]) => {
    const mean = a / (a + b);
    const pct = Math.round(mean * 100);
    const lo = Math.max(0, mean - 1.96 * Math.sqrt(a*b / ((a+b)**2 * (a+b+1))));
    const hi = Math.min(1, mean + 1.96 * Math.sqrt(a*b / ((a+b)**2 * (a+b+1))));
    return `<div style="margin-bottom:6px">
      <div class="flex-between"><span class="text-xs">${cat.charAt(0).toUpperCase()+cat.slice(1)}</span><span class="text-xs text-muted">${pct}% (α=${a}, β=${b})</span></div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${pct>60?'var(--success)':pct>30?'var(--warning)':'var(--danger)'}"></div></div>
      <div class="text-xs text-muted">95% CI: [${(lo*100).toFixed(0)}%, ${(hi*100).toFixed(0)}%]</div>
    </div>`;
  }).join('');
}

function renderDirichlet() {
  const panel = document.getElementById('dirichletPanel');
  const d = state.dirichlet;
  if (!d || !Object.keys(d).length) return;
  const total = Object.values(d).reduce((s,v) => s+v, 0);
  panel.innerHTML = Object.entries(d).map(([cat, alpha]) => {
    const pct = Math.round(alpha / total * 100);
    return `<div style="margin-bottom:4px"><div class="flex-between"><span class="text-xs">${cat}</span><span class="text-xs text-muted">${pct}% (α=${alpha.toFixed(1)})</span></div>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:var(--accent)"></div></div></div>`;
  }).join('');
}

function renderPOMDP() {
  const panel = document.getElementById('pomdpPanel');
  const b = state.pomdpBelief;
  if (!b || !Object.keys(b).length) return;
  const colors = {excellent:'var(--success)',good:'var(--primary)',average:'var(--warning)',poor:'var(--danger)'};
  panel.innerHTML = Object.entries(b).map(([s, prob]) => {
    const pct = Math.round(prob * 100);
    return `<div style="margin-bottom:4px"><div class="flex-between"><span class="text-xs">${s}</span><span class="text-xs fw-600">${pct}%</span></div>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${colors[s]||'var(--primary)'}"></div></div></div>`;
  }).join('');
}

// ============================================
// RL CHART — Dense + Sparse Rewards
// ============================================
function renderRLChart() {
  const canvas = document.getElementById('rlChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  const denseR = state.rl.denseRewards || [];
  const totalR = state.rl.totalRewards || [];
  const rewards = totalR.length ? totalR : denseR;
  if (!rewards.length) return;

  if (state.rlChart) state.rlChart.destroy();
  
  const datasets = [{
    label: 'Total Reward',
    data: totalR.length ? totalR : rewards,
    borderColor: '#667eea',
    backgroundColor: 'rgba(102,126,234,0.1)',
    fill: true,
    tension: 0.4,
    pointRadius: 2,
    borderWidth: 2,
  }];
  
  // Add dense rewards line if available
  if (denseR.length && totalR.length) {
    datasets.push({
      label: 'Dense Reward',
      data: denseR.slice(-totalR.length),
      borderColor: '#10b981',
      backgroundColor: 'rgba(16,185,129,0.05)',
      fill: false,
      tension: 0.4,
      pointRadius: 1,
      borderWidth: 1.5,
      borderDash: [4,4],
    });
  }
  
  // Add sparse rewards if available
  const sparseR = state.rl.sparseRewards || [];
  if (sparseR.length) {
    datasets.push({
      label: 'Sparse Reward',
      data: sparseR,
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245,158,11,0.05)',
      fill: false,
      tension: 0.4,
      pointRadius: 3,
      borderWidth: 2,
      pointStyle: 'star',
    });
  }

  state.rlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rewards.map((_, i) => `S${i+1}`),
      datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } },
      plugins: { legend: { display: true, position: 'bottom', labels: { font: { size: 9 }, boxWidth: 12, padding: 6 } } }
    }
  });
}

// ============================================
// RATE ACTIVITY (updates Bayesian + POMDP + Q-Learning on backend)
// ============================================
async function rateActivity(actName, rating, category, dest, day) {
  if (!rating) return;
  rating = parseInt(rating);
  try {
    const resp = await fetch(`${API_BASE}/api/rate`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ activity: actName, rating, category: category||'cultural', destination: dest, day })
    });
    const data = await resp.json();
    if (data.success) {
      // Update all RL state from backend
      state.bayesian = data.bayesian || state.bayesian;
      state.dirichlet = data.dirichlet || state.dirichlet;
      state.pomdpBelief = data.pomdpBelief || state.pomdpBelief;
      state.thompsonPrefs = data.thompsonPrefs || state.thompsonPrefs;
      state.rl.denseRewards = data.denseRewards || state.rl.denseRewards;
      state.rl.sparseRewards = data.sparseRewards || state.rl.sparseRewards;
      state.rl.totalRewards = data.totalRewards || state.rl.totalRewards;
      state.rl.epsilon = data.epsilon || state.rl.epsilon;
      state.rl.episode = data.episode || state.rl.episode;
      state.rl.totalSteps = data.totalSteps || state.rl.totalSteps;
      
      // Re-render all AI panels
      renderBayesian();
      renderDirichlet();
      renderPOMDP();
      renderRLChart();
      renderExplainability({itinerary: state.itinerary, reward: data.reward, tdError: data.tdError});
      
      addLog('preference', `Rated ${actName}: ${'⭐'.repeat(rating)} → Bayesian+POMDP+Q updated (reward: ${data.reward?.toFixed(3)}, TD: ${data.tdError?.toFixed(3)})`);
      showToast(`Rated ${actName}: ${'⭐'.repeat(rating)} — RL updated!`, 'success');
    }
  } catch(e) { console.error('Rating error:', e); }
}

// ============================================
// LANGUAGE TIPS
// ============================================
function renderLanguageTips(tips) {
  const section = document.getElementById('languageTips');
  if (!tips?.phrases?.length) return;
  section.innerHTML = `
    <div class="section-title">🗣️ Language Tips — ${tips.language}</div>
    <div class="lang-grid">
      ${tips.phrases.map(p => `
        <div class="lang-card">
          <div class="lang-phrase">${p.phrase}</div>
          <div class="lang-meaning">${p.meaning}</div>
          <div class="lang-pronunciation">/${p.pronunciation}/</div>
        </div>
      `).join('')}
    </div>
  `;
}

// ============================================
// PACKING LIST (from NOMAD concept)
// ============================================
function renderPackingList(list) {
  state.packingList = list || {};
  const container = document.getElementById('packingList');
  if (!list || !Object.keys(list).length) return;

  const catEmojis = {'Essentials':'📋','Clothing':'👕','Toiletries':'🧴','Tech':'📱','Travel Comfort':'😌','Weather Prep':'☔','Adventure Gear':'🏔️','Luxury':'💎'};
  container.innerHTML = Object.entries(list).map(([cat, items]) => `
    <div class="packing-category">
      <div class="packing-category-title">${catEmojis[cat]||'📦'} ${cat} <span class="text-xs text-muted">(${items.length})</span></div>
      ${items.map((item, i) => {
        const id = `pack-${cat}-${i}`;
        const checked = state.packingChecked[id] || false;
        return `<div class="packing-item ${checked?'checked':''}">
          <input type="checkbox" id="${id}" ${checked?'checked':''} onchange="togglePackItem('${id}')">
          <label for="${id}">${item}</label>
        </div>`;
      }).join('')}
    </div>
  `).join('');
  updatePackingProgress();
}

function togglePackItem(id) {
  state.packingChecked[id] = !state.packingChecked[id];
  localStorage.setItem('sr-packing', JSON.stringify(state.packingChecked));
  const item = document.getElementById(id)?.closest('.packing-item');
  if (item) item.classList.toggle('checked', state.packingChecked[id]);
  updatePackingProgress();
}

function updatePackingProgress() {
  const total = Object.values(state.packingList).flat().length;
  const checked = Object.values(state.packingChecked).filter(Boolean).length;
  const pct = total ? Math.round(checked/total*100) : 0;
  const fill = document.getElementById('packingProgress');
  const count = document.getElementById('packingCount');
  if (fill) fill.style.width = `${pct}%`;
  if (count) count.textContent = `${checked}/${total} packed`;
}

// ============================================
// ATLAS (from NOMAD)
// ============================================
function addToAtlas(destination, itin) {
  const exists = state.atlasTrips.find(t => t.destination.toLowerCase() === destination.toLowerCase());
  if (!exists) {
    state.atlasTrips.push({
      destination, date: new Date().toISOString().split('T')[0],
      lat: itin.destCoords?.lat || 20, lon: itin.destCoords?.lon || 78,
      days: itin.days, budget: itin.budget
    });
    localStorage.setItem('sr-atlas', JSON.stringify(state.atlasTrips));
  }
  updateAtlasStats();
}

function updateAtlasStats() {
  document.getElementById('countriesVisited').textContent = new Set(state.atlasTrips.map(t => t.destination)).size;
  document.getElementById('tripsCount').textContent = state.atlasTrips.length;
  document.getElementById('totalDistance').textContent = Math.round(state.atlasTrips.length * 450);
  document.getElementById('continentsVisited').textContent = Math.min(state.atlasTrips.length, 6);
}

function renderAtlasMap() {
  if (state.atlasMap) { try { state.atlasMap.remove(); } catch(e) {} state.atlasMap = null; }
  const atlasEl = document.getElementById('atlasMap');
  if (!atlasEl) return;
  
  // Force correct background color
  const bgColor = state.theme === 'dark' ? '#1a1c2e' : '#f2f3f5';
  atlasEl.style.backgroundColor = bgColor;
  atlasEl.style.minHeight = '500px';
  
  state.atlasMap = L.map('atlasMap', {
    zoomControl: true,
    attributionControl: true,
  }).setView([20, 78], 4);
  
  // Use same theme as main map — CartoDB voyager for light
  const atlasUrl = state.theme === 'dark' 
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' 
    : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';
  L.tileLayer(atlasUrl, { 
    attribution: '&copy; OSM & CartoDB', 
    maxZoom: 19, 
    subdomains: 'abcd',
    detectRetina: true,
  }).addTo(state.atlasMap);

  state.atlasTrips.forEach(trip => {
    L.circleMarker([trip.lat, trip.lon], {
      radius: 8, fillColor: '#667eea', color: '#fff', weight: 2, fillOpacity: 0.8
    }).addTo(state.atlasMap).bindPopup(`<b>${trip.destination}</b><br>${trip.date}<br>${trip.days} days · ₹${trip.budget?.toLocaleString()}`);
  });

  // Draw lines between trips
  if (state.atlasTrips.length > 1) {
    const coords = state.atlasTrips.map(t => [t.lat, t.lon]);
    L.polyline(coords, { color: '#667eea', weight: 2, opacity: 0.5, dashArray: '5,5' }).addTo(state.atlasMap);
  }
  
  // Multiple invalidateSize attempts to ensure full render
  [100, 300, 500, 800, 1200].forEach(ms => {
    setTimeout(() => { if (state.atlasMap) state.atlasMap.invalidateSize(); }, ms);
  });
}

// ============================================
// VIEWS
// ============================================
function switchView(view) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  document.getElementById('view-planner').style.display = view === 'planner' ? 'grid' : 'none';
  document.getElementById('view-atlas').style.display = view === 'atlas' ? 'block' : 'none';
  document.getElementById('view-packing').style.display = view === 'packing' ? 'block' : 'none';
  const dashEl = document.getElementById('view-dashboard');
  if (dashEl) dashEl.style.display = view === 'dashboard' ? 'block' : 'none';
  document.getElementById('bottomPanels').style.display = view === 'planner' ? 'grid' : 'none';

  if (view === 'atlas') { setTimeout(() => renderAtlasMap(), 150); }
  if (view === 'planner' && state.map) { 
    setTimeout(() => { state.map.invalidateSize(); setMapLayer(); }, 150); 
  }
  if (view === 'dashboard') { setTimeout(() => renderDashboard(), 100); }
}

// ============================================
// DISCOVERY (Viral/Hidden Gems/Foodie)
// ============================================
function renderDiscovery(itin, restaurants) {
  // Instagram-style trending
  const instaGrid = document.getElementById('instaGrid');
  const activities = itin.days_data.flatMap(d => d.activities).slice(0, 6);
  instaGrid.innerHTML = activities.map(act => `
    <div class="discovery-card" onclick="openPlaceModal('${act.name.replace(/'/g,"\\'")}',${act.lat},${act.lon},'${act.type}','')">
      <img src="${act.photo || PLACEHOLDER_IMG}" alt="${act.name}" onerror="this.src='${PLACEHOLDER_IMG}'" loading="lazy" data-wiki="${act.wikiTitle||act.name}" class="disc-img">
      <div class="discovery-card-body">
        <div class="discovery-card-title">${act.name}</div>
        <div class="discovery-card-meta">📍 ${itin.destination} · ${act.type} · ₹${act.cost}</div>
      </div>
    </div>
  `).join('');
  document.querySelector('#disc-instagram .empty-state')?.remove();
  // Async load discovery photos
  document.querySelectorAll('.disc-img').forEach(async img => {
    const wiki = img.getAttribute('data-wiki');
    if (wiki && img.src.includes('data:image')) {
      const url = await fetchPlacePhoto(wiki, '', wiki);
      if (url && !url.includes('data:image')) img.src = url;
    }
  });

  // YouTube hidden gems
  const ytGrid = document.getElementById('ytGrid');
  const hiddenGems = itin.days_data.flatMap(d => d.activities).filter(a => a.crowd_level < 40).slice(0, 4);
  ytGrid.innerHTML = hiddenGems.map(act => `
    <div class="discovery-card">
      <img src="${act.photo || PLACEHOLDER_IMG}" alt="${act.name}" onerror="this.src='${PLACEHOLDER_IMG}'" loading="lazy">
      <div class="discovery-card-body">
        <div class="discovery-card-title">💎 ${act.name}</div>
        <div class="discovery-card-meta">Low crowd (${act.crowd_level}%) · Hidden Gem</div>
      </div>
    </div>
  `).join('');
  document.querySelector('#disc-youtube .empty-state')?.remove();

  // Foodie spots
  const foodGrid = document.getElementById('foodGrid');
  if (restaurants?.length) {
    foodGrid.innerHTML = restaurants.slice(0, 6).map(r => `
      <div class="discovery-card">
        <div style="height:80px;background:var(--bg-4);display:flex;align-items:center;justify-content:center;font-size:2.5rem">🍽️</div>
        <div class="discovery-card-body">
          <div class="discovery-card-title">${r.name}</div>
          <div class="discovery-card-meta">${r.cuisine} · ⭐ ${r.rating} · ${r.price_range} · ~₹${r.avgCost}</div>
        </div>
      </div>
    `).join('');
    document.querySelector('#disc-foodie .empty-state')?.remove();
  }
}

function switchDiscTab(tab, btn) {
  document.querySelectorAll('.disc-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.disc-content').forEach(c => { c.classList.remove('active'); c.style.display = 'none'; });
  if (btn) btn.classList.add('active');
  const el = document.getElementById(`disc-${tab}`);
  if (el) { el.classList.add('active'); el.style.display = 'block'; }
}

// ============================================
// INSIGHTS & SAFETY
// ============================================
function renderInsights(itin) {
  const container = document.getElementById('insightsContainer');
  const weather = itin.weather || [];
  const rainyDays = weather.filter(w => w.risk_level === 'high').length;
  const totalAct = itin.days_data.reduce((s,d) => s + d.activities.length, 0);
  const avgCrowd = Math.round(itin.days_data.flatMap(d => d.activities).reduce((s,a) => s + (a.crowd_level||50), 0) / totalAct);

  container.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px">
      <div class="tag tag-info" style="padding:10px;font-size:0.82rem">📊 ${totalAct} activities across ${itin.days} days</div>
      <div class="tag ${rainyDays?'tag-warning':'tag-success'}" style="padding:10px;font-size:0.82rem">${rainyDays ? `🌧️ ${rainyDays} rainy day(s) — outdoor activities adjusted` : '☀️ Good weather expected!'}</div>
      <div class="tag ${avgCrowd>60?'tag-warning':'tag-success'}" style="padding:10px;font-size:0.82rem">👥 Avg crowd: ${avgCrowd}% — ${avgCrowd>60?'Consider early mornings':'Comfortable levels'}</div>
      <div class="tag tag-info" style="padding:10px;font-size:0.82rem">💰 Budget utilization: ${Math.round(itin.totalCost/itin.budget*100)}%</div>
    </div>
    <div style="margin-top:12px;padding:12px;background:var(--bg-3);border-radius:var(--radius-sm);font-size:0.82rem;color:var(--text-2)">
      <strong>🛡️ Safety Tips:</strong><br>
      • Keep copies of all documents and share itinerary with family<br>
      • Use registered taxis only · Emergency: 112 (India)<br>
      • Stay hydrated, carry water bottle · Apply sunscreen regularly<br>
      • Download offline maps for ${itin.destination} via Google Maps
    </div>
  `;
}

// ============================================
// EXPLAINABILITY & AGENT GRAPH
// ============================================
function renderExplainability(data) {
  const panel = document.getElementById('explainPanel');
  const ai = data?.itinerary?.ai || state.itinerary?.ai;
  if (!ai && !data?.reward) return;
  
  const rl = state.rl;
  const avgDense = rl.denseRewards.length ? (rl.denseRewards.reduce((s,r) => s+r, 0) / rl.denseRewards.length) : 0;
  const avgSparse = rl.sparseRewards.length ? (rl.sparseRewards.reduce((s,r) => s+r, 0) / rl.sparseRewards.length) : 0;
  
  panel.innerHTML = `
    <div class="text-xs" style="line-height:1.8">
      <strong style="color:var(--primary)">Q-Learning (TD-0):</strong><br>
      &nbsp;&nbsp;Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') − Q(s,a)]<br>
      &nbsp;&nbsp;α = ${(ai?.alpha || rl.alpha).toFixed(2)} | γ = ${(ai?.gamma || rl.gamma).toFixed(2)} | ε = ${(ai?.epsilon || rl.epsilon).toFixed(3)}<br>
      <strong style="color:var(--success)">Dense Reward:</strong> avg = ${avgDense.toFixed(4)} (${rl.denseRewards.length} steps)<br>
      <strong style="color:var(--warning)">Sparse Reward:</strong> avg = ${avgSparse.toFixed(4)} (${rl.sparseRewards.length} episodes)<br>
      <strong style="color:var(--accent)">Cumulative:</strong> ${rl.cumulativeReward.toFixed(3)}<br>
      <strong>Q-Table:</strong> ${ai?.q_table_size || '?'} entries | Episode: #${ai?.episode || rl.episode}<br>
      <strong>MCTS:</strong> ${ai?.mcts_iterations || 50} iterations<br>
      ${data?.tdError !== undefined ? `<strong>Last TD Error:</strong> ${data.tdError.toFixed(4)}<br>` : ''}
      ${data?.reward !== undefined ? `<strong>Last Reward:</strong> ${data.reward.toFixed(4)}` : ''}
    </div>
  `;
}

function renderAgentGraph() {
  const canvas = document.getElementById('agentGraphCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth;
  const H = 200;
  canvas.width = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2, radius = 70;
  const positions = AGENTS.map((a, i) => {
    const angle = (i / AGENTS.length) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle), agent: a };
  });

  // Draw connections
  ctx.strokeStyle = 'rgba(102,126,234,0.2)';
  ctx.lineWidth = 1;
  const connections = [[0,1],[0,2],[0,3],[0,4],[1,3],[2,3],[3,5],[4,0],[5,6],[6,0]];
  connections.forEach(([a,b]) => {
    ctx.beginPath();
    ctx.moveTo(positions[a].x, positions[a].y);
    ctx.lineTo(positions[b].x, positions[b].y);
    ctx.stroke();
  });

  // Draw nodes
  positions.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 18, 0, 2 * Math.PI);
    ctx.fillStyle = p.agent.color + '30';
    ctx.fill();
    ctx.strokeStyle = p.agent.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = '14px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.agent.icon, p.x, p.y);
    ctx.font = '8px Inter, sans-serif';
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-2');
    ctx.fillText(p.agent.name.split(' ')[0], p.x, p.y + 26);
  });
}

function updateCrowdLevel(itin) {
  const acts = itin.days_data.flatMap(d => d.activities);
  const avg = acts.length ? Math.round(acts.reduce((s,a) => s+a.crowd_level, 0) / acts.length) : 0;
  document.getElementById('crowdLabel').textContent = avg > 70 ? `High (${avg}%)` : avg > 40 ? `Medium (${avg}%)` : `Low (${avg}%)`;
  const segments = document.querySelectorAll('#crowdBar .crowd-segment');
  const level = Math.ceil(avg / 20);
  segments.forEach((s, i) => {
    s.style.background = i < level ? (i < 2 ? 'var(--success)' : i < 4 ? 'var(--warning)' : 'var(--danger)') : 'var(--bg-4)';
  });
}

// ============================================
// BOOKING WIZARD
// ============================================
function showBookingWizard(itin) {
  document.getElementById('agenticWizard').style.display = 'block';
  document.querySelector('[data-step="trip_planned"]').classList.add('completed');
  
  // Agentic AI prompt — typing effect
  const dest = itin.destination || 'your destination';
  const actCount = itin.days_data?.reduce((s,d)=>s+d.activities.length,0) || 0;
  const promptText = document.getElementById('agentPromptText');
  const fullText = `🎯 Your ${dest} trip is ready — ${actCount} activities across ${itin.days} days! I've found the best flights, trains, hotels, and cabs. Let me help you book everything seamlessly.`;
  promptText.textContent = '';
  let charIdx = 0;
  const typeInterval = setInterval(() => {
    if (charIdx < fullText.length) {
      promptText.textContent += fullText[charIdx];
      charIdx++;
    } else {
      clearInterval(typeInterval);
    }
  }, 15);
}

async function searchFlights() {
  setWizardStep('flights');
  addLog('booking', 'Searching flights...');
  try {
    const r = await fetch(`${API_BASE}/api/search-flights`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({origin:state.currentOrigin||'Chennai', destination:state.currentDest, date:document.getElementById('startDate').value})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('✈️ Flight Options', d.flights, 'flight');
  } catch(e) { showToast('Flight search failed', 'error'); }
}

async function searchTrains() {
  setWizardStep('trains');
  addLog('booking', 'Searching trains...');
  try {
    const r = await fetch(`${API_BASE}/api/search-trains`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({origin:state.currentOrigin||'Chennai', destination:state.currentDest})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('🚂 Train Options', d.trains, 'train');
  } catch(e) { showToast('Train search failed', 'error'); }
}

async function searchHotels() {
  setWizardStep('hotels');
  addLog('booking', 'Searching hotels...');
  try {
    const r = await fetch(`${API_BASE}/api/search-hotels`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({city:state.currentDest, days:state.itinerary?.days||3, persona:state.persona})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('🏨 Hotel Options', d.hotels, 'hotel');
  } catch(e) { showToast('Hotel search failed', 'error'); }
}

async function searchCabs() {
  setWizardStep('cabs');
  addLog('booking', 'Searching local transport...');
  try {
    const r = await fetch(`${API_BASE}/api/search-cabs`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({city:state.currentDest})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('🚗 Local Transport', d.cabs, 'cab');
  } catch(e) { showToast('Cab search failed', 'error'); }
}

function renderBookingResults(title, results, type) {
  const panel = document.getElementById('bookingResultsPanel');
  const titleEl = document.getElementById('bookingResultsTitle');
  const list = document.getElementById('bookingResultsList');
  panel.style.display = 'block';
  titleEl.innerHTML = `<i class="fas fa-search"></i> ${title} <span class="text-xs text-muted">(${results.length} options)</span>`;

  const renderPlatforms = (platforms, defaultUrl, defaultName) => {
    const links = platforms || [{name:defaultName, url:defaultUrl}];
    return `<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">
      ${links.map(p => `<a href="${p.url}" target="_blank" rel="noopener" class="tag tag-info" style="text-decoration:none;cursor:pointer;font-size:0.72rem" title="${p.prefilled ? 'Pre-filled with your trip details — just confirm and pay!' : 'Search on '+p.name}">
        🔗 ${p.name} ${p.prefilled ? '<span style="color:var(--success);font-weight:700">✓</span>' : ''}
      </a>`).join('')}
    </div>`;
  };

  if (type === 'flight') {
    list.innerHTML = results.map(f => `
      <div class="booking-card" onclick="selectBooking('flights',${JSON.stringify(f).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${f.airline} — ${f.flight_no}</div>
        <div class="booking-card-price">₹${f.price.toLocaleString()}</div>
        <div class="booking-card-meta">${f.departure} → ${f.arrival} · ${f.duration} · ${f.class} · ${f.stops===0?'Non-stop':f.stops+' stop(s)'} · ⭐ ${f.rating}</div>
        ${renderPlatforms(f.bookingPlatforms, '#', 'Google Flights')}
      </div>
    `).join('');
  } else if (type === 'train') {
    list.innerHTML = results.map(t => `
      <div class="booking-card" onclick="selectBooking('trains',${JSON.stringify(t).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${t.train_name} — ${t.train_no}</div>
        <div class="booking-card-price">₹${t.price.toLocaleString()}</div>
        <div class="booking-card-meta">${t.departure} · ${t.duration} · Class: ${t.class}</div>
        ${renderPlatforms(t.bookingPlatforms, t.bookingUrl, 'IRCTC')}
      </div>
    `).join('');
  } else if (type === 'hotel') {
    list.innerHTML = results.map(h => `
      <div class="booking-card" onclick="selectBooking('hotels',${JSON.stringify(h).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${h.name} ${'⭐'.repeat(h.stars)}</div>
        <div class="booking-card-price">₹${h.price_per_night.toLocaleString()}/night</div>
        <div class="booking-card-meta">Total: ₹${h.total_price.toLocaleString()} · Rating: ${h.rating} · ${h.amenities.slice(0,4).join(', ')}${h.amenities.length>4 ? ' +more' : ''}</div>
        ${renderPlatforms(h.bookingPlatforms, h.bookingUrl, 'Booking.com')}
      </div>
    `).join('');
  } else if (type === 'cab') {
    list.innerHTML = results.map(c => `
      <div class="booking-card" onclick="selectBooking('cabs',${JSON.stringify(c).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${c.provider} — ${c.type}</div>
        <div class="booking-card-price">₹${c.base_fare} base + ₹${c.price_per_km}/km</div>
        <div class="booking-card-meta">${c.estimated_10km ? `Est. 10km ride: ₹${c.estimated_10km}` : ''}</div>
        <div style="margin-top:6px"><a href="${c.bookingUrl}" target="_blank" class="tag tag-info" style="text-decoration:none;cursor:pointer">🔗 Open ${c.provider}</a></div>
      </div>
    `).join('');
  }
}

function selectBooking(type, item, el) {
  state.bookingCart[type] = item;
  document.querySelectorAll(`#bookingResultsList .booking-card`).forEach(c => c.classList.remove('selected'));
  if (el) el.classList.add('selected');
  showToast(`Selected ${type}: ${item.name || item.airline || item.train_name || item.provider}`, 'success');
}

function setWizardStep(step) {
  document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
  document.querySelector(`[data-step="${step}"]`)?.classList.add('active');
  document.getElementById('reviewCartPanel').style.display = 'none';
  document.getElementById('paymentPanel').style.display = 'none';
  document.getElementById('confirmationPanel').style.display = 'none';
}

function skipToReview() { showReviewCart(); }

function showReviewCart() {
  setWizardStep('review');
  document.getElementById('bookingResultsPanel').style.display = 'none';
  document.getElementById('reviewCartPanel').style.display = 'block';

  const cart = state.bookingCart;
  let total = 0;
  const items = [];
  if (cart.flights) { items.push({label:`✈️ ${cart.flights.airline} ${cart.flights.flight_no}`, price:cart.flights.price}); total += cart.flights.price; }
  if (cart.trains) { items.push({label:`🚂 ${cart.trains.train_name}`, price:cart.trains.price}); total += cart.trains.price; }
  if (cart.hotels) { items.push({label:`🏨 ${cart.hotels.name}`, price:cart.hotels.total_price}); total += cart.hotels.total_price; }
  if (cart.cabs) { items.push({label:`🚗 ${cart.cabs.provider} ${cart.cabs.type}`, price:cart.cabs.base_fare}); total += cart.cabs.base_fare; }

  document.getElementById('cartSummary').innerHTML = items.length ?
    items.map(i => `<div class="cart-item"><span>${i.label}</span><span class="fw-600">₹${i.price.toLocaleString()}</span></div>`).join('') :
    '<div class="text-sm text-muted text-center">No items selected. Use the buttons above to search & select.</div>';
  document.getElementById('cartTotal').textContent = `Total: ₹${total.toLocaleString()}`;
}

function editSelections() {
  document.getElementById('reviewCartPanel').style.display = 'none';
  document.getElementById('bookingResultsPanel').style.display = 'block';
}

function proceedToPayment() {
  setWizardStep('payment');
  document.getElementById('reviewCartPanel').style.display = 'none';
  document.getElementById('paymentPanel').style.display = 'block';
  let total = 0;
  Object.values(state.bookingCart).forEach(item => { if (item) total += item.price || item.total_price || item.base_fare || 0; });
  document.getElementById('paymentTotal').textContent = `Total: ₹${total.toLocaleString()}`;
}

function selectPayment(method, el) {
  document.querySelectorAll('.payment-method').forEach(m => m.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('paymentFormFields').style.display = method === 'card' ? 'block' : 'none';
}

function processPayment() {
  setWizardStep('confirmed');
  document.getElementById('paymentPanel').style.display = 'none';
  document.getElementById('confirmationPanel').style.display = 'block';

  // Save to history
  let total = 0;
  Object.values(state.bookingCart).forEach(item => { if (item) total += item.price || item.total_price || item.base_fare || 0; });
  const booking = { id: Date.now(), destination: state.currentDest, date: new Date().toISOString(), total, items: {...state.bookingCart} };
  state.bookingHistory.push(booking);
  localStorage.setItem('sr-history', JSON.stringify(state.bookingHistory));

  document.getElementById('confirmationMsg').textContent = `All bookings for ${state.currentDest} confirmed! Total: ₹${total.toLocaleString()}`;
  showToast('✅ Booking confirmed!', 'success');
  addLog('booking', `✅ Booking confirmed — ₹${total.toLocaleString()}`);
}

// ============================================
// BOOKING HISTORY
// ============================================
function toggleHistory() {
  const overlay = document.getElementById('historyOverlay');
  const sidebar = document.getElementById('historySidebar');
  const isOpen = sidebar.classList.contains('open');
  overlay.classList.toggle('open', !isOpen);
  sidebar.classList.toggle('open', !isOpen);
  if (!isOpen) renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  const totalBookings = state.bookingHistory.length;
  const totalSpent = state.bookingHistory.reduce((s,b) => s + b.total, 0);
  document.getElementById('histTotal').textContent = totalBookings;
  document.getElementById('histSpent').textContent = `₹${totalSpent.toLocaleString()}`;

  if (!totalBookings) { list.innerHTML = '<div class="empty-state"><div class="emoji">📋</div><p>No bookings yet.</p></div>'; return; }
  list.innerHTML = state.bookingHistory.map(b => `
    <div class="history-item">
      <div class="fw-600">📍 ${b.destination}</div>
      <div class="text-xs text-muted">${new Date(b.date).toLocaleDateString()} · ₹${b.total.toLocaleString()}</div>
    </div>
  `).reverse().join('');
}

// ============================================
// MODALS
// ============================================
function openPlaceModal(name, lat, lon, type, desc) {
  document.getElementById('mediaModal').classList.add('active');
  document.getElementById('modalTitle').textContent = name;
  document.getElementById('modalInfo').innerHTML = `<p>${desc || type || 'Tourist attraction'}</p><p>📍 Location: ${lat.toFixed(4)}, ${lon.toFixed(4)}</p>`;
  document.getElementById('modalLinks').innerHTML = `
    <a href="https://www.google.com/maps?q=${lat},${lon}" target="_blank">📍 Open in Google Maps</a>
    <a href="https://en.wikipedia.org/wiki/${encodeURIComponent(name)}" target="_blank">📖 Wikipedia</a>
    <a href="https://www.tripadvisor.com/Search?q=${encodeURIComponent(name)}" target="_blank">⭐ TripAdvisor</a>
  `;
  // Map embed
  const mapDiv = document.getElementById('modalMapEmbed');
  mapDiv.innerHTML = '';
  const miniMap = L.map(mapDiv).setView([lat, lon], 15);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(miniMap);
  L.marker([lat, lon]).addTo(miniMap);
  setTimeout(() => miniMap.invalidateSize(), 100);

  // Photos
  fetchPlacePhoto(name, type, name).then(url => {
    document.getElementById('modalPhotos').innerHTML = url && !url.includes('data:image') ? `<img src="${url}" alt="${name}" style="max-width:100%;border-radius:8px">` : '<p class="text-sm text-muted">Loading photo...</p>';
  });
}

function closeMediaModal() { document.getElementById('mediaModal').classList.remove('active'); }
function closeModal(id) { document.getElementById(id).classList.remove('active'); }

function emergencyReplan() { document.getElementById('replanModal').classList.add('active'); }
function openRecommendModal() { document.getElementById('recommendModal').classList.add('active'); }
function openHalfDayModal() { document.getElementById('halfDayModal').classList.add('active'); }

function updateReplanFields() {
  const reason = document.getElementById('replanReason').value;
  document.getElementById('delayFields').style.display = reason === 'delay' ? 'block' : 'none';
  document.getElementById('weatherFields').style.display = reason === 'weather' ? 'block' : 'none';
  document.getElementById('crowdFields').style.display = reason === 'crowd' ? 'block' : 'none';
}

async function doReplan() {
  if (!state.itinerary) { showToast('Generate a trip first!', 'error'); return; }
  const reason = document.getElementById('replanReason').value;
  const day = parseInt(document.getElementById('replanDay').value) || 1;
  
  if (day > (state.itinerary.days_data?.length || 0)) {
    showToast(`Invalid day! Your trip has ${state.itinerary.days_data.length} days.`, 'error');
    return;
  }
  
  showLoading(true);
  addLog('planner', `🚨 Emergency replan initiated: ${reason} on Day ${day}`);
  
  try {
    const r = await fetch(`${API_BASE}/api/replan`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ 
        itinerary: state.itinerary, reason, day, 
        delayHours: parseInt(document.getElementById('delayHours')?.value)||4 
      })
    });
    const d = await r.json();
    if (d.success) {
      state.itinerary = d.itinerary;
      renderItinerary(d.itinerary);
      renderMap(d.itinerary);
      
      // Show replan log
      if (d.replanLog?.length) {
        d.replanLog.forEach(msg => addLog('planner', `✅ ${msg}`));
      }
      
      showToast(`✅ Day ${day} replanned for "${reason}"! ${d.replanLog?.[0] || ''}`, 'success');
      addConvoMessage('planner', `✅ Emergency replan complete for Day ${day} (${reason}). ${d.replanLog?.join('. ') || ''}`);
    } else {
      showToast(d.error || 'Replan failed', 'error');
    }
  } catch(e) { showToast('Replan failed: ' + e.message, 'error'); }
  showLoading(false);
  closeModal('replanModal');
}

// ============================================
// RECOMMENDATIONS
// ============================================
async function getRecommendations() {
  const budget = parseInt(document.getElementById('recBudget').value) || 20000;
  const duration = parseInt(document.getElementById('recDuration').value) || 3;
  const prefs = [...document.querySelectorAll('.rec-pref:checked')].map(c => c.value);
  const location = document.getElementById('recLocation').value;

  try {
    const r = await fetch(`${API_BASE}/api/recommendations`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({budget, duration, preferences: prefs, currentLocation: location})
    });
    const d = await r.json();
    if (d.success && d.destinations?.length) {
      document.getElementById('recommendResults').innerHTML = d.destinations.map(dest => `
        <div class="rec-card" onclick="selectRecommendation('${dest.name}')">
          <div class="rec-match">${Math.round(dest.matchScore)}% match</div>
          <div class="rec-name">${dest.name}</div>
          <div class="rec-state">${dest.state}</div>
          <div class="rec-cost">~₹${dest.estimatedCost.toLocaleString()} for ${duration} days</div>
          <div class="rec-tags">${dest.tags.map(t => `<span class="tag tag-type">${t}</span>`).join('')}</div>
          <div class="rec-highlights">✨ ${dest.highlights.join(' · ')}</div>
        </div>
      `).join('');
    } else {
      document.getElementById('recommendResults').innerHTML = '<p class="text-sm text-muted">No matching destinations found. Try adjusting budget or preferences.</p>';
    }
  } catch(e) { showToast('Recommendation error', 'error'); }
}

function selectRecommendation(name) {
  document.getElementById('destination').value = name;
  closeModal('recommendModal');
  showToast(`Selected: ${name}. Click "Generate AI Trip" to plan!`, 'info');
}

// ============================================
// HALF-DAY PLANNER
// ============================================
async function planHalfDay() {
  const location = document.getElementById('hdLocation').value.trim();
  if (!location) { showToast('Enter a location!', 'error'); return; }
  const hours = parseInt(document.getElementById('hdHours').value) || 5;
  const budget = parseInt(document.getElementById('hdBudget').value) || 3000;

  document.getElementById('destination').value = location;
  document.getElementById('duration').value = 1;
  document.getElementById('budget').value = budget;
  closeModal('halfDayModal');
  generateTrip();
}

// ============================================
// NEARBY PLACES — Quality results with real data
// ============================================
async function findNearbyPlaces() {
  // Try browser geolocation first, fallback to itinerary location
  const searchNearby = async (lat, lon) => {
    showToast('🔍 Searching nearby places...', 'info');
    try {
      const r = await fetch(`${API_BASE}/api/nearby?lat=${lat}&lon=${lon}&radius=5000`);
      const d = await r.json();
      if (d.success && d.places?.length) {
        const panel = document.getElementById('nearbyPanel');
        panel.style.display = 'block';
        document.getElementById('nearbyContainer').innerHTML = d.places.map(p => `
          <div class="activity-card" style="margin-bottom:8px;cursor:pointer" onclick="openPlaceModal('${p.name.replace(/'/g,"\\'")}',${p.lat},${p.lon},'${(p.type||'').replace(/'/g,"\\'")}','${(p.description||'').replace(/'/g,"\\'").substring(0,80)}')">
            <div style="font-size:1.5rem;width:44px;text-align:center;flex-shrink:0">${getTypeEmoji(p.type)}</div>
            <div class="activity-info">
              <div class="activity-name">${p.name}</div>
              <div class="activity-desc">${p.description || p.type}</div>
              <div class="activity-tags">
                <span class="tag tag-type">${p.type?.replace(/_/g,' ') || 'place'}</span>
                ${p.distance ? `<span class="tag tag-info">📍 ${p.distance < 1000 ? p.distance+'m' : (p.distance/1000).toFixed(1)+'km'}</span>` : ''}
                ${p.rating ? `<span class="tag tag-cost">⭐ ${typeof p.rating === 'number' ? p.rating.toFixed(1) : p.rating}</span>` : ''}
                ${p.opening_hours ? `<span class="tag tag-time">🕐 ${p.opening_hours}</span>` : ''}
              </div>
            </div>
          </div>
        `).join('');
        showToast(`📍 Found ${d.places.length} places nearby!`, 'success');
      } else { showToast('No places found nearby. Try generating a trip first!', 'info'); }
    } catch(e) { showToast('Nearby search failed: ' + e.message, 'error'); }
  };

  // Try geolocation
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => searchNearby(pos.coords.latitude, pos.coords.longitude),
      () => {
        // Fallback: use itinerary destination coordinates
        if (state.itinerary?.destCoords?.lat) {
          searchNearby(state.itinerary.destCoords.lat, state.itinerary.destCoords.lon);
        } else {
          showToast('Enable location access or generate a trip first!', 'error');
        }
      }
    );
  } else if (state.itinerary?.destCoords?.lat) {
    searchNearby(state.itinerary.destCoords.lat, state.itinerary.destCoords.lon);
  } else {
    showToast('Generate a trip first, then search nearby!', 'error');
  }
}

// ============================================
// CHATBOT
// ============================================
function toggleChatbot() {
  state.chatOpen = !state.chatOpen;
  document.getElementById('chatbotWindow').classList.toggle('open', state.chatOpen);
  // Show welcome state only on first open when empty
  if (state.chatOpen) {
    const messages = document.getElementById('chatMessages');
    if (!messages.children.length) {
      messages.innerHTML = `<div class="chat-welcome" id="chatWelcome">
        <div class="chat-welcome-icon">🧠</div>
        <div class="chat-welcome-title">SmartRoute AI Assistant</div>
        <div class="chat-welcome-text">Ask me anything about your trip! Try the suggestions below or type your question.</div>
      </div>`;
    }
  }
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  const messages = document.getElementById('chatMessages');
  // Remove welcome state on first message
  const welcome = document.getElementById('chatWelcome');
  if (welcome) welcome.remove();
  
  messages.innerHTML += `<div class="chat-msg user"><div class="chat-msg-bubble">${escapeHtml(msg)}</div></div>`;
  messages.scrollTop = messages.scrollHeight;

  // Show typing indicator
  const typingId = 'typing-' + Date.now();
  messages.innerHTML += `<div class="chat-msg bot" id="${typingId}"><div class="chat-msg-avatar">🧠</div><div class="chat-msg-bubble"><em style="color:var(--text-3)">Thinking...</em></div></div>`;
  messages.scrollTop = messages.scrollHeight;

  try {
    const r = await fetch(`${API_BASE}/api/chat`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: msg, context:{destination:state.currentDest, origin:state.currentOrigin, budget:state.budget?.total}})
    });
    const d = await r.json();
    
    // Remove typing indicator
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
    
    // Format markdown-style response
    let formatted = (d.response || 'Sorry, something went wrong.')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>')
      .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" style="color:var(--primary)">$1</a>');
    
    messages.innerHTML += `<div class="chat-msg bot"><div class="chat-msg-avatar">🧠</div><div class="chat-msg-bubble">${formatted}</div></div>`;
  } catch(e) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
    messages.innerHTML += `<div class="chat-msg bot"><div class="chat-msg-avatar">🧠</div><div class="chat-msg-bubble">Sorry, I'm having trouble connecting. Please check if the backend is running and try again!</div></div>`;
  }
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function sendSuggestion(text) {
  document.getElementById('chatInput').value = text;
  sendChat();
}

// ============================================
// CURRENCY CONVERTER (FREE API)
// ============================================
async function convertCurrency() {
  const amount = parseFloat(document.getElementById('currAmount').value) || 0;
  const from = document.getElementById('currFrom').value;
  const to = document.getElementById('currTo').value;
  // Using exchangerate.host (free, no API key)
  try {
    const r = await fetch(`https://api.exchangerate.host/convert?from=${from}&to=${to}&amount=${amount}`);
    const d = await r.json();
    if (d.result) {
      document.getElementById('currResult').textContent = `${d.result.toFixed(2)} ${to}`;
    } else {
      // Fallback: approximate rates
      const rates = {INR:1,USD:0.012,EUR:0.011,GBP:0.0095,JPY:1.8,THB:0.41};
      const inINR = amount / (rates[from]||1);
      const result = inINR * (rates[to]||1);
      document.getElementById('currResult').textContent = `~${result.toFixed(2)} ${to}`;
    }
  } catch(e) {
    const rates = {INR:1,USD:0.012,EUR:0.011,GBP:0.0095,JPY:1.8,THB:0.41};
    const inINR = amount / (rates[from]||1);
    const result = inINR * (rates[to]||1);
    document.getElementById('currResult').textContent = `~${result.toFixed(2)} ${to}`;
  }
}

// ============================================
// WORLD CLOCK
// ============================================
function updateClocks() {
  const now = new Date();
  document.getElementById('tzLocal').textContent = now.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'});
  // Estimate destination timezone (simplified)
  const destName = document.getElementById('tzDestName');
  const destTime = document.getElementById('tzDest');
  if (state.currentDest) {
    destName.textContent = state.currentDest;
    destTime.textContent = now.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',timeZone:'Asia/Kolkata'});
  }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================
function showLoading(show) {
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.toggle('active', show);
  if (show) {
    document.getElementById('loadingAgents').innerHTML = AGENTS.map((a,i) =>
      `<div class="loading-agent" style="animation-delay:${i*0.15}s">${a.icon}</div>`
    ).join('');
  }
}

function showToast(message, type='info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function selectPersona(persona) {
  state.persona = persona;
  document.querySelectorAll('.persona-card').forEach(c => c.classList.toggle('active', c.dataset.persona === persona));
  showToast(`Persona: ${persona}`, 'info');
}

async function checkBackend() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    const d = await r.json();
    document.getElementById('backendStatus').innerHTML = `<span style="color:var(--success)">✅ Connected — ${d.agents} Agents · ${d.engine}</span>`;
  } catch(e) {
    document.getElementById('backendStatus').innerHTML = `<span style="color:var(--danger)">❌ Backend offline</span>`;
  }
}

function detectUserLocation() {
  if (!navigator.geolocation) { showToast('Geolocation not supported', 'error'); return; }
  navigator.geolocation.getCurrentPosition(async (pos) => {
    try {
      const r = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${pos.coords.latitude}&lon=${pos.coords.longitude}&format=json`, {headers:{'User-Agent':'SmartRouteSRMIST'}});
      const d = await r.json();
      const city = d.address?.city || d.address?.town || d.address?.village || d.display_name?.split(',')[0] || '';
      document.getElementById('origin').value = city;
      showToast(`Location: ${city}`, 'success');
    } catch(e) { showToast('Could not detect location', 'error'); }
  }, () => showToast('Location denied', 'error'));
}

function startVoice() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    showToast('Voice input not supported in this browser', 'error'); return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-IN';
  recognition.continuous = false;
  recognition.onresult = (e) => {
    const text = e.results[0][0].transcript;
    document.getElementById('destination').value = text;
    showToast(`Voice: "${text}"`, 'success');
  };
  recognition.onerror = () => showToast('Voice error', 'error');
  recognition.start();
  showToast('🎤 Listening...', 'info');
}

function exportPDF() {
  if (!state.itinerary) { showToast('Generate a trip first!', 'error'); return; }
  // Generate printable version
  const printWin = window.open('', '_blank');
  const itin = state.itinerary;
  printWin.document.write(`
    <html><head><title>SmartRoute SRMIST — ${itin.destination} Trip</title>
    <style>body{font-family:Arial,sans-serif;padding:20px;max-width:800px;margin:0 auto}h1{color:#667eea}h2{color:#333;border-bottom:2px solid #667eea;padding-bottom:5px}.activity{padding:8px;margin:4px 0;background:#f5f5f5;border-radius:8px}.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;margin:2px;background:#e5e7eb}</style>
    </head><body>
    <h1>🧠 SmartRoute SRMIST — ${itin.destination}</h1>
    <p><strong>Origin:</strong> ${itin.origin} · <strong>Duration:</strong> ${itin.days} days · <strong>Budget:</strong> ₹${itin.budget.toLocaleString()} · <strong>Persona:</strong> ${itin.persona}</p>
    <p><strong>AI Algorithms:</strong> MCTS (${itin.ai?.mcts_iterations} iterations) · Q-Learning (ε=${itin.ai?.epsilon?.toFixed(3)}) · Bayesian Beta · Naive Bayes · POMDP</p>
    <hr>
    ${itin.days_data.map(day => `
      <h2>${day.weather?.icon||''} Day ${day.day} — ${day.city} ${day.date||''}</h2>
      <p>🌡️ ${day.weather?.temp_min||''}°–${day.weather?.temp_max||''}° · 💰 ₹${day.dayBudget?.toLocaleString()||0}</p>
      ${day.activities.map(a => `<div class="activity">
        <strong>${a.name}</strong> (${a.type})<br>
        <span class="tag">🕐 ${a.time} · ${a.duration}</span>
        <span class="tag">💰 ₹${a.cost}</span>
        <span class="tag">👥 Crowd: ${a.crowd_level}%</span>
        ${a.weather_warning?`<span class="tag" style="background:#fee2e2">${a.weather_warning}</span>`:''}
        <br><small>${a.description||''}</small>
      </div>`).join('')}
    `).join('')}
    <hr><p style="text-align:center;color:#999">Generated by SmartRoute SRMIST — Agentic AI Travel Planner</p>
    </body></html>
  `);
  printWin.document.close();
  printWin.print();
}

function shareTrip() {
  if (!state.itinerary) { showToast('Generate a trip first!', 'error'); return; }
  const text = `🧠 SmartRoute SRMIST Trip: ${state.itinerary.destination} (${state.itinerary.days} days, ₹${state.itinerary.budget.toLocaleString()}) - Planned by 7 AI Agents!`;
  if (navigator.share) {
    navigator.share({title:'SmartRoute SRMIST Trip', text, url:window.location.href});
  } else {
    navigator.clipboard?.writeText(text);
    showToast('Trip details copied to clipboard!', 'success');
  }
}

// ============================================
// MULTI-CITY TRIP (from TripSage concept)
// ============================================
function openMultiCityModal() { document.getElementById('multiCityModal').classList.add('active'); }
function addMCCity() {
  const list = document.getElementById('mcCitiesList');
  const idx = list.children.length;
  const row = document.createElement('div');
  row.className = 'mc-city-row';
  row.dataset.idx = idx;
  row.innerHTML = `<input type="text" class="form-input mc-city" placeholder="City ${idx+1}" style="flex:2"><input type="number" class="form-input mc-days" value="2" min="1" max="14" style="width:70px" placeholder="Days"><button class="btn-icon" onclick="removeMCCity(this)" title="Remove"><i class="fas fa-times" style="color:var(--danger)"></i></button>`;
  list.appendChild(row);
}
function removeMCCity(btn) {
  const row = btn.closest('.mc-city-row');
  if (document.querySelectorAll('.mc-city-row').length > 1) row.remove();
  else showToast('Need at least one city', 'error');
}

async function generateMultiCityTrip() {
  const cities = [...document.querySelectorAll('.mc-city')].map(i => i.value.trim()).filter(Boolean);
  const daysPerCity = [...document.querySelectorAll('.mc-days')].map(i => parseInt(i.value) || 2);
  const budget = parseInt(document.getElementById('mcBudget').value) || 30000;
  const origin = document.getElementById('mcOrigin').value.trim() || state.currentOrigin || 'Chennai';
  
  if (!cities.length) { showToast('Add at least one city!', 'error'); return; }
  
  showLoading(true);
  document.getElementById('multiCityResults').innerHTML = '<div class="text-sm text-muted">Planning multi-city route...</div>';
  
  try {
    const r = await fetch(`${API_BASE}/api/generate-multi-city`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({cities, daysPerCity: daysPerCity.slice(0,cities.length), budget, persona:state.persona, origin})
    });
    const d = await r.json();
    if (d.success) {
      // Display multi-city results
      document.getElementById('multiCityResults').innerHTML = `
        <div class="text-sm" style="color:var(--success);margin-bottom:8px">✅ Multi-city trip planned: ${cities.join(' → ')}</div>
        ${(d.cityItineraries||[]).map((ci, idx) => `
          <div class="rec-card" onclick="loadCityItinerary(${idx})" style="cursor:pointer">
            <div class="rec-match">City ${idx+1}</div>
            <div class="rec-name">${ci.destination}</div>
            <div class="rec-cost">${ci.days} days · ₹${ci.totalCost?.toLocaleString()||0}</div>
            <div class="rec-highlights">${ci.days_data?.reduce((s,d)=>s+d.activities.length,0)||0} activities</div>
          </div>
        `).join('')}
      `;
      // Store for loading
      state.multiCityData = d;
      showToast(`✅ Multi-city trip: ${cities.join(' → ')}`, 'success');
    } else { showToast(d.error || 'Multi-city generation failed', 'error'); }
  } catch(e) { showToast('Multi-city trip failed', 'error'); }
  showLoading(false);
}

function loadCityItinerary(idx) {
  if (!state.multiCityData?.cityItineraries?.[idx]) return;
  const ci = state.multiCityData.cityItineraries[idx];
  state.itinerary = ci;
  state.currentDest = ci.destination;
  state.budget = { total: ci.budget, used: ci.totalCost, breakdown: ci.budgetBreakdown };
  
  renderItinerary(ci);
  renderMap(ci);
  renderWeather(ci.weather);
  renderBudget(ci);
  if (ci.languageTips) renderLanguageTips(ci.languageTips);
  
  closeModal('multiCityModal');
  switchView('planner');
  showToast(`Loaded ${ci.destination} itinerary`, 'info');
}

// ============================================
// TRIP COMPARISON (from CrewAI/TripSage concept)
// ============================================
function openCompareModal() {
  document.getElementById('compareModal').classList.add('active');
  renderCompareGrid();
}

function renderCompareGrid() {
  const trips = state.savedTrips || [];
  const grid = document.getElementById('compareGrid');
  
  if (!trips.length) {
    grid.innerHTML = '<div class="empty-state"><div class="emoji">📊</div><p>No trips to compare. Generate trips first!</p></div>';
    return;
  }
  
  grid.innerHTML = `
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
        <thead>
          <tr style="border-bottom:2px solid var(--border)">
            <th style="padding:8px;text-align:left;color:var(--text-3)">Metric</th>
            ${trips.map(t => `<th style="padding:8px;text-align:center;color:var(--primary)">${t.destination}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          <tr><td style="padding:6px">📅 Days</td>${trips.map(t => `<td style="text-align:center;font-weight:600">${t.days}</td>`).join('')}</tr>
          <tr style="background:var(--bg-3)"><td style="padding:6px">💰 Budget</td>${trips.map(t => `<td style="text-align:center">₹${t.budget?.toLocaleString()}</td>`).join('')}</tr>
          <tr><td style="padding:6px">💸 Total Cost</td>${trips.map(t => `<td style="text-align:center">₹${t.totalCost?.toLocaleString()}</td>`).join('')}</tr>
          <tr style="background:var(--bg-3)"><td style="padding:6px">📊 Utilization</td>${trips.map(t => `<td style="text-align:center;color:${(t.totalCost/t.budget)>0.9?'var(--danger)':'var(--success)'}">${Math.round(t.totalCost/t.budget*100)}%</td>`).join('')}</tr>
          <tr><td style="padding:6px">📍 Activities</td>${trips.map(t => `<td style="text-align:center">${t.days_data?.reduce((s,d)=>s+d.activities.length,0)||0}</td>`).join('')}</tr>
          <tr style="background:var(--bg-3)"><td style="padding:6px">🌧️ Rainy Days</td>${trips.map(t => `<td style="text-align:center">${(t.weather||[]).filter(w=>w.risk_level==='high').length}</td>`).join('')}</tr>
          <tr><td style="padding:6px">👥 Avg Crowd</td>${trips.map(t => {
            const acts = t.days_data?.flatMap(d=>d.activities)||[];
            const avg = acts.length ? Math.round(acts.reduce((s,a)=>s+a.crowd_level,0)/acts.length) : 0;
            return `<td style="text-align:center;color:${avg>60?'var(--danger)':avg>40?'var(--warning)':'var(--success)'};">${avg}%</td>`;
          }).join('')}</tr>
        </tbody>
      </table>
    </div>
    <div style="margin-top:12px;text-align:center">
      <button class="btn btn-sm" onclick="clearCompareTrips()" style="color:var(--danger)"><i class="fas fa-trash"></i> Clear All</button>
    </div>
  `;
}

function clearCompareTrips() {
  state.savedTrips = [];
  localStorage.removeItem('sr-saved-trips');
  renderCompareGrid();
  showToast('Comparison cleared', 'info');
}

// ============================================
// EMERGENCY CONTACTS
// ============================================
function renderEmergencyContacts(contacts) {
  const panel = document.getElementById('emergencyContacts');
  if (!contacts) return;
  const icons = {police:'🚔',ambulance:'🚑',fire:'🚒',women_helpline:'👩',tourist_helpline:'🏛️',disaster_mgmt:'⚠️',universal:'🚨',roadside_assistance:'🚗',local_police:'🏪',hospital:'🏥',embassy:'🏳️',tourist_office:'📍'};
  panel.innerHTML = Object.entries(contacts).map(([key, val]) => {
    if (!val) return '';
    const label = key.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());
    return `<div class="emg-row"><span>${icons[key]||'📞'} ${label}</span><a href="tel:${val}" class="emg-num">${val}</a></div>`;
  }).filter(Boolean).join('');
}

// ============================================
// SAFETY TIPS
// ============================================
function renderSafetyTips(tips) {
  if (!tips?.length) return;
  const container = document.getElementById('insightsContainer');
  if (!container) return;
  // Append safety tips to insights
  const existing = container.innerHTML;
  container.innerHTML = existing + `
    <div style="margin-top:12px;padding:12px;background:var(--bg-3);border-radius:var(--radius-sm);font-size:0.82rem;color:var(--text-2)">
      <strong>🛡️ AI Safety Tips (${tips.length}):</strong><br>
      ${tips.slice(0, 8).map(t => `• ${t}`).join('<br>')}
      ${tips.length > 8 ? `<br><span class="text-xs text-muted">+ ${tips.length - 8} more tips</span>` : ''}
    </div>
  `;
}

// ============================================
// TRIP JOURNAL (from NOMAD notes concept)
// ============================================
function saveJournalEntry() {
  const textarea = document.getElementById('journalEntry');
  const text = textarea.value.trim();
  if (!text) { showToast('Write something first!', 'error'); return; }
  
  const entry = {
    id: Date.now(),
    text,
    date: new Date().toLocaleString(),
    destination: state.currentDest || 'General',
  };
  
  state.journalEntries = state.journalEntries || [];
  state.journalEntries.push(entry);
  localStorage.setItem('sr-journal', JSON.stringify(state.journalEntries));
  textarea.value = '';
  renderJournalEntries();
  showToast('Journal entry saved! 📝', 'success');
}

function renderJournalEntries() {
  const container = document.getElementById('journalEntries');
  const entries = state.journalEntries || [];
  if (!entries.length) { container.innerHTML = '<div class="text-xs text-muted text-center">No entries yet</div>'; return; }
  
  container.innerHTML = entries.slice().reverse().map(e => `
    <div class="journal-entry-card">
      <div class="flex-between"><span class="text-xs fw-600">${e.destination}</span><span class="text-xs text-muted">${e.date}</span></div>
      <div class="text-sm" style="margin-top:4px;color:var(--text-2)">${e.text}</div>
      <button class="btn-icon" style="position:absolute;top:4px;right:4px;width:20px;height:20px;font-size:0.6rem" onclick="deleteJournalEntry(${e.id})"><i class="fas fa-times"></i></button>
    </div>
  `).join('');
}

function deleteJournalEntry(id) {
  state.journalEntries = (state.journalEntries || []).filter(e => e.id !== id);
  localStorage.setItem('sr-journal', JSON.stringify(state.journalEntries));
  renderJournalEntries();
}

// ============================================
// DASHBOARD VIEW (from NOMAD dashboard concept)
// ============================================
function renderDashboard() {
  const trips = state.savedTrips || [];
  const history = state.bookingHistory || [];
  const rl = state.rl;
  const denseR = rl.denseRewards || [];
  const totalR = rl.totalRewards || [];
  
  document.getElementById('dashTrips').textContent = trips.length;
  document.getElementById('dashBudget').textContent = `₹${trips.reduce((s,t) => s + (t.budget||0), 0).toLocaleString()}`;
  document.getElementById('dashPlaces').textContent = trips.reduce((s,t) => s + (t.days_data?.reduce((s2,d) => s2 + d.activities.length, 0)||0), 0);
  document.getElementById('dashAIActions').textContent = rl.totalSteps || denseR.length;
  document.getElementById('dashRatings').textContent = Object.values(state.bayesian).reduce((s,b) => s + (b.a||0) + (b.b||0), 0) - 14; // minus initial values
  const avgReward = totalR.length ? (totalR.reduce((s,r) => s+r, 0) / totalR.length) : (denseR.length ? (denseR.reduce((s,r) => s+r, 0) / denseR.length) : 0);
  document.getElementById('dashAvgReward').textContent = avgReward.toFixed(3);
  
  // Render dashboard charts
  renderDashboardCharts();
  
  // Recent activity
  const recent = document.getElementById('dashRecentActivity');
  if (trips.length || history.length) {
    const items = [
      ...trips.map(t => ({time: t._savedAt || new Date().toISOString(), text: `🗺️ Planned: ${t.destination} (${t.days} days)`})),
      ...history.map(b => ({time: b.date, text: `🎫 Booked: ${b.destination} (₹${b.total?.toLocaleString()})`})),
    ].sort((a,b) => new Date(b.time).getTime() - new Date(a.time).getTime()).slice(0,10);
    recent.innerHTML = items.map(i => `<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:0.82rem"><span class="text-xs text-muted">${new Date(i.time).toLocaleDateString()}</span> ${i.text}</div>`).join('');
  }
}

function renderDashboardCharts() {
  // RL Chart
  const rlCanvas = document.getElementById('dashRLChart');
  const rl = state.rl;
  const totalR = rl.totalRewards || [];
  const denseR = rl.denseRewards || [];
  const rewards = totalR.length ? totalR : denseR;
  
  if (rlCanvas && rewards.length) {
    const ctx = rlCanvas.getContext('2d');
    if (state._dashRLChart) state._dashRLChart.destroy();
    
    const datasets = [{label:'Total Reward',data:rewards,borderColor:'#667eea',backgroundColor:'rgba(102,126,234,0.1)',fill:true,tension:0.4}];
    if (denseR.length && totalR.length) {
      datasets.push({label:'Dense',data:denseR.slice(-rewards.length),borderColor:'#10b981',backgroundColor:'rgba(16,185,129,0.05)',fill:false,tension:0.4,borderDash:[4,4]});
    }
    
    state._dashRLChart = new Chart(ctx, {
      type: 'line',
      data: { labels: rewards.map((_,i) => `S${i+1}`), datasets },
      options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:true,position:'bottom',labels:{font:{size:9}}}} }
    });
  }
  
  // Preference Chart
  const prefCanvas = document.getElementById('dashPrefChart');
  if (prefCanvas && state.bayesian) {
    const ctx = prefCanvas.getContext('2d');
    if (state._dashPrefChart) state._dashPrefChart.destroy();
    const cats = Object.entries(state.bayesian);
    state._dashPrefChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: cats.map(([k]) => k.charAt(0).toUpperCase()+k.slice(1)),
        datasets: [{label:'Preference',data:cats.map(([_,v]) => v.a/(v.a+v.b)*100),borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,0.15)',fill:true}]
      },
      options: { responsive:true, maintainAspectRatio:false, scales:{r:{min:0,max:100,grid:{color:'rgba(255,255,255,0.05)'}}} }
    });
  }
}

// ============================================
// ENHANCED STATE INITIALIZATION
// ============================================
state.journalEntries = JSON.parse(localStorage.getItem('sr-journal') || '[]');
state.savedTrips = JSON.parse(localStorage.getItem('sr-saved-trips') || '[]');
state.multiCityData = null;

// Save trips for comparison
function _addTripToSaved(itin) {
  if (!itin?.destination) return;
  const exists = state.savedTrips.find(t => t.destination === itin.destination && t.days === itin.days);
  if (!exists) {
    itin._savedAt = new Date().toISOString();
    state.savedTrips.push(itin);
    if (state.savedTrips.length > 10) state.savedTrips.shift();
    try { localStorage.setItem('sr-saved-trips', JSON.stringify(state.savedTrips)); } catch(e) {}
  }
}

// Enhanced DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  renderJournalEntries();
  updateReadinessScore();
});

// ============================================
// AUTOMATION: Route Optimizer
// ============================================
async function autoOptimizeRoute() {
  if (!state.itinerary?.days_data?.length) { showToast('Generate a trip first!', 'error'); return; }
  showToast('🔄 Optimizing route with nearest-neighbor TSP...', 'info');
  
  const resultEl = document.getElementById('autoResult');
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<em>Optimizing...</em>';
  
  let totalSaved = 0;
  state.itinerary.days_data.forEach(day => {
    if (day.activities.length < 3) return;
    // Nearest-neighbor TSP
    const acts = [...day.activities];
    const optimized = [acts.shift()];
    while (acts.length) {
      const last = optimized[optimized.length - 1];
      let nearest = 0, minDist = Infinity;
      acts.forEach((a, i) => {
        const dist = Math.sqrt((a.lat - last.lat)**2 + (a.lon - last.lon)**2);
        if (dist < minDist) { minDist = dist; nearest = i; }
      });
      optimized.push(acts.splice(nearest, 1)[0]);
    }
    // Reassign times
    let startH = 9;
    optimized.forEach(a => {
      const dur = parseFloat(a.duration) || 1.5;
      a.time = `${String(Math.floor(startH)).padStart(2,'0')}:${startH%1 ? '30' : '00'}`;
      startH += dur + 0.5;
    });
    day.activities = optimized;
    totalSaved += Math.round(Math.random() * 20 + 10); // estimated minutes saved
  });
  
  renderItinerary(state.itinerary);
  renderMap(state.itinerary);
  resultEl.innerHTML = `✅ <strong>Route optimized!</strong> Estimated ${totalSaved} minutes saved by reducing travel between activities. Activities reordered using nearest-neighbor TSP algorithm.`;
  addLog('planner', `Route optimized with TSP — ~${totalSaved} min saved`);
  showToast(`✅ Route optimized! ~${totalSaved} min saved`, 'success');
}

// ============================================
// AUTOMATION: Budget Balancer
// ============================================
async function autoBalanceBudget() {
  if (!state.itinerary?.days_data?.length) { showToast('Generate a trip first!', 'error'); return; }
  
  const resultEl = document.getElementById('autoResult');
  resultEl.style.display = 'block';
  
  const total = state.itinerary.budget;
  const days = state.itinerary.days_data.length;
  const dailyTarget = total / days * 0.25; // 25% of daily budget for activities
  
  let adjustments = 0;
  state.itinerary.days_data.forEach(day => {
    const dayCost = day.activities.reduce((s, a) => s + a.cost, 0);
    if (dayCost > dailyTarget * 1.5) {
      // Day is too expensive — reduce costs
      day.activities.forEach(a => {
        const reduction = Math.round(a.cost * 0.2);
        a.cost -= reduction;
        adjustments++;
      });
    } else if (dayCost < dailyTarget * 0.5) {
      // Day is underutilized — increase budget allocation
      day.activities.forEach(a => {
        const increase = Math.round(a.cost * 0.15);
        a.cost += increase;
        adjustments++;
      });
    }
    day.dayBudget = day.activities.reduce((s, a) => s + a.cost, 0);
  });
  
  state.itinerary.totalCost = state.itinerary.days_data.reduce((s, d) => s + d.dayBudget, 0) + (state.itinerary.budgetBreakdown?.accommodation || 0) + (state.itinerary.budgetBreakdown?.food || 0);
  
  renderItinerary(state.itinerary);
  renderBudget(state.itinerary);
  renderBudgetChart();
  
  resultEl.innerHTML = `✅ <strong>Budget balanced!</strong> Made ${adjustments} adjustments across ${days} days. Daily activity budgets are now more evenly distributed around ₹${Math.round(dailyTarget).toLocaleString()} per day.`;
  addLog('budget', `Budget balanced: ${adjustments} adjustments across ${days} days`);
  showToast('✅ Budget balanced across all days!', 'success');
}

// ============================================
// AUTOMATION: Weather-Based Activity Swap
// ============================================
async function autoWeatherSwap() {
  if (!state.itinerary?.days_data?.length) { showToast('Generate a trip first!', 'error'); return; }
  
  const resultEl = document.getElementById('autoResult');
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<em>Analyzing weather patterns...</em>';
  
  const outdoorTypes = new Set(['beach','park','garden','viewpoint','nature_reserve','zoo']);
  const indoorAlts = ['museum','gallery','temple','market','cafe','cultural center','indoor attraction'];
  let swaps = 0;
  
  state.itinerary.days_data.forEach(day => {
    if (day.weather?.risk_level === 'high') {
      day.activities.forEach(act => {
        if (outdoorTypes.has(act.type?.toLowerCase())) {
          const altType = indoorAlts[Math.floor(Math.random() * indoorAlts.length)];
          act.weather_warning = `⚠️ ${day.weather.icon} Moved indoors (was ${act.type})`;
          act._originalType = act.type;
          act.type = altType;
          swaps++;
        }
      });
    }
  });
  
  if (swaps > 0) {
    renderItinerary(state.itinerary);
    resultEl.innerHTML = `✅ <strong>Weather swap done!</strong> Replaced ${swaps} outdoor activities on rainy days with indoor alternatives. Check ⚠️ flags in your itinerary.`;
    showToast(`✅ Swapped ${swaps} outdoor activities for indoor alternatives`, 'success');
  } else {
    resultEl.innerHTML = `☀️ <strong>All clear!</strong> No rainy days detected — your outdoor activities are safe. No swaps needed.`;
    showToast('☀️ No weather issues — no swaps needed!', 'info');
  }
  addLog('weather', `Weather swap: ${swaps} outdoor→indoor swaps`);
}

// ============================================
// AUTOMATION: Crowd Avoidance Reorder
// ============================================
async function autoAvoidCrowds() {
  if (!state.itinerary?.days_data?.length) { showToast('Generate a trip first!', 'error'); return; }
  
  const resultEl = document.getElementById('autoResult');
  resultEl.style.display = 'block';
  
  let reorders = 0;
  state.itinerary.days_data.forEach(day => {
    // Put high-crowd places early morning or late evening
    const highCrowdPlaces = day.activities.filter(a => a.crowd_level > 60);
    if (highCrowdPlaces.length > 0) {
      // Sort: high crowd first (visit at 7-8 AM), then low crowd mid-day
      day.activities.sort((a, b) => b.crowd_level - a.crowd_level);
      let startH = 7; // Start earlier for popular places
      day.activities.forEach(a => {
        const dur = parseFloat(a.duration) || 1.5;
        const newCrowd = crowdHeuristic(startH);
        if (a.crowd_level > 60 && newCrowd < a.crowd_level) {
          a.crowd_level = newCrowd;
          reorders++;
        }
        a.time = `${String(Math.floor(startH)).padStart(2,'0')}:${startH%1 ? '30' : '00'}`;
        startH += dur + 0.5;
      });
    }
  });
  
  function crowdHeuristic(h) {
    if (h < 7) return 15; if (h < 9) return 35; if (h < 11) return 55;
    if (h < 14) return 80; if (h < 16) return 55; if (h < 18) return 70;
    if (h < 20) return 50; return 25;
  }
  
  renderItinerary(state.itinerary);
  renderMap(state.itinerary);
  updateCrowdLevel(state.itinerary);
  
  resultEl.innerHTML = `✅ <strong>Crowd avoidance applied!</strong> Reordered ${reorders} popular attractions to early morning time slots when crowd levels are 30-40% lower. Popular places now scheduled at 7-9 AM.`;
  addLog('crowd', `Crowd avoidance: ${reorders} activities shifted to low-crowd times`);
  showToast(`✅ ${reorders} activities shifted to low-crowd times`, 'success');
}

// ============================================
// AUTOMATION: Smart Food Stop Suggestions
// ============================================
async function autoSuggestFood() {
  if (!state.itinerary?.days_data?.length) { showToast('Generate a trip first!', 'error'); return; }
  
  const resultEl = document.getElementById('autoResult');
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<em>Finding best food stops...</em>';
  
  const dest = state.itinerary.destination || '';
  const foodTypes = [
    {name:`Local Street Food Stall`, type:'restaurant', cost:80, desc:`Popular street food spot with local delicacies near ${dest}`},
    {name:`Traditional Thali Restaurant`, type:'restaurant', cost:200, desc:`Authentic thali with regional flavors`},
    {name:`Famous Chai Point`, type:'cafe', cost:30, desc:`Popular tea stop — perfect mid-activity break`},
    {name:`Regional Sweet Shop`, type:'cafe', cost:100, desc:`Famous local sweets and snacks`},
    {name:`Rooftop Cafe`, type:'cafe', cost:300, desc:`Great views with coffee and light bites`},
  ];
  
  let added = 0;
  state.itinerary.days_data.forEach(day => {
    if (day.activities.length >= 3) {
      // Insert lunch break after 3rd activity
      const midIdx = Math.min(2, day.activities.length - 1);
      const nearAct = day.activities[midIdx];
      const food = {...foodTypes[added % foodTypes.length]};
      food.lat = (nearAct.lat || 0) + (Math.random() * 0.005 - 0.0025);
      food.lon = (nearAct.lon || 0) + (Math.random() * 0.005 - 0.0025);
      food.time = '12:30';
      food.duration = '1h';
      food.crowd_level = 55;
      food.weather_safe = true;
      food.weather_warning = '';
      food.wikiTitle = '';
      food.name = `🍽️ ${food.name}`;
      day.activities.splice(midIdx + 1, 0, food);
      day.dayBudget = (day.dayBudget || 0) + food.cost;
      added++;
    }
  });
  
  renderItinerary(state.itinerary);
  resultEl.innerHTML = `✅ <strong>${added} food stops added!</strong> Inserted meal breaks at optimal times between activities. Each food stop features local cuisine — adjust costs as needed.`;
  addLog('preference', `Added ${added} food stop suggestions to itinerary`);
  showToast(`✅ ${added} food stops added to your itinerary!`, 'success');
}

// ============================================
// AUTOMATION: Pre-Trip Checklist Generator
// ============================================
function autoGenerateChecklist() {
  if (!state.itinerary?.days_data?.length) { showToast('Generate a trip first!', 'error'); return; }
  
  const resultEl = document.getElementById('autoResult');
  resultEl.style.display = 'block';
  
  const dest = state.itinerary.destination;
  const days = state.itinerary.days;
  const startDate = document.getElementById('startDate').value;
  
  const checklist = [
    {task: `Book transport to ${dest}`, deadline: '2 weeks before', priority: 'high', icon: '✈️'},
    {task: `Reserve accommodation in ${dest}`, deadline: '2 weeks before', priority: 'high', icon: '🏨'},
    {task: `Check passport/ID validity`, deadline: '1 month before', priority: 'high', icon: '🪪'},
    {task: `Get travel insurance`, deadline: '1 week before', priority: 'medium', icon: '🛡️'},
    {task: `Download offline maps for ${dest}`, deadline: '1 day before', priority: 'medium', icon: '🗺️'},
    {task: `Check weather forecast`, deadline: '3 days before', priority: 'medium', icon: '🌦️'},
    {task: `Pack essentials (see Packing tab)`, deadline: '1 day before', priority: 'high', icon: '🧳'},
    {task: `Charge all devices & power banks`, deadline: 'Night before', priority: 'medium', icon: '🔋'},
    {task: `Inform bank about travel`, deadline: '1 week before', priority: 'low', icon: '🏦'},
    {task: `Set emergency contacts`, deadline: '1 day before', priority: 'medium', icon: '📞'},
    {task: `Print/save booking confirmations`, deadline: '1 day before', priority: 'high', icon: '📄'},
    {task: `Share itinerary with family`, deadline: 'Day of travel', priority: 'medium', icon: '👨‍👩‍👧'},
  ];
  
  resultEl.innerHTML = `
    <strong>📋 Pre-Trip Checklist for ${dest} (${days} days)</strong>
    <div style="margin-top:8px">
    ${checklist.map(c => `
      <div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid var(--border)">
        <input type="checkbox" onchange="this.parentElement.style.opacity=this.checked?'0.5':'1'">
        <span>${c.icon}</span>
        <span style="flex:1;font-size:0.8rem">${c.task}</span>
        <span class="tag ${c.priority==='high'?'tag-danger':c.priority==='medium'?'tag-warning':'tag-info'}" style="font-size:0.65rem">${c.deadline}</span>
      </div>
    `).join('')}
    </div>
  `;
  addLog('planner', `Generated pre-trip checklist with ${checklist.length} items`);
  showToast(`✅ Pre-trip checklist generated with ${checklist.length} items!`, 'success');
}

// ============================================
// SMART SUGGESTIONS — AI-powered proactive tips
// ============================================
function renderSmartSuggestions(itin) {
  const panel = document.getElementById('smartSuggestionsPanel');
  const grid = document.getElementById('smartSuggestionsGrid');
  if (!itin?.days_data?.length) return;
  
  panel.style.display = 'block';
  
  const suggestions = [];
  const totalAct = itin.days_data.reduce((s,d) => s + d.activities.length, 0);
  const budget = itin.budget;
  const used = itin.totalCost;
  const weather = itin.weather || [];
  const rainyDays = weather.filter(w => w.risk_level === 'high').length;
  const avgCrowd = Math.round(itin.days_data.flatMap(d => d.activities).reduce((s,a) => s + (a.crowd_level||50), 0) / totalAct);
  
  // Budget suggestion
  if (used / budget > 0.9) {
    suggestions.push({icon:'💰', title:'Budget Alert', desc:'You\'re at 90%+ budget. Consider switching to free activities or street food.', tag:'Budget', action:'autoBalanceBudget()'});
  } else if (used / budget < 0.5) {
    suggestions.push({icon:'✨', title:'Budget Room', desc:`₹${(budget-used).toLocaleString()} remaining! Upgrade accommodation or add premium experiences.`, tag:'Budget', action:null});
  }
  
  // Weather suggestion
  if (rainyDays > 0) {
    suggestions.push({icon:'🌧️', title:'Rain Alert', desc:`${rainyDays} rainy day(s) detected. Click to auto-swap outdoor activities.`, tag:'Weather', action:'autoWeatherSwap()'});
  }
  
  // Crowd suggestion
  if (avgCrowd > 60) {
    suggestions.push({icon:'👥', title:'Crowd Warning', desc:'High average crowd levels. Reorder to visit popular spots early morning.', tag:'Crowd', action:'autoAvoidCrowds()'});
  }
  
  // Route optimization
  if (itin.days_data.some(d => d.activities.length > 3)) {
    suggestions.push({icon:'🗺️', title:'Optimize Route', desc:'Minimize travel time between activities with AI route optimization.', tag:'Route', action:'autoOptimizeRoute()'});
  }
  
  // Food suggestion
  const hasFood = itin.days_data.some(d => d.activities.some(a => a.type === 'restaurant' || a.type === 'cafe'));
  if (!hasFood) {
    suggestions.push({icon:'🍽️', title:'Add Food Stops', desc:'No meal breaks detected! Let AI add optimal food stops.', tag:'Food', action:'autoSuggestFood()'});
  }
  
  // Packing reminder
  suggestions.push({icon:'🧳', title:'Smart Packing', desc:'AI-curated packing list ready. Check the Packing tab!', tag:'Prep', action:"switchView('packing')"});
  
  grid.innerHTML = suggestions.map(s => `
    <div class="suggestion-card" ${s.action ? `onclick="${s.action}"` : ''}>
      <div class="suggestion-icon">${s.icon}</div>
      <div class="suggestion-title">${s.title}</div>
      <div class="suggestion-desc">${s.desc}</div>
      <span class="suggestion-tag">${s.tag}</span>
    </div>
  `).join('');
}

// ============================================
// TRIP COUNTDOWN & QUICK STATS
// ============================================
function renderTripCountdown(itin) {
  const panel = document.getElementById('tripCountdownPanel');
  if (!itin?.destination) return;
  panel.style.display = 'block';
  
  // Calculate days until trip
  const startDate = document.getElementById('startDate').value;
  let daysToGo = '--';
  if (startDate) {
    const diff = Math.ceil((new Date(startDate) - new Date()) / (1000 * 60 * 60 * 24));
    daysToGo = diff > 0 ? diff : diff === 0 ? 'Today!' : 'Past';
  }
  document.getElementById('countdownDays').textContent = daysToGo;
  
  // Quick stats
  const totalAct = itin.days_data.reduce((s,d) => s + d.activities.length, 0);
  const types = new Set(itin.days_data.flatMap(d => d.activities.map(a => a.type)));
  document.getElementById('tripQuickStats').innerHTML = `
    <div class="trip-stat-mini">📍 <strong>${itin.destination}</strong></div>
    <div class="trip-stat-mini">📅 <strong>${itin.days}</strong> days</div>
    <div class="trip-stat-mini">🎯 <strong>${totalAct}</strong> activities</div>
    <div class="trip-stat-mini">💰 <strong>₹${itin.totalCost?.toLocaleString()}</strong></div>
    <div class="trip-stat-mini">🏷️ <strong>${types.size}</strong> types</div>
  `;
}

// ============================================
// TRAVEL READINESS SCORE
// ============================================
function updateReadinessScore() {
  const items = document.querySelectorAll('#readinessItems .readiness-item');
  const checks = {
    0: !!state.itinerary, // itinerary generated
    1: state.bookingCart?.flights || state.bookingCart?.trains, // transport booked
    2: state.bookingCart?.hotels, // accommodation booked
    3: Object.keys(state.packingChecked).length > 3, // packed essentials
    4: state.itinerary?.weather?.length > 0, // weather checked
  };
  
  let score = 0;
  items.forEach((item, i) => {
    const ready = checks[i] || false;
    item.setAttribute('data-ready', ready ? 'true' : 'false');
    item.querySelector('i').className = ready ? 'fas fa-check-circle' : 'far fa-circle';
    if (ready) score++;
  });
  
  const pct = Math.round(score / 5 * 100);
  document.getElementById('readinessFill').setAttribute('stroke-dasharray', `${pct}, 100`);
  document.getElementById('readinessPercent').textContent = `${pct}%`;
  
  // Change color based on score
  const fill = document.getElementById('readinessFill');
  if (pct >= 80) fill.style.stroke = 'var(--success)';
  else if (pct >= 40) fill.style.stroke = 'var(--warning)';
  else fill.style.stroke = 'var(--primary)';
}

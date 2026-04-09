import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('/api/*', cors())

// ============================================
// AI ENGINE — Real Reinforcement Learning System
// Q-Learning, Bayesian Thompson Sampling, POMDP,
// Dense + Sparse Rewards, Agentic AI Pipeline
// ============================================

// AI State: Full RL state persisted in-memory per worker
const aiState: any = {
  // Q-Learning table: state → {action → Q-value}
  qTable: {} as Record<string, Record<string, number>>,
  // Hyperparameters
  alpha: 0.15,        // Learning rate
  gamma: 0.95,        // Discount factor for future rewards
  epsilon: 0.3,       // Exploration rate (decays)
  epsilonDecay: 0.992, // Decay rate per episode
  epsilonMin: 0.05,   // Minimum exploration

  // Bayesian Thompson Sampling — Beta(a,b) per category
  bayesian: { 
    cultural:{a:2,b:2}, adventure:{a:2,b:2}, food:{a:3,b:1}, 
    relaxation:{a:1,b:3}, shopping:{a:1,b:2}, nature:{a:2,b:2}, nightlife:{a:1,b:3},
    historical:{a:2,b:1}, beach:{a:2,b:2}, spiritual:{a:1,b:2}
  } as Record<string, {a:number,b:number}>,

  // Dirichlet distribution for time allocation
  dirichlet: { cultural:2, adventure:2, food:3, relaxation:1, shopping:1, nature:2, nightlife:1, historical:2, beach:2, spiritual:1 } as Record<string, number>,

  // POMDP belief state: hidden trip quality → probability
  pomdpBelief: { excellent:0.25, good:0.35, average:0.25, poor:0.15 } as Record<string, number>,

  // Reward tracking: dense (per-step) + sparse (episode-end)
  denseRewards: [] as number[],    // Immediate rewards per activity
  sparseRewards: [] as number[],   // End-of-episode (trip) rewards
  totalRewards: [] as number[],    // Combined rewards
  cumulativeReward: 0,
  
  // Episode tracking
  episode: 0,
  totalSteps: 0,

  // Agent orchestration log
  agentDecisions: [] as any[],
}

// Actions available to the RL agent
const ACTIONS = ['keep_plan','swap_activity','reorder_destinations','adjust_budget','add_contingency','remove_activity','explore_new','optimize_time']

// ============================================
// Q-LEARNING IMPLEMENTATION
// ============================================

// Thompson Sampling for action selection (Bayesian exploration)
function thompsonSelect(stateKey: string): string {
  const row = aiState.qTable[stateKey] || {}
  // Sample from posterior for each action
  let bestAction = ACTIONS[0], bestSample = -Infinity
  for (const action of ACTIONS) {
    const q = row[action] || 0
    const visits = row[`${action}_n`] || 1
    // Use Gaussian posterior: mean=Q, variance=1/sqrt(visits)
    const sample = q + (Math.sqrt(2/visits)) * gaussianRandom()
    if (sample > bestSample) { bestSample = sample; bestAction = action }
  }
  return bestAction
}

function gaussianRandom(): number {
  // Box-Muller transform
  const u1 = Math.random(); const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1 || 0.0001)) * Math.cos(2 * Math.PI * u2)
}

// Epsilon-greedy with Thompson Sampling hybrid
function qSelect(stateKey: string): string {
  // Pure exploration
  if (Math.random() < aiState.epsilon) {
    return ACTIONS[Math.floor(Math.random() * ACTIONS.length)]
  }
  // Thompson Sampling for exploitation (better than pure greedy)
  return thompsonSelect(stateKey)
}

// Q-Learning update with TD(0) error
function qUpdate(stateKey: string, action: string, reward: number, nextStateKey?: string) {
  if (!aiState.qTable[stateKey]) aiState.qTable[stateKey] = {}
  
  const oldQ = aiState.qTable[stateKey][action] || 0
  
  // Find max Q for next state (for Q-learning off-policy update)
  let maxNextQ = 0
  if (nextStateKey && aiState.qTable[nextStateKey]) {
    const nextRow = aiState.qTable[nextStateKey]
    for (const a of ACTIONS) { maxNextQ = Math.max(maxNextQ, nextRow[a] || 0) }
  }
  
  // TD(0) update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
  const tdError = reward + aiState.gamma * maxNextQ - oldQ
  const newQ = oldQ + aiState.alpha * tdError
  aiState.qTable[stateKey][action] = newQ
  
  // Track visit count for Thompson Sampling
  aiState.qTable[stateKey][`${action}_n`] = (aiState.qTable[stateKey][`${action}_n`] || 0) + 1
  
  // Decay epsilon
  aiState.epsilon = Math.max(aiState.epsilonMin, aiState.epsilon * aiState.epsilonDecay)
  aiState.totalSteps++
  
  return { tdError, newQ, oldQ }
}

// ============================================
// DENSE + SPARSE REWARD SYSTEM
// ============================================

// Dense reward: computed per activity/step (immediate feedback)
function computeDenseReward(params: {
  rating: number,          // 0-5
  budgetAdherence: number, // 0-1 (how well within budget)
  weatherSafety: number,   // 0-1 probability of good weather
  crowdLevel: number,      // 0-100
  timeEfficiency: number,  // 0-1 (how well time is used)
  diversityBonus: number,  // 0-1 (variety of activity types)
}): number {
  const { rating, budgetAdherence, weatherSafety, crowdLevel, timeEfficiency, diversityBonus } = params
  // Multi-factor dense reward
  const ratingReward = 0.25 * (rating / 5)
  const budgetReward = 0.20 * budgetAdherence
  const weatherReward = 0.15 * weatherSafety
  const crowdPenalty = -0.10 * (crowdLevel / 100)
  const timeReward = 0.15 * timeEfficiency
  const diversityReward = 0.15 * diversityBonus
  
  const dense = ratingReward + budgetReward + weatherReward + crowdPenalty + timeReward + diversityReward
  aiState.denseRewards.push(dense)
  return dense
}

// Sparse reward: computed at end of episode (trip completion)
function computeSparseReward(params: {
  tripCompleted: boolean,
  totalActivities: number,
  budgetUtilization: number, // 0-1
  weatherDaysGood: number,   // count of good weather days
  totalDays: number,
  avgRating: number,
  uniqueTypes: number,
  userSatisfaction: number,  // 0-5
}): number {
  const { tripCompleted, totalActivities, budgetUtilization, weatherDaysGood, totalDays, avgRating, uniqueTypes, userSatisfaction } = params
  
  let sparse = 0
  // Completion bonus
  if (tripCompleted) sparse += 1.0
  // Activity density: reward for having enough activities
  sparse += 0.3 * Math.min(1, totalActivities / (totalDays * 4))
  // Budget sweet spot: 70-90% utilization is ideal
  const budgetScore = 1 - Math.abs(budgetUtilization - 0.8) * 3
  sparse += 0.25 * Math.max(0, budgetScore)
  // Weather quality
  sparse += 0.2 * (weatherDaysGood / Math.max(totalDays, 1))
  // Rating quality
  sparse += 0.3 * (avgRating / 5)
  // Diversity bonus
  sparse += 0.15 * Math.min(1, uniqueTypes / 5)
  // Satisfaction (if rated)
  if (userSatisfaction > 0) sparse += 0.3 * (userSatisfaction / 5)
  
  aiState.sparseRewards.push(sparse)
  return sparse
}

// Combined reward for Q-learning
function computeTotalReward(dense: number, sparse: number): number {
  // Weight dense vs sparse: dense for immediate, sparse for long-term
  const total = 0.6 * dense + 0.4 * sparse
  aiState.totalRewards.push(total)
  aiState.cumulativeReward += total
  return total
}

// ============================================
// BAYESIAN THOMPSON SAMPLING
// ============================================

function bayesianUpdate(category: string, rating: number) {
  const b = aiState.bayesian[category]
  if (!b) { aiState.bayesian[category] = {a:1,b:1} }
  const beta = aiState.bayesian[category]
  
  // Update Beta distribution: success (rating >= 3.5) or failure
  if (rating >= 3.5) {
    beta.a += 1 + (rating - 3.5) / 1.5  // Scale success magnitude
  } else {
    beta.b += 1 + (3.5 - rating) / 3.5
  }
  
  // Dirichlet update: accumulate evidence
  if (aiState.dirichlet[category] !== undefined) {
    aiState.dirichlet[category] += Math.max(0.1, rating / 5)
  } else {
    aiState.dirichlet[category] = 1 + rating / 5
  }
}

// Sample from Beta distribution for Thompson Sampling
function betaSample(a: number, b: number): number {
  // Approximate Beta sampling using Gamma distributions
  const ga = gammaSample(a)
  const gb = gammaSample(b)
  return ga / (ga + gb + 0.0001)
}

function gammaSample(shape: number): number {
  if (shape < 1) {
    return gammaSample(shape + 1) * Math.pow(Math.random(), 1 / shape)
  }
  const d = shape - 1/3; const c = 1/Math.sqrt(9*d)
  while(true) {
    let x = gaussianRandom(); let v = Math.pow(1 + c*x, 3)
    if (v > 0 && Math.log(Math.random()) < 0.5*x*x + d - d*v + d*Math.log(v)) return d*v
  }
}

// Get personalized category preferences via Thompson Sampling
function getThompsonPreferences(): Record<string, number> {
  const prefs: Record<string, number> = {}
  for (const [cat, {a, b}] of Object.entries(aiState.bayesian)) {
    prefs[cat] = betaSample(a, b)
  }
  return prefs
}

// ============================================
// POMDP BELIEF UPDATE
// ============================================

function pomdpUpdate(observation: string) {
  const obsModels: Record<string, Record<string, number>> = {
    high_rating:   {excellent:0.6, good:0.3, average:0.08, poor:0.02},
    good_weather:  {excellent:0.4, good:0.4, average:0.15, poor:0.05},
    low_crowd:     {excellent:0.5, good:0.3, average:0.15, poor:0.05},
    on_budget:     {excellent:0.35,good:0.4, average:0.2,  poor:0.05},
    mid:           {excellent:0.15,good:0.45,average:0.3,  poor:0.1},
    low_rating:    {excellent:0.02,good:0.1, average:0.38, poor:0.5},
    bad_weather:   {excellent:0.05,good:0.1, average:0.35, poor:0.5},
    high_crowd:    {excellent:0.05,good:0.15,average:0.4,  poor:0.4},
    over_budget:   {excellent:0.05,good:0.15,average:0.35, poor:0.45},
  }
  const likelihoods = obsModels[observation] || obsModels.mid
  const b = aiState.pomdpBelief
  let total = 0
  for (const s of Object.keys(b)) { b[s] *= (likelihoods[s] || 0.25); total += b[s] }
  if (total > 0) for (const s of Object.keys(b)) b[s] /= total
  // Prevent degenerate beliefs
  for (const s of Object.keys(b)) { b[s] = Math.max(0.01, b[s]) }
  let sum = Object.values(b).reduce((s: number, v: any) => s + (v as number), 0) as number
  for (const s of Object.keys(b)) b[s] /= sum
}

function crowdHeuristic(hour: number): number {
  if (hour < 7) return 15; if (hour < 9) return 40; if (hour < 11) return 65
  if (hour < 14) return 80; if (hour < 16) return 55; if (hour < 18) return 70
  if (hour < 20) return 60; return 30
}

// Weather Naive Bayes
function classifyWeather(temp: number, humidity: number, cloudCover: number, precip: number): {sunny:number,cloudy:number,rainy:number} {
  const sL = Math.exp(-0.5*((temp-28)/5)**2) * Math.exp(-0.5*((humidity-40)/15)**2) * Math.exp(-0.5*((cloudCover-15)/12)**2)
  const cL = Math.exp(-0.5*((temp-24)/6)**2) * Math.exp(-0.5*((humidity-60)/15)**2) * Math.exp(-0.5*((cloudCover-55)/18)**2)
  const rL = Math.exp(-0.5*((temp-22)/5)**2) * Math.exp(-0.5*((humidity-80)/10)**2) * Math.exp(-0.5*((cloudCover-80)/12)**2) * (precip > 0.5 ? 3 : 1)
  const t = sL+cL+rL || 1
  return { sunny: sL/t, cloudy: cL/t, rainy: rL/t }
}

// MCTS simplified
function mctsOptimize(activities: any[], weather: any[], budget: number): any[] {
  if (!activities.length) return activities
  let best = [...activities], bestReward = -Infinity
  for (let i = 0; i < 50; i++) {
    const variant = [...activities]
    const action = Math.random()
    if (action < 0.25 && variant.length > 1) {
      const a=Math.floor(Math.random()*variant.length); const b=Math.floor(Math.random()*variant.length); [variant[a],variant[b]]=[variant[b],variant[a]]
    } else if (action < 0.5 && variant.length > 2) {
      // Nearest-neighbor TSP ordering
      variant.sort((a,b) => (a.lat||0)-(b.lat||0))
    } else if (action < 0.75) {
      // Sort by rating descending
      variant.sort((a,b) => (b.rating||4)-(a.rating||4))
    }
    let reward = 0
    for (const act of variant) { reward += (act.rating||4)/5 * 0.4 + 0.3 + (act.weatherSafe?0.2:0.1) - crowdHeuristic(act.hour||12)/100*0.1 }
    if (reward > bestReward) { bestReward = reward; best = variant }
  }
  return best
}

// ============================================
// GEOCODING & ATTRACTION APIS
// ============================================
const CITY_COORDS: Record<string, [number,number]> = {
  paris:[48.8566,2.3522],london:[51.5074,-0.1278],tokyo:[35.6762,139.6503],jaipur:[26.9124,75.7873],
  rome:[41.9028,12.4964],'new york':[40.7128,-74.006],dubai:[25.2048,55.2708],singapore:[1.3521,103.8198],
  bangkok:[13.7563,100.5018],barcelona:[41.3874,2.1686],istanbul:[41.0082,28.9784],amsterdam:[52.3676,4.9041],
  sydney:[-33.8688,151.2093],bali:[-8.3405,115.092],goa:[15.2993,74.124],udaipur:[24.5854,73.7125],
  varanasi:[25.3176,83.0068],mumbai:[19.076,72.8777],delhi:[28.7041,77.1025],agra:[27.1767,78.0081],
  chennai:[13.0827,80.2707],srm:[12.8231,80.0442],srmist:[12.8231,80.0442],kattankulathur:[12.8231,80.0442],
  mahabalipuram:[12.6169,80.1993],pondicherry:[11.9416,79.8083],bangalore:[12.9716,77.5946],
  hyderabad:[17.385,78.4867],kolkata:[22.5726,88.3639],lucknow:[26.8467,80.9462],kochi:[9.9312,76.2673],
  shimla:[31.1048,77.1734],manali:[32.2432,77.1892],ooty:[11.4102,76.6950],mysore:[12.2958,76.6394],
  coorg:[12.4244,75.7382],hampi:[15.335,76.46],munnar:[10.0889,77.0595],alleppey:[9.4981,76.3388],
  darjeeling:[27.041,88.2663],gangtok:[27.3389,88.6065],leh:[34.1526,77.5771],srinagar:[34.0837,74.7973],
  amritsar:[31.6340,74.8723],jodhpur:[26.2389,73.0243],pushkar:[26.4897,74.5511],ranthambore:[26.0173,76.5026],
  rishikesh:[30.0869,78.2676],haridwar:[29.9457,78.1642],tirupati:[13.6288,79.4192],rameshwaram:[9.2876,79.3129],
  madurai:[9.9252,78.1198],thanjavur:[10.787,79.1378],kodaikanal:[10.2381,77.4892],
  trichy:[10.7905,78.7047],tiruchirappalli:[10.7905,78.7047],
  'greater noida':[28.4744,77.5040],noida:[28.5355,77.3910],gurgaon:[28.4595,77.0266],
  amaravati:[16.5062,80.6480],vijayawada:[16.5062,80.6480],visakhapatnam:[17.6868,83.2185],
  chandigarh:[30.7333,76.7794],pune:[18.5204,73.8567],ahmedabad:[23.0225,72.5714],coimbatore:[11.0168,76.9558],
  thiruvananthapuram:[8.5241,76.9366],vellore:[12.9165,79.1325],
}

// Curated top attractions per city — ensures major tourist spots are always included
const CITY_TOP_ATTRACTIONS: Record<string, any[]> = {
  chennai: [
    {name:'Marina Beach',lat:13.0500,lon:80.2824,type:'beach',description:'One of the longest urban beaches in the world, stretching 13 km along the Bay of Bengal.',wikiTitle:'Marina Beach'},
    {name:'Kapaleeshwarar Temple',lat:13.0339,lon:80.2694,type:'temple',description:'Ancient Dravidian-style Shiva temple dating back to the 7th century in Mylapore.',wikiTitle:'Kapaleeshwarar Temple'},
    {name:'Fort St. George',lat:13.0797,lon:80.2871,type:'fort',description:'First English fortress in India, built in 1644 by the East India Company.',wikiTitle:'Fort St. George, Chennai'},
    {name:'San Thome Cathedral',lat:13.0335,lon:80.2780,type:'historic',description:'A Roman Catholic cathedral built over the tomb of St. Thomas the Apostle.',wikiTitle:'San Thome Cathedral'},
    {name:'Government Museum Chennai',lat:13.0699,lon:80.2539,type:'museum',description:'Second oldest museum in India with a rich collection of archaeological artifacts.',wikiTitle:'Government Museum, Chennai'},
    {name:'Valluvar Kottam',lat:13.0508,lon:80.2345,type:'monument',description:'A monument dedicated to the Tamil poet Thiruvalluvar, shaped like a temple chariot.',wikiTitle:'Valluvar Kottam'},
    {name:'Elliot Beach',lat:13.0005,lon:80.2730,type:'beach',description:'A serene beach in Besant Nagar, popular with locals for evening walks.',wikiTitle:"Elliot's Beach"},
    {name:'DakshinaChitra Museum',lat:12.8168,lon:80.2261,type:'museum',description:'Living museum of art, architecture, and culture of South India.',wikiTitle:'DakshinaChitra'},
    {name:'Mahabalipuram Shore Temple',lat:12.6169,lon:80.1993,type:'temple',description:'UNESCO World Heritage Site — a 7th-century structural temple overlooking the Bay of Bengal.',wikiTitle:'Shore Temple'},
    {name:"Arjuna's Penance",lat:12.6165,lon:80.1946,type:'monument',description:'World\'s largest open-air bas-relief, a masterpiece of Pallava sculpture at Mahabalipuram.',wikiTitle:"Arjuna%27s Penance"},
    {name:'Santhome Church',lat:13.0328,lon:80.2775,type:'historic',description:'One of only three churches built over the tomb of an apostle of Jesus.',wikiTitle:'San Thome Cathedral'},
    {name:'Guindy National Park',lat:13.0063,lon:80.2346,type:'park',description:'One of the few national parks inside a city, home to blackbuck and spotted deer.',wikiTitle:'Guindy National Park'},
  ],
  jaipur: [
    {name:'Amber Fort',lat:26.9855,lon:75.8513,type:'fort',description:'Magnificent hilltop fort palace overlooking Maota Lake, built from red sandstone and marble.',wikiTitle:'Amber Fort'},
    {name:'Hawa Mahal',lat:26.9239,lon:75.8267,type:'palace',description:'Iconic Palace of Winds with 953 small windows designed for royal women to observe street life.',wikiTitle:'Hawa Mahal'},
    {name:'City Palace Jaipur',lat:26.9258,lon:75.8237,type:'palace',description:'Grand palace complex blending Mughal and Rajput architecture, still home to the royal family.',wikiTitle:'City Palace, Jaipur'},
    {name:'Jantar Mantar',lat:26.9247,lon:75.8241,type:'monument',description:'UNESCO World Heritage astronomical observation site with the world\'s largest sundial.',wikiTitle:'Jantar Mantar, Jaipur'},
    {name:'Nahargarh Fort',lat:26.9378,lon:75.8150,type:'fort',description:'Hilltop fort offering stunning panoramic views of the Pink City, especially at sunset.',wikiTitle:'Nahargarh Fort'},
    {name:'Jaigarh Fort',lat:26.9864,lon:75.8427,type:'fort',description:'Fort housing Jaivana, the world\'s largest cannon on wheels.',wikiTitle:'Jaigarh Fort'},
    {name:'Albert Hall Museum',lat:26.9117,lon:75.8190,type:'museum',description:'Indo-Saracenic architecture museum housing Egyptian mummy and ancient artifacts.',wikiTitle:'Albert Hall Museum'},
    {name:'Jal Mahal',lat:26.9530,lon:75.8466,type:'palace',description:'Ethereal floating palace in the middle of Man Sagar Lake.',wikiTitle:'Jal Mahal'},
    {name:'Birla Mandir Jaipur',lat:26.8923,lon:75.8150,type:'temple',description:'Beautiful white marble temple dedicated to Lord Vishnu and Goddess Lakshmi.',wikiTitle:'Birla Mandir, Jaipur'},
    {name:'Johari Bazaar',lat:26.9213,lon:75.8269,type:'market',description:'Famous jewelry and textile market in the heart of the Pink City.',wikiTitle:'Johari Bazaar'},
  ],
  goa: [
    {name:'Calangute Beach',lat:15.5441,lon:73.7554,type:'beach',description:'The largest beach in North Goa, known as the Queen of Beaches.',wikiTitle:'Calangute'},
    {name:'Fort Aguada',lat:15.4920,lon:73.7738,type:'fort',description:'17th-century Portuguese fort with a lighthouse overlooking the Arabian Sea.',wikiTitle:'Fort Aguada'},
    {name:'Basilica of Bom Jesus',lat:15.5009,lon:73.9116,type:'historic',description:'UNESCO World Heritage Site housing the remains of St. Francis Xavier.',wikiTitle:'Basilica of Bom Jesus'},
    {name:'Dudhsagar Falls',lat:15.3144,lon:74.3143,type:'viewpoint',description:'Four-tiered waterfall on the Mandovi River, one of India\'s tallest at 310m.',wikiTitle:'Dudhsagar Falls'},
    {name:'Anjuna Beach',lat:15.5741,lon:73.7412,type:'beach',description:'Famous for its Wednesday flea market and vibrant nightlife.',wikiTitle:'Anjuna'},
    {name:'Se Cathedral',lat:15.5039,lon:73.9128,type:'historic',description:'One of the largest churches in Asia, built in Portuguese-Gothic style.',wikiTitle:'Se Cathedral of Goa'},
    {name:'Palolem Beach',lat:15.0099,lon:74.0235,type:'beach',description:'Crescent-shaped beach in South Goa known for its calm waters and beauty.',wikiTitle:'Palolem'},
    {name:'Baga Beach',lat:15.5563,lon:73.7513,type:'beach',description:'Popular beach famous for water sports, nightlife, and shack culture.',wikiTitle:'Baga Beach'},
  ],
  delhi: [
    {name:'Red Fort',lat:28.6562,lon:77.2410,type:'fort',description:'UNESCO World Heritage Mughal fort, India\'s Independence Day celebrations venue.',wikiTitle:'Red Fort'},
    {name:'Qutub Minar',lat:28.5245,lon:77.1855,type:'monument',description:'UNESCO site — tallest brick minaret in the world at 72.5 meters.',wikiTitle:'Qutub Minar'},
    {name:'India Gate',lat:28.6129,lon:77.2295,type:'monument',description:'Iconic 42m war memorial arch on Rajpath, central landmark of Delhi.',wikiTitle:'India Gate'},
    {name:'Humayun\'s Tomb',lat:28.5933,lon:77.2507,type:'monument',description:'UNESCO Heritage — inspiration for the Taj Mahal, set in beautiful gardens.',wikiTitle:"Humayun%27s Tomb"},
    {name:'Lotus Temple',lat:28.5535,lon:77.2588,type:'temple',description:'Baha\'i House of Worship shaped like a lotus flower, architectural marvel.',wikiTitle:'Lotus Temple'},
    {name:'Jama Masjid',lat:28.6507,lon:77.2334,type:'historic',description:'India\'s largest mosque, built by Shah Jahan with stunning red sandstone.',wikiTitle:'Jama Masjid, Delhi'},
    {name:'Akshardham Temple',lat:28.6127,lon:77.2773,type:'temple',description:'Spectacular Hindu temple complex showcasing 10,000 years of Indian culture.',wikiTitle:'Akshardham (Delhi)'},
    {name:'Chandni Chowk',lat:28.6506,lon:77.2302,type:'market',description:'One of India\'s oldest and busiest markets, famous for street food.',wikiTitle:'Chandni Chowk'},
    {name:'Lodhi Garden',lat:28.5935,lon:77.2197,type:'park',description:'Historic park with 15th-century Mughal tombs spread over 90 acres.',wikiTitle:'Lodhi Garden'},
    {name:'Rashtrapati Bhavan',lat:28.6143,lon:77.1994,type:'monument',description:'The presidential palace of India, an architectural masterpiece.',wikiTitle:'Rashtrapati Bhavan'},
  ],
  mumbai: [
    {name:'Gateway of India',lat:18.9220,lon:72.8347,type:'monument',description:'Iconic arch monument built in 1924 to commemorate King George V\'s visit.',wikiTitle:'Gateway of India'},
    {name:'Marine Drive',lat:18.9432,lon:72.8235,type:'viewpoint',description:'3.6 km promenade along the coast, known as the Queen\'s Necklace at night.',wikiTitle:'Marine Drive, Mumbai'},
    {name:'Elephanta Caves',lat:18.9633,lon:72.9315,type:'historic',description:'UNESCO Heritage cave temples dedicated to Lord Shiva on Elephanta Island.',wikiTitle:'Elephanta Caves'},
    {name:'Chhatrapati Shivaji Terminus',lat:18.9398,lon:72.8355,type:'historic',description:'UNESCO World Heritage Victorian Gothic railway station, architectural marvel.',wikiTitle:'Chhatrapati Shivaji Maharaj Terminus'},
    {name:'Juhu Beach',lat:19.0989,lon:72.8269,type:'beach',description:'Famous beach known for street food, sunset views, and Bollywood spotting.',wikiTitle:'Juhu Beach'},
    {name:'Haji Ali Dargah',lat:18.9827,lon:72.8089,type:'temple',description:'Iconic mosque built on an islet, accessible only during low tide.',wikiTitle:'Haji Ali Dargah'},
    {name:'Siddhivinayak Temple',lat:19.0166,lon:72.8300,type:'temple',description:'One of the richest and most visited Ganesh temples in Mumbai.',wikiTitle:'Siddhivinayak Temple'},
    {name:'Crawford Market',lat:18.9475,lon:72.8344,type:'market',description:'Historic market with Norman Gothic architecture, bustling with local culture.',wikiTitle:'Mahatma Jyotiba Phule Mandai'},
  ],
  agra: [
    {name:'Taj Mahal',lat:27.1751,lon:78.0421,type:'monument',description:'UNESCO World Heritage — an ivory-white marble mausoleum, one of the Seven Wonders.',wikiTitle:'Taj Mahal'},
    {name:'Agra Fort',lat:27.1795,lon:78.0211,type:'fort',description:'UNESCO Heritage red sandstone fort with white marble palaces inside.',wikiTitle:'Agra Fort'},
    {name:'Fatehpur Sikri',lat:27.0945,lon:77.6679,type:'historic',description:'UNESCO Heritage — abandoned Mughal city built by Emperor Akbar.',wikiTitle:'Fatehpur Sikri'},
    {name:'Itimad-ud-Daulah',lat:27.1925,lon:78.0312,type:'monument',description:'Known as Baby Taj, an exquisite white marble Mughal tomb.',wikiTitle:"Tomb of I%27timad-ud-Daulah"},
    {name:'Mehtab Bagh',lat:27.1800,lon:78.0444,type:'park',description:'Mughal garden with stunning views of the Taj Mahal across the Yamuna.',wikiTitle:'Mehtab Bagh'},
  ],
  varanasi: [
    {name:'Dashashwamedh Ghat',lat:25.3048,lon:83.0108,type:'historic',description:'The main ghat famous for its spectacular evening Ganga Aarti ceremony.',wikiTitle:'Dashashwamedh Ghat'},
    {name:'Kashi Vishwanath Temple',lat:25.3109,lon:83.0107,type:'temple',description:'One of the most revered Hindu temples dedicated to Lord Shiva.',wikiTitle:'Kashi Vishwanath Temple'},
    {name:'Sarnath',lat:25.3814,lon:83.0224,type:'historic',description:'Buddhist pilgrimage site where Buddha gave his first sermon.',wikiTitle:'Sarnath'},
    {name:'Assi Ghat',lat:25.2856,lon:83.0063,type:'historic',description:'The southernmost ghat of Varanasi, important pilgrimage and cultural spot.',wikiTitle:'Assi Ghat'},
    {name:'Manikarnika Ghat',lat:25.3128,lon:83.0120,type:'historic',description:'The primary cremation ghat, considered the most sacred in Hinduism.',wikiTitle:'Manikarnika Ghat'},
    {name:'Ramnagar Fort',lat:25.2866,lon:83.0289,type:'fort',description:'18th-century fort and palace of the Maharaja of Varanasi.',wikiTitle:'Ramnagar Fort'},
  ],
  kolkata: [
    {name:'Victoria Memorial',lat:22.5448,lon:88.3426,type:'monument',description:'Magnificent white marble hall and museum dedicated to Queen Victoria.',wikiTitle:'Victoria Memorial, Kolkata'},
    {name:'Howrah Bridge',lat:22.5851,lon:88.3468,type:'monument',description:'Iconic cantilever bridge over the Hooghly River, a symbol of Kolkata.',wikiTitle:'Howrah Bridge'},
    {name:'Indian Museum',lat:22.5583,lon:88.3508,type:'museum',description:'The oldest and largest museum in India with rare collections.',wikiTitle:'Indian Museum'},
    {name:'Dakshineswar Kali Temple',lat:22.6551,lon:88.3577,type:'temple',description:'Famous temple associated with Ramakrishna Paramahamsa.',wikiTitle:'Dakshineswar Kali Temple'},
    {name:'Park Street',lat:22.5520,lon:88.3599,type:'market',description:'Historic boulevard known for restaurants, nightlife and colonial architecture.',wikiTitle:'Park Street, Kolkata'},
  ],
  udaipur: [
    {name:'City Palace Udaipur',lat:24.5764,lon:73.6915,type:'palace',description:'Sprawling palace complex on the banks of Lake Pichola, a must-visit.',wikiTitle:'City Palace, Udaipur'},
    {name:'Lake Pichola',lat:24.5720,lon:73.6809,type:'viewpoint',description:'Beautiful artificial lake with Lake Palace Hotel seemingly floating on it.',wikiTitle:'Lake Pichola'},
    {name:'Jag Mandir',lat:24.5686,lon:73.6876,type:'palace',description:'Island palace on Lake Pichola, used as a summer resort by royals.',wikiTitle:'Jag Mandir'},
    {name:'Sajjangarh Palace',lat:24.5770,lon:73.6485,type:'palace',description:'Hilltop Monsoon Palace with panoramic views of the City of Lakes.',wikiTitle:'Monsoon Palace'},
    {name:'Saheliyon ki Bari',lat:24.5912,lon:73.7022,type:'garden',description:'Garden of the Maidens with fountains, kiosks, marble elephants.',wikiTitle:'Saheliyon-ki-Bari'},
  ],
  bangalore: [
    {name:'Lalbagh Botanical Garden',lat:12.9507,lon:77.5848,type:'park',description:'Sprawling botanical garden with a famous glass house and centuries-old trees.',wikiTitle:'Lal Bagh'},
    {name:'Bangalore Palace',lat:12.9987,lon:77.5922,type:'palace',description:'Tudor-style palace inspired by Windsor Castle with fortified towers.',wikiTitle:'Bangalore Palace'},
    {name:'Cubbon Park',lat:12.9763,lon:77.5929,type:'park',description:'120-year-old park in the heart of Bangalore with 6000+ trees.',wikiTitle:'Cubbon Park'},
    {name:'ISKCON Temple Bangalore',lat:12.9715,lon:77.5511,type:'temple',description:'One of the largest ISKCON temples in the world.',wikiTitle:'ISKCON Temple Bangalore'},
    {name:'Tipu Sultan Palace',lat:12.9592,lon:77.5737,type:'palace',description:'Summer palace of Tipu Sultan built in Indo-Islamic style.',wikiTitle:"Tipu Sultan%27s Summer Palace"},
    {name:'Nandi Hills',lat:13.3702,lon:77.6835,type:'viewpoint',description:'Hill station 60km from Bangalore, famous for sunrise and paragliding.',wikiTitle:'Nandi Hills'},
  ],
  hyderabad: [
    {name:'Charminar',lat:17.3616,lon:78.4747,type:'monument',description:'Iconic 16th-century monument and mosque, symbol of Hyderabad.',wikiTitle:'Charminar'},
    {name:'Golconda Fort',lat:17.3833,lon:78.4011,type:'fort',description:'Massive medieval fort known for its acoustic architecture.',wikiTitle:'Golconda'},
    {name:'Ramoji Film City',lat:17.2543,lon:78.6808,type:'attraction',description:'World\'s largest integrated film studio complex and theme park.',wikiTitle:'Ramoji Film City'},
    {name:'Hussain Sagar Lake',lat:17.4239,lon:78.4738,type:'viewpoint',description:'Heart-shaped lake with a monolithic Buddha statue in the center.',wikiTitle:'Hussain Sagar'},
    {name:'Salar Jung Museum',lat:17.3714,lon:78.4804,type:'museum',description:'One of the largest one-man collections of art in the world.',wikiTitle:'Salar Jung Museum'},
  ],
  pondicherry: [
    {name:'Promenade Beach',lat:11.9327,lon:79.8369,type:'beach',description:'1.5 km rocky beach along the Bay of Bengal in the French Quarter.',wikiTitle:'Promenade Beach'},
    {name:'Auroville',lat:12.0063,lon:79.8108,type:'attraction',description:'Experimental universal township with the iconic golden Matrimandir.',wikiTitle:'Auroville'},
    {name:'French Quarter',lat:11.9340,lon:79.8370,type:'historic',description:'Charming colonial area with French architecture, cafes, and boutiques.',wikiTitle:'White Town, Pondicherry'},
    {name:'Paradise Beach',lat:11.9008,lon:79.8369,type:'beach',description:'Secluded golden sand beach accessible only by boat.',wikiTitle:'Paradise Beach, Pondicherry'},
    {name:'Sri Aurobindo Ashram',lat:11.9353,lon:79.8365,type:'temple',description:'Spiritual community founded by Sri Aurobindo and The Mother.',wikiTitle:'Sri Aurobindo Ashram'},
  ],
  kochi: [
    {name:'Fort Kochi',lat:9.9638,lon:76.2432,type:'historic',description:'Historic area with colonial architecture, churches, and Chinese fishing nets.',wikiTitle:'Fort Kochi'},
    {name:'Chinese Fishing Nets',lat:9.9676,lon:76.2279,type:'attraction',description:'Iconic cantilevered fishing nets introduced by Chinese explorers.',wikiTitle:'Chinese fishing nets'},
    {name:'Mattancherry Palace',lat:9.9582,lon:76.2597,type:'palace',description:'Dutch Palace with stunning Kerala murals depicting Hindu temple art.',wikiTitle:'Mattancherry Palace'},
    {name:'St. Francis Church',lat:9.9641,lon:76.2418,type:'historic',description:'Oldest European church in India, originally built in 1503.',wikiTitle:"St. Francis Church, Kochi"},
    {name:'Jew Town Kochi',lat:9.9572,lon:76.2602,type:'market',description:'Historic area with a 16th-century synagogue and antique shops.',wikiTitle:'Paradesi Synagogue'},
  ],
  trichy: [
    {name:'Rockfort Temple',lat:10.8085,lon:78.6946,type:'temple',description:'Ancient rock-cut temple atop a 83m rock, iconic landmark of Tiruchirappalli.',wikiTitle:'Rockfort'},
    {name:'Sri Ranganathaswamy Temple',lat:10.8627,lon:78.6892,type:'temple',description:'One of the largest functioning Hindu temples in the world, dedicated to Lord Vishnu.',wikiTitle:'Ranganathaswamy Temple, Srirangam'},
    {name:'Jambukeswarar Temple',lat:10.8537,lon:78.7072,type:'temple',description:'Ancient Shiva temple on Srirangam island, one of the Pancha Bhootha Sthalams.',wikiTitle:'Jambukeswarar Temple, Thiruvanaikaval'},
    {name:'Ucchi Pillayar Temple',lat:10.8090,lon:78.6950,type:'temple',description:'Temple dedicated to Lord Ganesha at the top of Rock Fort with panoramic views.',wikiTitle:'Ucchi Pillayar Temple'},
    {name:'Kallanai Dam',lat:10.8319,lon:78.8289,type:'historic',description:'Grand Anicut — one of the oldest water-diversion structures in the world, built by Cholas.',wikiTitle:'Kallanai'},
    {name:'Government Museum Trichy',lat:10.8052,lon:78.6887,type:'museum',description:'Museum housing ancient artifacts, sculptures, and geological specimens.',wikiTitle:'Government Museum, Tiruchirappalli'},
  ],
  'greater noida': [
    {name:'India Expo Centre',lat:28.4611,lon:77.5133,type:'attraction',description:'One of the largest exhibition centers in South Asia.',wikiTitle:'India Expo Centre and Mart'},
    {name:'Buddh International Circuit',lat:28.3484,lon:77.5338,type:'attraction',description:'Formula 1 racing circuit, one of the finest in Asia.',wikiTitle:'Buddh International Circuit'},
    {name:'Surajpur Bird Sanctuary',lat:28.5017,lon:77.5033,type:'park',description:'Wetland bird sanctuary with over 180 bird species.',wikiTitle:'Surajpur Wetland'},
    {name:'Great India Place Mall',lat:28.5686,lon:77.3234,type:'market',description:'One of the largest malls in North India with entertainment and shopping.',wikiTitle:'The Great India Place'},
    {name:'Akshardham Temple',lat:28.6127,lon:77.2773,type:'temple',description:'Spectacular Hindu temple complex showcasing Indian culture (nearby in Delhi).',wikiTitle:'Akshardham (Delhi)'},
    {name:'Worlds of Wonder',lat:28.5686,lon:77.3234,type:'attraction',description:'Amusement and water park with thrilling rides.',wikiTitle:'Worlds of Wonder (amusement park)'},
  ],
  amaravati: [
    {name:'Amaravati Stupa',lat:16.5725,lon:80.3572,type:'monument',description:'Ancient Buddhist stupa, one of the most important Buddhist sites in India.',wikiTitle:'Amaravati Stupa'},
    {name:'Undavalli Caves',lat:16.4961,lon:80.5810,type:'historic',description:'Rock-cut cave temples dating to 4th-5th century with monolithic Vishnu statue.',wikiTitle:'Undavalli Caves'},
    {name:'Prakasam Barrage',lat:16.5086,lon:80.6148,type:'viewpoint',description:'Dam across Krishna River connecting Vijayawada and Guntur.',wikiTitle:'Prakasam Barrage'},
    {name:'Kanaka Durga Temple',lat:16.5170,lon:80.6095,type:'temple',description:'Famous hilltop temple dedicated to Goddess Durga on Indrakeeladri hill.',wikiTitle:'Kanaka Durga Temple'},
    {name:'Bhavani Island',lat:16.5106,lon:80.5972,type:'attraction',description:'Largest river island in Krishna river with boating and water sports.',wikiTitle:'Bhavani Island'},
    {name:'Mangalagiri Temple',lat:16.4319,lon:80.5619,type:'temple',description:'Ancient hilltop temple dedicated to Lord Narasimha.',wikiTitle:'Mangalagiri'},
  ],
}

// CAMPUS + INSTITUTION MAPPING — resolves campus names to their actual city
const CAMPUS_MAP: Record<string, {city:string, lat:number, lon:number, label:string}> = {
  'srm university':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRM University, Kattankulathur (Chennai)'},
  'srm kattankulathur':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRM Kattankulathur Campus (Chennai)'},
  'srmist':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRMIST Main Campus (Chennai)'},
  'srm chennai':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRM Chennai Campus'},
  'srm trichy':{city:'Trichy',lat:10.7578,lon:78.8154,label:'SRM Trichy Campus'},
  'srm tiruchirappalli':{city:'Trichy',lat:10.7578,lon:78.8154,label:'SRM Trichy Campus'},
  'srm ncr':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM NCR Campus (Greater Noida)'},
  'srm delhi':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM Delhi NCR Campus (Greater Noida)'},
  'srm delhi ncr':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM Delhi NCR Campus (Greater Noida)'},
  'srm noida':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM NCR Campus (Greater Noida)'},
  'srm greater noida':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM NCR Campus (Greater Noida)'},
  'srm andhra':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm andhra pradesh':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm amaravati':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm ap':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm sikkim':{city:'Gangtok',lat:27.3314,lon:88.6138,label:'SRM Sikkim Campus (Gangtok)'},
  'iit madras':{city:'Chennai',lat:12.9916,lon:80.2336,label:'IIT Madras (Chennai)'},
  'iit bombay':{city:'Mumbai',lat:19.1334,lon:72.9133,label:'IIT Bombay (Mumbai)'},
  'iit delhi':{city:'Delhi',lat:28.5456,lon:77.1926,label:'IIT Delhi'},
  'vit vellore':{city:'Vellore',lat:12.9692,lon:79.1559,label:'VIT Vellore'},
  'bits pilani':{city:'Pilani',lat:28.3643,lon:75.5870,label:'BITS Pilani'},
  'anna university':{city:'Chennai',lat:13.0108,lon:80.2354,label:'Anna University (Chennai)'},
  'nit trichy':{city:'Trichy',lat:10.7601,lon:78.8137,label:'NIT Trichy'},
}

// Smart geocoding with campus/institution resolution
function resolveCampus(input: string): {city:string, lat:number, lon:number, label:string} | null {
  const key = input.toLowerCase().trim().replace(/[,.\-]/g,' ').replace(/\s+/g,' ').trim()
  
  // Pattern matching FIRST for multi-word patterns (higher priority)
  const srmMatch = key.match(/srm\s*(?:university|ist|institute)?\s*(?:,?\s*)(trichy|tiruchirappalli|tiruchi)/i)
  if (srmMatch) return CAMPUS_MAP['srm trichy']
  const srmNcr = key.match(/srm\s*(?:university|ist|institute)?\s*(?:,?\s*)(ncr|delhi|noida|greater\s*noida)/i)
  if (srmNcr) return CAMPUS_MAP['srm ncr']
  const srmAp = key.match(/srm\s*(?:university|ist|institute)?\s*(?:,?\s*)(andhra|ap|amaravati|guntur)/i)
  if (srmAp) return CAMPUS_MAP['srm andhra']
  const srmSikkim = key.match(/srm\s*(?:university|ist|institute)?\s*(?:,?\s*)(sikkim|gangtok)/i)
  if (srmSikkim) return CAMPUS_MAP['srm sikkim']
  const srmChennai = key.match(/srm\s*(?:university|ist|institute)?\s*(?:,?\s*)(chennai|kattankulathur|chengalpattu)/i)
  if (srmChennai) return CAMPUS_MAP['srmist']
  
  // Direct campus match (exact key lookup)
  for (const [campus, info] of Object.entries(CAMPUS_MAP)) {
    if (key === campus || key.includes(campus)) return info
  }
  
  // Bare "srm" without any location specifier → default to Kattankulathur
  if (/^srm\b/.test(key) && !key.includes('nagar')) return CAMPUS_MAP['srmist']
  return null
}

async function geocode(place: string): Promise<{lat:number,lon:number,name:string,resolvedCity?:string}> {
  const key = place.toLowerCase().trim().replace(/[,.\-]/g,' ').replace(/\s+/g,' ').trim()
  
  // 1. Check campus/institution mapping FIRST
  const campus = resolveCampus(key)
  if (campus) return { lat: campus.lat, lon: campus.lon, name: campus.label, resolvedCity: campus.city }
  
  // 2. Check known city coordinates — longest match first
  const sortedCities = Object.entries(CITY_COORDS).sort((a,b) => b[0].length - a[0].length)
  for (const [city, [lat,lon]] of sortedCities) {
    if (key.includes(city) || city.includes(key)) return {lat,lon,name:place, resolvedCity: city}
  }
  
  // 3. Nominatim fallback
  try {
    const r = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(place)}&format=json&limit=1`, {headers:{'User-Agent':'SmartRouteSRMIST/4.0'}})
    const d: any = await r.json()
    if (d.length) return {lat:parseFloat(d[0].lat),lon:parseFloat(d[0].lon),name:d[0].display_name?.split(',')[0]||place, resolvedCity: d[0].display_name?.split(',')[0]||place}
  } catch(e) {}
  
  return {lat:20.5937,lon:78.9629,name:place, resolvedCity: place}
}

async function fetchAttractions(lat: number, lon: number, city: string, days: number): Promise<any[]> {
  const needed = days * 5
  const cityKey = city.toLowerCase().trim().replace(/[^a-z\s]/g,'').replace(/\s+/g,' ')
  
  // 1. Start with curated top attractions for known cities
  let places: any[] = []
  for (const [key, attractions] of Object.entries(CITY_TOP_ATTRACTIONS)) {
    // Check if cityKey matches, or if the campus-resolved city matches
    if (cityKey.includes(key) || key.includes(cityKey) || 
        cityKey.split(' ').some(w => w.length > 3 && key.includes(w))) {
      places = [...attractions]
      break
    }
  }
  
  // 2. Supplement with Overpass API for additional/unknown cities
  if (places.length < needed) {
    const radius = Math.min(30000, 10000 + days * 3000)
    // Improved Overpass query: target only major tourist attractions with names
    const query = `[out:json][timeout:25];(
      node(around:${radius},${lat},${lon})[tourism~"^(attraction|museum|gallery|viewpoint|zoo|theme_park|artwork)$"]["name"];
      node(around:${radius},${lat},${lon})[tourism="yes"]["name"];
      node(around:${radius},${lat},${lon})[historic~"^(monument|memorial|castle|fort|ruins|archaeological_site|palace)$"]["name"];
      node(around:${radius},${lat},${lon})[amenity="place_of_worship"]["name"]["tourism"];
      node(around:${radius},${lat},${lon})[leisure~"^(park|garden|nature_reserve|beach_resort)$"]["name"];
      way(around:${radius},${lat},${lon})[tourism~"^(attraction|museum|gallery|viewpoint)$"]["name"];
      way(around:${radius},${lat},${lon})[historic~"^(monument|memorial|castle|fort|ruins|palace)$"]["name"];
    );out center 60;`
    
    try {
      const r = await fetch('https://overpass-api.de/api/interpreter', {method:'POST', body:`data=${encodeURIComponent(query)}`, headers:{'Content-Type':'application/x-www-form-urlencoded','User-Agent':'SmartRouteSRMIST/3.0'}})
      const d: any = await r.json()
      const seen = new Set<string>(places.map(p => p.name.toLowerCase().replace(/\s+/g,'')))
      for (const el of (d.elements||[])) {
        const tags = el.tags || {}
        const name = tags['name:en'] || tags.name || ''
        if (!name || name.length < 3) continue
        const nKey = name.toLowerCase().replace(/\s+/g,'')
        if (seen.has(nKey)) continue
        // Filter out generic/irrelevant items: street names, person names (George V, etc.)
        if (/^(statue|bust|plaque|bench|sign|information|george|king|queen|prince|princess|memorial (to|of)|tomb of unknown)/i.test(name)) continue
        if (name.length < 5 && !tags.tourism) continue // skip very short generic names
        seen.add(nKey)
        const plat = el.lat || el.center?.lat
        const plon = el.lon || el.center?.lon
        if (!plat || !plon) continue
        const ptype = tags.tourism || tags.historic || tags.leisure || 'attraction'
        places.push({
          name, lat: plat, lon: plon, type: ptype,
          description: tags.description || tags['description:en'] || `${ptype.replace(/_/g,' ')} in ${city}`,
          wikiTitle: tags.wikipedia?.split(':')[1] || tags.wikidata || name,
          opening_hours: tags.opening_hours || '',
          phone: tags.phone || '',
          website: tags.website || '',
          wheelchair: tags.wheelchair || '',
          fee: tags.fee || '',
        })
        if (places.length >= needed + 10) break
      }
    } catch(e) { console.error('Overpass error:', e) }
  }

  // 3. Supplement with OpenTripMap if still short
  if (places.length < needed) {
    try {
      const radius2 = Math.min(30000, 10000 + days * 3000)
      const r2 = await fetch(`https://api.opentripmap.com/0.1/en/places/radius?radius=${radius2}&lon=${lon}&lat=${lat}&kinds=interesting_places,cultural,historic,natural,architecture&format=json&limit=${needed - places.length + 5}&rate=3&apikey=5ae2e3f221c38a28845f05b6aec53ea2b07e9e48b7f89b38bd76ca73`)
      const otm: any = await r2.json()
      const seen2 = new Set(places.map(p => p.name.toLowerCase().replace(/\s+/g,'')))
      for (const p of (otm||[])) {
        if (!p.name || p.name.length < 4) continue
        const nKey = p.name.toLowerCase().replace(/\s+/g,'')
        if (seen2.has(nKey)) continue
        seen2.add(nKey)
        places.push({
          name: p.name, lat: p.point?.lat||lat, lon: p.point?.lon||lon,
          type: p.kinds?.split(',')[0] || 'attraction',
          description: `${p.kinds?.split(',')[0]?.replace(/_/g,' ') || 'attraction'} in ${city}`,
          wikiTitle: p.wikipedia || p.name, rating: p.rate || 0,
        })
        if (places.length >= needed + 5) break
      }
    } catch(e) {}
  }
  return places
}

async function fetchWeather(lat: number, lon: number, days: number): Promise<any[]> {
  try {
    const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max,uv_index_max&hourly=relativehumidity_2m&current_weather=true&timezone=auto&forecast_days=${Math.min(days+1,16)}`)
    const d: any = await r.json()
    const daily = d.daily || {}
    const result: any[] = []
    for (let i = 0; i < Math.min(days, (daily.time||[]).length); i++) {
      const tMax = daily.temperature_2m_max?.[i] || 30
      const tMin = daily.temperature_2m_min?.[i] || 20
      const precip = daily.precipitation_sum?.[i] || 0
      const wcode = daily.weathercode?.[i] || 0
      const wind = daily.windspeed_10m_max?.[i] || 10
      const uv = daily.uv_index_max?.[i] || 5
      const humidity = d.hourly?.relativehumidity_2m?.[i*24+12] || 55
      const risk = precip > 10 || wcode >= 60 ? 'high' : precip > 2 || wcode >= 40 ? 'medium' : 'low'
      const icon = wcode <= 1 ? '☀️' : wcode <= 3 ? '⛅' : wcode <= 50 ? '☁️' : wcode <= 70 ? '🌧️' : wcode <= 80 ? '🌦️' : '⛈️'
      const bayes = classifyWeather(tMax, humidity, wcode*1.2, precip)
      result.push({ day: i+1, date: daily.time?.[i], temp_max: tMax, temp_min: tMin, precipitation: precip, weathercode: wcode, wind, uv, humidity, risk_level: risk, icon, classification: bayes })
    }
    return result
  } catch(e) { return [] }
}

async function fetchWikiPhoto(name: string): Promise<string> {
  // Try exact title first, then search
  const attempts = [name, name.replace(/\s+(temple|fort|beach|palace|museum|church|mosque|garden|park|lake)/i, ' ($1)')]
  for (const title of attempts) {
    try {
      const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(title)}&prop=pageimages&piprop=thumbnail&pithumbsize=600&redirects=1&origin=*`)
      const d: any = await r.json()
      const pages = d?.query?.pages || {}
      for (const p of Object.values(pages) as any[]) {
        if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg') && !p.thumbnail.source.includes('Flag_of')) return p.thumbnail.source
      }
    } catch(e) {}
  }
  // Try Wikipedia search API as fallback
  try {
    const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search&gsrsearch=${encodeURIComponent(name)}&gsrlimit=3&prop=pageimages&piprop=thumbnail&pithumbsize=600&origin=*`)
    const d: any = await r.json()
    const pages = d?.query?.pages || {}
    for (const p of Object.values(pages) as any[]) {
      if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg') && !p.thumbnail.source.includes('Flag_of')) return p.thumbnail.source
    }
  } catch(e) {}
  return ''
}

// ============================================
// BUILD ITINERARY
// ============================================
function buildItinerary(places: any[], weather: any[], days: number, budget: number, city: string, persona: string, origin: string, originCoords: any): any {
  const perDay = Math.max(3, Math.min(6, Math.ceil(places.length / days)))
  const dailyBudget = budget / days
  const itinDays: any[] = []
  let usedNames = new Set<string>()
  let totalCost = 0
  const agentLog: any[] = []

  // Agent 1: Planner Agent — MCTS + Thompson Sampling
  agentLog.push({agent:'planner', action:'initialize', msg:`Starting MCTS optimization for ${city}, ${days} days, budget ₹${budget}`})

  // Get personalized preferences via Thompson Sampling
  const preferences = getThompsonPreferences()
  agentLog.push({agent:'preference', action:'thompson_sample', msg:`Sampled preferences: ${Object.entries(preferences).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([k,v])=>`${k}:${(v*100).toFixed(0)}%`).join(', ')}`})

  // Track all dense rewards for this episode
  const episodeDenseRewards: number[] = []
  const usedTypes = new Set<string>()

  for (let d = 0; d < days; d++) {
    // Agent 2: Weather Agent — classify each day
    const w = weather[d] || {}
    const weatherSafe = (w.risk_level || 'low') !== 'high'
    
    // Select appropriate places, weighted by Thompson preferences
    let dayPlaces = places.filter(p => !usedNames.has(p.name))
    
    // Score and sort places using Bayesian preferences
    dayPlaces = dayPlaces.map(p => {
      const catPref = preferences[p.type] || preferences['cultural'] || 0.5
      const baseScore = catPref * 0.4 + (p.rating || 4) / 5 * 0.3 + Math.random() * 0.3
      return {...p, _score: baseScore}
    }).sort((a: any, b: any) => (b._score || 0) - (a._score || 0)).slice(0, perDay)
    
    dayPlaces.forEach(p => usedNames.add(p.name))
    
    // Agent 3: Crowd Analyzer — predict crowd per time slot
    agentLog.push({agent:'crowd', action:'predict', msg:`Day ${d+1}: Crowd predictions generated for ${dayPlaces.length} time slots`})
    
    // MCTS optimize order
    const optimized = mctsOptimize(dayPlaces, weather, dailyBudget)
    
    const activities: any[] = []
    let startHour = 9
    for (const place of optimized) {
      const duration = place.type === 'museum' ? 2 : place.type === 'park' ? 1.5 : place.type === 'beach' ? 2.5 : 1.5
      const cost = persona === 'luxury' ? Math.round(300 + Math.random()*500) : persona === 'adventure' ? Math.round(100 + Math.random()*300) : Math.round(50 + Math.random()*200)
      const crowd = crowdHeuristic(startHour)
      const actWeatherSafe = weatherSafe
      
      usedTypes.add(place.type || 'attraction')
      
      // Compute dense reward for this activity
      const denseR = computeDenseReward({
        rating: place.rating || 4,
        budgetAdherence: Math.max(0, 1 - Math.abs(cost - dailyBudget/perDay) / (dailyBudget/perDay)),
        weatherSafety: actWeatherSafe ? 0.9 : 0.3,
        crowdLevel: crowd,
        timeEfficiency: Math.min(1, duration / 2),
        diversityBonus: usedTypes.size / 5,
      })
      episodeDenseRewards.push(denseR)
      
      // Q-Learning: select action for this state, then update
      const stateKey = `${city}|d${d+1}|${place.type}|crowd${Math.round(crowd/20)*20}`
      const nextStateKey = `${city}|d${d+1}|next`
      const action = qSelect(stateKey)
      const qlResult = qUpdate(stateKey, action, denseR, nextStateKey)
      
      // POMDP update based on activity conditions
      if (crowd < 30) pomdpUpdate('low_crowd')
      else if (crowd > 70) pomdpUpdate('high_crowd')
      if (actWeatherSafe) pomdpUpdate('good_weather')
      else pomdpUpdate('bad_weather')
      
      activities.push({
        name: place.name, lat: place.lat, lon: place.lon, type: place.type,
        description: place.description, time: `${String(Math.floor(startHour)).padStart(2,'0')}:${startHour%1?'30':'00'}`,
        duration: `${duration}h`, cost, crowd_level: crowd,
        weather_safe: actWeatherSafe, weather_warning: !actWeatherSafe ? `⚠️ ${w.icon} Weather risk` : '',
        wikiTitle: place.wikiTitle || place.name, opening_hours: place.opening_hours || '',
        phone: place.phone || '', website: place.website || '', wheelchair: place.wheelchair || '',
        rating: place.rating || (3.5 + Math.random()*1.5),
        notes: '',
        // RL metadata
        rl: { action, denseReward: denseR, tdError: qlResult.tdError, qValue: qlResult.newQ }
      })
      totalCost += cost
      startHour += duration + 0.5
    }

    itinDays.push({
      day: d+1, city, date: weather[d]?.date || '',
      weather: weather[d] || {icon:'☀️',temp_max:30,temp_min:22,risk_level:'low'},
      activities,
      dayBudget: Math.round(activities.reduce((s,a) => s+a.cost, 0)),
      dayNotes: '',
    })
  }

  // Budget breakdown
  const accommodation = Math.round(budget * (persona==='luxury'?0.4:0.3))
  const food = Math.round(budget * 0.2)
  const transport = Math.round(budget * 0.15)
  const activityBudget = Math.round(budget * 0.25)
  const emergency = Math.round(budget * 0.1)

  // Compute sparse reward for the complete trip
  const totalActivities = itinDays.reduce((s: number,d: any) => s + d.activities.length, 0)
  const avgRating = itinDays.flatMap((d: any) => d.activities).reduce((s: number,a: any) => s + (a.rating||4), 0) / Math.max(totalActivities, 1)
  const goodWeatherDays = weather.filter((w: any) => w.risk_level !== 'high').length
  const budgetUtil = (totalCost + accommodation + food) / budget
  
  const sparseR = computeSparseReward({
    tripCompleted: true,
    totalActivities,
    budgetUtilization: budgetUtil,
    weatherDaysGood: goodWeatherDays,
    totalDays: days,
    avgRating,
    uniqueTypes: usedTypes.size,
    userSatisfaction: 0, // Will be updated when user rates
  })
  
  // Combined reward
  const avgDense = episodeDenseRewards.length ? episodeDenseRewards.reduce((s,r) => s+r, 0) / episodeDenseRewards.length : 0
  const totalReward = computeTotalReward(avgDense, sparseR)
  
  aiState.episode++
  agentLog.push({agent:'explain', action:'episode_complete', msg:`Episode ${aiState.episode}: dense=${avgDense.toFixed(3)}, sparse=${sparseR.toFixed(3)}, total=${totalReward.toFixed(3)}, ε=${aiState.epsilon.toFixed(3)}`})

  // Store agent decisions
  aiState.agentDecisions.push({
    episode: aiState.episode, city, days, persona, totalReward,
    actions: itinDays.flatMap((d: any) => d.activities.map((a: any) => a.rl?.action)).filter(Boolean)
  })

  return {
    destination: city, origin, days, budget, persona,
    totalCost: totalCost + accommodation + food,
    originCoords, destCoords: {lat: places[0]?.lat, lon: places[0]?.lon},
    budgetBreakdown: { accommodation, food, activities: activityBudget, transport, emergency },
    days_data: itinDays, weather,
    agentLog,
    ai: {
      mcts_iterations: 50, 
      q_table_size: Object.keys(aiState.qTable).length,
      bayesian: aiState.bayesian, 
      dirichlet: aiState.dirichlet,
      pomdp_belief: aiState.pomdpBelief, 
      denseRewards: aiState.denseRewards.slice(-30),
      sparseRewards: aiState.sparseRewards.slice(-20),
      totalRewards: aiState.totalRewards.slice(-30),
      cumulativeReward: aiState.cumulativeReward,
      epsilon: aiState.epsilon,
      episode: aiState.episode,
      totalSteps: aiState.totalSteps,
      alpha: aiState.alpha,
      gamma: aiState.gamma,
      thompsonPrefs: getThompsonPreferences(),
    }
  }
}

// ============================================
// MULTI-CITY TRIP (from TripSage concept)
// ============================================
async function buildMultiCityTrip(cities: string[], daysPerCity: number[], totalBudget: number, persona: string, origin: string): Promise<any> {
  const cityResults: any[] = []
  const budgetPerCity = totalBudget / cities.length
  let originCoords = origin ? await geocode(origin) : {lat:13.08,lon:80.27,name:'Chennai'}

  for (let i = 0; i < cities.length; i++) {
    const city = cities[i]
    const days = daysPerCity[i] || 2
    const destGeo = await geocode(city)
    const [attractions, weather] = await Promise.all([
      fetchAttractions(destGeo.lat, destGeo.lon, city, days),
      fetchWeather(destGeo.lat, destGeo.lon, days)
    ])
    const topPlaces = attractions.slice(0, 8)
    const photos = await Promise.all(topPlaces.map(p => fetchWikiPhoto(p.wikiTitle || p.name)))
    topPlaces.forEach((p, j) => { if (photos[j]) p.photo = photos[j] })
    
    const itinerary = buildItinerary(attractions, weather, days, budgetPerCity, city, persona, i===0 ? origin : cities[i-1], i===0 ? originCoords : {lat: cityResults[i-1]?.destCoords?.lat, lon: cityResults[i-1]?.destCoords?.lon})
    const langTips = getLanguageTips(city)
    cityResults.push({ ...itinerary, languageTips: langTips, cityOrder: i+1, photos: topPlaces.filter(p=>p.photo).map(p=>({name:p.name,url:p.photo})) })
  }

  return {
    isMultiCity: true,
    cities: cities,
    totalDays: daysPerCity.reduce((s,d)=>s+d,0),
    totalBudget,
    persona,
    origin,
    cityItineraries: cityResults,
    transitInfo: cities.map((c,i) => i < cities.length-1 ? {from: c, to: cities[i+1], type: 'auto'} : null).filter(Boolean)
  }
}

// ============================================
// BOOKING ENGINE — Realistic Prices & Real Links
// ============================================

// Approximate distances between major Indian cities (km) for price estimation
const CITY_DISTANCES: Record<string, Record<string, number>> = {
  chennai: {delhi:2180,mumbai:1340,jaipur:2000,goa:600,bangalore:350,hyderabad:630,kolkata:1660,agra:2100,varanasi:1680,udaipur:1670,kochi:600,shimla:2550,manali:2700,pondicherry:150,amritsar:2600,jodhpur:1950,leh:3200,darjeeling:1900,ooty:280,mysore:480,mahabalipuram:60,madurai:460,thanjavur:340,kodaikanal:430,rishikesh:2350,hampi:580,munnar:500,alleppey:640,tirupati:140,srinagar:3100},
  delhi: {mumbai:1400,jaipur:280,goa:1900,bangalore:2150,hyderabad:1500,kolkata:1500,agra:230,varanasi:820,udaipur:670,kochi:2700,shimla:350,manali:530,chennai:2180,pondicherry:2300,amritsar:470,jodhpur:590,leh:1000,darjeeling:1550,rishikesh:250,haridwar:220,srinagar:850},
  mumbai: {jaipur:1150,goa:590,bangalore:980,hyderabad:710,kolkata:2050,agra:1220,varanasi:1330,udaipur:660,kochi:1500,delhi:1400,chennai:1340,pondicherry:1490,amritsar:1840,jodhpur:830,shimla:1750,manali:1850},
  bangalore: {mysore:150,ooty:275,kochi:550,chennai:350,hyderabad:570,goa:560,mumbai:980,hampi:340,coorg:250},
  kolkata: {darjeeling:600,gangtok:640,varanasi:680,delhi:1500,chennai:1660,mumbai:2050},
}

function getDistance(origin: string, dest: string): number {
  const oKey = origin.toLowerCase().replace(/[^a-z]/g,'')
  const dKey = dest.toLowerCase().replace(/[^a-z]/g,'')
  for (const [city, dists] of Object.entries(CITY_DISTANCES)) {
    if (oKey.includes(city) || city.includes(oKey)) {
      for (const [d, km] of Object.entries(dists)) {
        if (dKey.includes(d) || d.includes(dKey)) return km
      }
    }
  }
  // Fallback: rough estimate based on coordinates
  return 800 + Math.floor(Math.random() * 500)
}

function generateFlights(origin: string, dest: string, date: string): any[] {
  const dist = getDistance(origin, dest)
  const airlines = [
    {name:'IndiGo',code:'6E',base:1.8,rating:4.0},
    {name:'Air India',code:'AI',base:2.2,rating:3.8},
    {name:'Vistara',code:'UK',base:2.5,rating:4.3},
    {name:'SpiceJet',code:'SG',base:1.6,rating:3.6},
    {name:'AirAsia India',code:'I5',base:1.5,rating:3.7},
    {name:'Akasa Air',code:'QP',base:1.7,rating:4.1},
  ]
  // Pre-filled search URLs with actual trip details
  const dateParam = date || new Date().toISOString().split('T')[0]
  const oEnc = encodeURIComponent(origin)
  const dEnc = encodeURIComponent(dest)
  
  return airlines.map((airline, i) => {
    const basePrice = Math.round(dist * airline.base + 500 + (Math.random() * 400 - 200))
    const flightNo = `${airline.code}-${100 + Math.floor(Math.random()*900)}`
    const depH = [6,7,8,10,14,17,20][i % 7]
    const durH = Math.max(1, Math.round(dist / 700))
    const durM = Math.random() > 0.5 ? 15 : 45
    const isNonstop = dist < 1500
    const classes = [
      {type:'Economy',multiplier:1},
      {type:'Premium Economy',multiplier:1.6},
      {type:'Business',multiplier:3},
    ]
    const cls = classes[i < 3 ? 0 : i < 5 ? 1 : 2]
    const price = Math.round(basePrice * cls.multiplier)
    
    return {
      id: `FL${Date.now()}${i}`, airline: airline.name, flight_no: flightNo,
      departure: `${String(depH).padStart(2,'0')}:${Math.random()>0.5?'00':'30'}`,
      arrival: `${String((depH+durH)%24).padStart(2,'0')}:${durM>30?'45':'15'}`,
      duration: `${durH}h ${durM}m`, price, currency: '₹',
      class: cls.type,
      stops: isNonstop ? 0 : (Math.random() > 0.6 ? 1 : 0),
      rating: airline.rating.toFixed(1),
      bookingPlatforms: [
        {name:'Google Flights', url: `https://www.google.com/travel/flights?q=flights+from+${oEnc}+to+${dEnc}+on+${dateParam}&curr=INR`, icon:'google', prefilled:true},
        {name:'MakeMyTrip', url: `https://www.makemytrip.com/flight/search?itinerary=${oEnc}-${dEnc}-${dateParam.replace(/-/g,'/')}&tripType=O&paxType=A-1_C-0_I-0&intl=false&cabinClass=E`, prefilled:true},
        {name:'Skyscanner', url: `https://www.skyscanner.co.in/transport/flights/${oEnc}/${dEnc}/${dateParam.replace(/-/g,'')}/?adultsv2=1&cabinclass=economy&childrenv2=&ref=home`, prefilled:true},
        {name:'ixigo', url: `https://www.ixigo.com/search/result/flight?from=${oEnc}&to=${dEnc}&date=${dateParam}&adults=1&children=0&infants=0&class=e&source=Search+Form`, prefilled:true},
        {name:'Cleartrip', url: `https://www.cleartrip.com/flights/results?adults=1&childs=0&infants=0&class=Economy&depart_date=${dateParam}&from=${oEnc}&to=${dEnc}&intl=false`, prefilled:true},
        {name:'EaseMyTrip', url: `https://flight.easemytrip.com/FlightList/Index?from=${oEnc}&to=${dEnc}&ddate=${dateParam}&isow=true&isdm=true&adult=1&child=0&infant=0&sc=E`, prefilled:true},
      ]
    }
  }).sort((a,b) => a.price - b.price)
}

function generateTrains(origin: string, dest: string): any[] {
  const dist = getDistance(origin, dest)
  const irctcUrl = `https://www.irctc.co.in/nget/train-search`
  const trainTypes = [
    {name:'Rajdhani Express',code:'RAJ',speedKmh:100,base:1.5,classes:['1A','2A','3A']},
    {name:'Shatabdi Express',code:'SHT',speedKmh:90,base:1.2,classes:['CC','EC']},
    {name:'Vande Bharat Express',code:'VBE',speedKmh:130,base:1.8,classes:['CC','EC']},
    {name:'Duronto Express',code:'DUR',speedKmh:85,base:1.3,classes:['1A','2A','3A','SL']},
    {name:'Garib Rath',code:'GR',speedKmh:75,base:0.7,classes:['3A','SL']},
    {name:'Superfast Express',code:'SF',speedKmh:70,base:0.8,classes:['2A','3A','SL']},
  ]
  const confirmtktUrl = `https://www.confirmtkt.com/train-search?from=${encodeURIComponent(origin)}&to=${encodeURIComponent(dest)}`
  
  return trainTypes.filter(t => {
    if (dist < 300 && t.code === 'RAJ') return false
    if (dist > 1500 && t.code === 'SHT') return false
    return true
  }).map((train, i) => {
    const durH = Math.max(2, Math.round(dist / train.speedKmh))
    const cls = train.classes[0]
    const classMultipliers: Record<string,number> = {'1A':3.5,'2A':2.2,'3A':1.5,'SL':0.7,'CC':1.8,'EC':2.5}
    const price = Math.round(dist * train.base * (classMultipliers[cls] || 1))
    const depH = [5,6,8,15,17,20][i % 6]
    
    return {
      id: `TR${Date.now()}${i}`, train_name: train.name,
      train_no: `${10000+Math.floor(Math.random()*89999)}`,
      departure: `${String(depH).padStart(2,'0')}:${Math.random()>0.5?'00':'30'}`,
      duration: `${durH}h ${Math.random()>0.5?'00':'30'}m`, price, currency: '₹',
      class: cls,
      bookingUrl: irctcUrl,
      bookingPlatforms: [
        {name:'IRCTC', url: irctcUrl},
        {name:'ConfirmTkt', url: confirmtktUrl},
        {name:'RailYatri', url: `https://www.railyatri.in/booking/search?from=${encodeURIComponent(origin)}&to=${encodeURIComponent(dest)}`},
        {name:'ixigo Trains', url: `https://www.ixigo.com/search/result/train/${encodeURIComponent(origin)}/${encodeURIComponent(dest)}/`},
        {name:'MakeMyTrip Trains', url: `https://www.makemytrip.com/railways/`},
        {name:'Cleartrip Trains', url: `https://www.cleartrip.com/trains`},
      ]
    }
  }).sort((a,b) => a.price - b.price)
}

function generateHotels(city: string, days: number, persona: string): any[] {
  const budget_hotels = [
    {name:`OYO Rooms ${city}`,stars:2,basePrice:600,rating:3.4,amenities:['WiFi','AC']},
    {name:`Treebo ${city} Central`,stars:3,basePrice:900,rating:3.7,amenities:['WiFi','AC','Breakfast']},
    {name:`FabHotel ${city}`,stars:3,basePrice:800,rating:3.5,amenities:['WiFi','AC','Parking']},
  ]
  const mid_hotels = [
    {name:`Lemon Tree ${city}`,stars:3,basePrice:2500,rating:4.0,amenities:['WiFi','AC','Breakfast','Pool','Gym']},
    {name:`Radisson ${city}`,stars:4,basePrice:4000,rating:4.2,amenities:['WiFi','AC','Breakfast','Pool','Gym','Spa']},
    {name:`Novotel ${city}`,stars:4,basePrice:3500,rating:4.1,amenities:['WiFi','AC','Breakfast','Pool','Restaurant']},
  ]
  const luxury_hotels = [
    {name:`Taj ${city}`,stars:5,basePrice:8000,rating:4.7,amenities:['WiFi','AC','Breakfast','Pool','Gym','Spa','Restaurant','Bar','Concierge']},
    {name:`ITC ${city}`,stars:5,basePrice:7000,rating:4.6,amenities:['WiFi','AC','Breakfast','Pool','Gym','Spa','Restaurant']},
    {name:`The Leela ${city}`,stars:5,basePrice:9000,rating:4.8,amenities:['WiFi','AC','Breakfast','Pool','Gym','Spa','Butler']},
  ]
  
  let hotels = persona === 'luxury' ? [...luxury_hotels, ...mid_hotels] : persona === 'adventure' ? [...budget_hotels, ...mid_hotels] : [...budget_hotels, ...mid_hotels, ...luxury_hotels.slice(0,1)]
  
  const checkinDate = new Date().toISOString().split('T')[0]
  const checkoutDate = new Date(Date.now()+days*86400000).toISOString().split('T')[0]
  const cityEnc = encodeURIComponent(city)
  const searchUrl = `https://www.booking.com/searchresults.html?ss=${cityEnc}&checkin=${checkinDate}&checkout=${checkoutDate}&group_adults=2&no_rooms=1`
  
  return hotels.map((h, i) => {
    const priceVariation = 0.8 + Math.random() * 0.4
    const ppn = Math.round(h.basePrice * priceVariation)
    return {
      id: `HT${Date.now()}${i}`, name: h.name, stars: h.stars,
      price_per_night: ppn,
      total_price: ppn * days,
      rating: h.rating.toFixed(1), amenities: h.amenities,
      bookingUrl: searchUrl,
      bookingPlatforms: [
        {name:'Booking.com', url: searchUrl, prefilled:true},
        {name:'MakeMyTrip', url: `https://www.makemytrip.com/hotels/hotel-listing?city=${cityEnc}&checkin=${checkinDate.replace(/-/g,'')}&checkout=${checkoutDate.replace(/-/g,'')}&roomStayQualifier=2e0e`, prefilled:true},
        {name:'Goibibo', url: `https://www.goibibo.com/hotels/hotels-in-${city.toLowerCase().replace(/\s+/g,'-')}/?checkin=${checkinDate}&checkout=${checkoutDate}&adults_count=2&rooms_count=1`, prefilled:true},
        {name:'Agoda', url: `https://www.agoda.com/search?city=${cityEnc}&checkIn=${checkinDate}&checkOut=${checkoutDate}&rooms=1&adults=2`, prefilled:true},
        {name:'Trivago', url: `https://www.trivago.in/en-IN/srl?search=${cityEnc}&dr=${checkinDate}--${checkoutDate}&pa=2`, prefilled:true},
        {name:'OYO', url: `https://www.oyorooms.com/search?location=${cityEnc}&checkin=${checkinDate}&checkout=${checkoutDate}`, prefilled:true},
      ],
      image: '', currency: '₹',
    }
  }).sort((a,b) => a.price_per_night - b.price_per_night)
}

function generateCabs(city: string): any[] {
  const providers = [
    {name:'Ola',types:[{type:'Micro',baseFare:40,perKm:7},{type:'Mini',baseFare:60,perKm:9},{type:'Sedan',baseFare:90,perKm:12},{type:'SUV',baseFare:120,perKm:15}],url:'https://www.olacabs.com'},
    {name:'Uber',types:[{type:'UberGo',baseFare:50,perKm:8},{type:'Uber Premier',baseFare:80,perKm:11},{type:'UberXL',baseFare:110,perKm:14},{type:'Auto',baseFare:25,perKm:5}],url:'https://www.uber.com'},
    {name:'Rapido',types:[{type:'Bike',baseFare:15,perKm:4},{type:'Auto',baseFare:25,perKm:5}],url:'https://www.rapido.bike'},
  ]
  
  const results: any[] = []
  for (const prov of providers) {
    for (const t of prov.types) {
      results.push({
        id: `CB${Date.now()}${results.length}`, provider: prov.name,
        type: t.type,
        price_per_km: t.perKm,
        base_fare: t.baseFare,
        bookingUrl: prov.url,
        estimated_10km: t.baseFare + t.perKm * 10,
      })
    }
  }
  return results.sort((a,b) => a.estimated_10km - b.estimated_10km)
}

function generateRestaurants(city: string, lat: number, lon: number): any[] {
  const cuisines = ['South Indian','North Indian','Chinese','Continental','Street Food','Biryani','Seafood','Italian']
  return Array.from({length:8},(_,i) => ({
    id: `RS${Date.now()}${i}`, name: `${['Spice','Royal','Golden','Green','Silver','Paradise','Annapoorna','Saravana'][i]} ${['Kitchen','Restaurant','Diner','Cafe','Bhavan','Palace','Bistro','Garden'][i]}`,
    cuisine: cuisines[i%cuisines.length], rating: (3.5+Math.random()*1.5).toFixed(1),
    price_range: ['₹','₹₹','₹₹₹'][Math.floor(Math.random()*3)],
    avgCost: Math.round(150+Math.random()*500), lat: lat+Math.random()*0.02-0.01, lon: lon+Math.random()*0.02-0.01,
    bookingUrl: `https://www.zomato.com/${city.toLowerCase()}`,
  }))
}

// ============================================
// LANGUAGE TIPS
// ============================================
function getLanguageTips(city: string): any {
  const regionMap: Record<string,any> = {
    chennai: {language:'Tamil',phrases:[{phrase:'Vanakkam',meaning:'Hello',pronunciation:'va-NAK-kam'},{phrase:'Nandri',meaning:'Thank You',pronunciation:'NAN-dri'},{phrase:'Evvalavu?',meaning:'How much?',pronunciation:'ev-va-LA-vu'},{phrase:'Sapadu',meaning:'Food',pronunciation:'SAA-pa-du'},{phrase:'Thanni',meaning:'Water',pronunciation:'THAN-ni'},{phrase:'Illa',meaning:'No',pronunciation:'IL-la'},{phrase:'Aamaa',meaning:'Yes',pronunciation:'AA-maa'}]},
    mumbai: {language:'Hindi/Marathi',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Dhanyavaad',meaning:'Thank You',pronunciation:'dhan-ya-VAAD'},{phrase:'Kitna?',meaning:'How much?',pronunciation:'KIT-na'},{phrase:'Khaana',meaning:'Food',pronunciation:'KHAA-na'},{phrase:'Paani',meaning:'Water',pronunciation:'PAA-ni'}]},
    jaipur: {language:'Hindi/Rajasthani',phrases:[{phrase:'Khamma Ghani',meaning:'Hello (Rajasthani)',pronunciation:'KHAM-ma GHA-ni'},{phrase:'Shukriya',meaning:'Thank You',pronunciation:'shuk-RI-ya'},{phrase:'Kitna hai?',meaning:'How much?',pronunciation:'KIT-na hai'}]},
    delhi: {language:'Hindi',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Shukriya',meaning:'Thank You',pronunciation:'shuk-RI-ya'},{phrase:'Kidhar hai?',meaning:'Where is it?',pronunciation:'KID-har hai'},{phrase:'Kitne ka hai?',meaning:'How much?',pronunciation:'KIT-ne ka hai'}]},
    kolkata: {language:'Bengali',phrases:[{phrase:'Nomoshkar',meaning:'Hello',pronunciation:'no-mosh-KAR'},{phrase:'Dhonnobad',meaning:'Thank You',pronunciation:'dhon-no-BAD'},{phrase:'Koto dam?',meaning:'How much?',pronunciation:'ko-to DAM'}]},
    bangalore: {language:'Kannada',phrases:[{phrase:'Namaskara',meaning:'Hello',pronunciation:'na-mas-KA-ra'},{phrase:'Dhanyavadagalu',meaning:'Thank You',pronunciation:'dhan-ya-VA-da-ga-lu'},{phrase:'Eshthu?',meaning:'How much?',pronunciation:'ESH-thu'}]},
    hyderabad: {language:'Telugu/Urdu',phrases:[{phrase:'Namaskaaram',meaning:'Hello',pronunciation:'na-mas-KAA-ram'},{phrase:'Dhanyavaadaalu',meaning:'Thank You',pronunciation:'dhan-ya-VAA-daa-lu'}]},
    kochi: {language:'Malayalam',phrases:[{phrase:'Namaskaram',meaning:'Hello',pronunciation:'na-mas-KA-ram'},{phrase:'Nanni',meaning:'Thank You',pronunciation:'NAN-ni'},{phrase:'Ethra?',meaning:'How much?',pronunciation:'ETH-ra'}]},
  }
  const key = city.toLowerCase().replace(/[^a-z]/g,'')
  for (const [c, data] of Object.entries(regionMap)) { if (key.includes(c) || c.includes(key)) return data }
  return {language:'Hindi (default)',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Dhanyavaad',meaning:'Thank You',pronunciation:'dhan-ya-VAAD'},{phrase:'Kitna?',meaning:'How much?',pronunciation:'KIT-na'},{phrase:'Khaana',meaning:'Food',pronunciation:'KHAA-na'},{phrase:'Paani',meaning:'Water',pronunciation:'PAA-ni'},{phrase:'Haan',meaning:'Yes',pronunciation:'HAAN'},{phrase:'Nahi',meaning:'No',pronunciation:'na-HI'},{phrase:'Madat',meaning:'Help',pronunciation:'MA-dat'}]}
}

// ============================================
// PACKING LIST GENERATOR (from NOMAD)
// ============================================
function generatePackingList(days: number, weather: any[], persona: string): any {
  const categories: any = {
    'Essentials': ['Passport/ID','Phone + Charger','Power Bank','Cash + Cards','Travel Insurance Docs','Medicines'],
    'Clothing': [`${days+1} T-shirts/Tops`,`${days} Pants/Shorts`,'Comfortable Walking Shoes','Sleepwear','Undergarments'],
    'Toiletries': ['Toothbrush + Paste','Sunscreen SPF 50','Deodorant','Hand Sanitizer','Wet Wipes','Lip Balm'],
    'Tech': ['Phone Charger','Earphones','Camera (optional)','Universal Adapter'],
    'Travel Comfort': ['Neck Pillow','Eye Mask','Reusable Water Bottle','Snacks'],
  }
  const hasRain = weather.some(w => w.risk_level === 'high' || w.precipitation > 5)
  const hasHeat = weather.some(w => w.temp_max > 35)
  const hasCold = weather.some(w => w.temp_min < 15)
  
  if (hasRain) { categories['Weather Prep'] = ['Umbrella/Raincoat','Waterproof Bag','Quick-dry Towel'] }
  if (hasHeat) { categories['Clothing'].push('Hat/Cap','Sunglasses'); categories['Toiletries'].push('After-sun Lotion') }
  if (hasCold) { categories['Clothing'].push('Jacket/Sweater','Warm Socks','Gloves') }
  if (persona === 'adventure') { categories['Adventure Gear'] = ['Hiking Boots','Daypack','First Aid Kit','Torch/Headlamp','Compass','Insect Repellent'] }
  if (persona === 'luxury') { categories['Luxury'] = ['Formal Outfit','Jewelry','Premium Toiletry Kit','Travel Pillow (Memory Foam)'] }
  if (persona === 'family') { categories['Family Essentials'] = ['Kids Snacks','Entertainment for Children','First Aid Kit','Baby Wipes','Extra Bags'] }
  
  return categories
}

// ============================================
// EMERGENCY CONTACTS DATABASE
// ============================================
function getEmergencyContacts(city: string): any {
  const base = {
    police: '100', ambulance: '108', fire: '101', 
    women_helpline: '1091', tourist_helpline: '1363', 
    disaster_mgmt: '1078', universal: '112',
    roadside_assistance: '1033'
  }
  const citySpecific: Record<string, any> = {
    chennai: { ...base, local_police: '044-28447777', hospital: 'Apollo Hospital: 044-28290200', embassy: '', tourist_office: '044-25340802' },
    mumbai: { ...base, local_police: '022-22621855', hospital: 'Lilavati Hospital: 022-26751000', embassy: '', tourist_office: '022-22074333' },
    delhi: { ...base, local_police: '011-23490100', hospital: 'AIIMS: 011-26588500', embassy: 'US Embassy: 011-24198000', tourist_office: '011-23365358' },
    jaipur: { ...base, local_police: '0141-2560063', hospital: 'SMS Hospital: 0141-2518291', tourist_office: '0141-5110598' },
    goa: { ...base, local_police: '0832-2225003', hospital: 'GMC Hospital: 0832-2458727', tourist_office: '0832-2438750' },
    bangalore: { ...base, local_police: '080-22942222', hospital: 'Manipal Hospital: 080-25024444', tourist_office: '080-22352828' },
    kolkata: { ...base, local_police: '033-22145050', hospital: 'AMRI Hospital: 033-66261000', tourist_office: '033-22485917' },
    hyderabad: { ...base, local_police: '040-27852400', hospital: 'NIMS: 040-23390631', tourist_office: '040-23262143' },
  }
  const key = city.toLowerCase().replace(/[^a-z]/g,'')
  for (const [c, data] of Object.entries(citySpecific)) { if (key.includes(c) || c.includes(key)) return data }
  return base
}

// ============================================
// SAFETY TIPS GENERATOR
// ============================================
function getSafetyTips(city: string, persona: string): string[] {
  const general = [
    'Keep copies of all documents (digital + physical)',
    'Share your itinerary with family/friends',
    'Use registered taxis/cabs only',
    'Keep emergency numbers handy (Universal: 112)',
    'Stay in well-lit areas at night',
    'Use hotel safes for valuables and extra cash',
    'Download offline maps for the destination',
    'Carry a basic first aid kit',
    'Stay hydrated and carry a water bottle',
    'Be aware of local scams and tourist traps',
  ]
  const cityTips: Record<string, string[]> = {
    delhi: ['Metro is safest public transport','Avoid auto-rickshaws without meters','Prepaid taxi counters at airport/station'],
    mumbai: ['Use local trains during non-peak hours','Carry change for local transport','Avoid lonely beaches at night'],
    jaipur: ['Negotiate prices at markets','Carry water in summer (40°C+)','Beware of "guide" scams at forts'],
    goa: ['Rent two-wheelers with proper license','Do NOT swim at unmarked beaches','Keep valuables secure on beaches'],
    varanasi: ['Wear comfortable shoes for ghats','Bargain for boat rides','Be cautious of self-appointed guides'],
  }
  const personaTips: Record<string, string[]> = {
    solo: ['Stay in hostels to meet other travelers','Share your live location with someone','Trust your instincts in unfamiliar areas'],
    family: ['Plan kid-friendly activities','Carry entertainment for children during travel','Book family rooms in advance'],
    adventure: ['Check equipment before adventure activities','Hire certified guides for treks','Carry emergency supplies'],
    luxury: ['Book premium lounge access at airports','Pre-arrange airport transfers','Verify hotel cancellation policies'],
  }
  const key = city.toLowerCase().replace(/[^a-z]/g,'')
  const extra: string[] = []
  for (const [c, tips] of Object.entries(cityTips)) { if (key.includes(c)) extra.push(...tips) }
  return [...general, ...extra, ...(personaTips[persona]||[])]
}

// ============================================
// DESTINATION RECOMMENDATIONS
// ============================================
function getRecommendations(budget: number, duration: number, preferences: string[], currentLocation: string): any[] {
  const destinations: any[] = [
    {name:'Jaipur',state:'Rajasthan',tags:['culture','history','shopping','food'],budget_range:[8000,25000],best_months:['october','november','december','january','february','march'],weather:'warm',coords:[26.91,75.79],highlights:['Amber Fort','Hawa Mahal','City Palace','Nahargarh Fort']},
    {name:'Goa',state:'Goa',tags:['beach','nightlife','food','adventure'],budget_range:[10000,40000],best_months:['november','december','january','february','march'],weather:'warm',coords:[15.30,74.12],highlights:['Baga Beach','Fort Aguada','Dudhsagar Falls']},
    {name:'Manali',state:'Himachal Pradesh',tags:['adventure','nature','spiritual'],budget_range:[8000,30000],best_months:['march','april','may','june','september','october'],weather:'cold',coords:[32.24,77.19],highlights:['Rohtang Pass','Solang Valley','Old Manali']},
    {name:'Varanasi',state:'Uttar Pradesh',tags:['spiritual','culture','history','food'],budget_range:[5000,15000],best_months:['october','november','december','january','february','march'],weather:'moderate',coords:[25.32,83.01],highlights:['Dashashwamedh Ghat','Kashi Vishwanath Temple','Sarnath']},
    {name:'Udaipur',state:'Rajasthan',tags:['culture','history','nature'],budget_range:[8000,25000],best_months:['september','october','november','december','january','february','march'],weather:'moderate',coords:[24.59,73.71],highlights:['City Palace','Lake Pichola','Jag Mandir']},
    {name:'Pondicherry',state:'Tamil Nadu',tags:['beach','culture','food','history'],budget_range:[5000,20000],best_months:['october','november','december','january','february','march'],weather:'warm',coords:[11.94,79.81],highlights:['Promenade Beach','Auroville','French Quarter']},
    {name:'Darjeeling',state:'West Bengal',tags:['nature','adventure','food'],budget_range:[7000,20000],best_months:['march','april','may','september','october','november'],weather:'cold',coords:[27.04,88.27],highlights:['Tiger Hill','Toy Train','Tea Gardens']},
    {name:'Munnar',state:'Kerala',tags:['nature','adventure'],budget_range:[6000,18000],best_months:['september','october','november','december','january','february','march'],weather:'moderate',coords:[10.09,77.06],highlights:['Tea Plantations','Eravikulam National Park','Mattupetty Dam']},
    {name:'Hampi',state:'Karnataka',tags:['history','culture','adventure'],budget_range:[4000,12000],best_months:['october','november','december','january','february'],weather:'warm',coords:[15.34,76.46],highlights:['Virupaksha Temple','Vittala Temple','Royal Enclosure']},
    {name:'Alleppey',state:'Kerala',tags:['nature','food','culture'],budget_range:[8000,25000],best_months:['august','september','october','november','december','january','february','march'],weather:'warm',coords:[9.50,76.34],highlights:['Houseboat Cruise','Alappuzha Beach','Kumarakom Bird Sanctuary']},
    {name:'Rishikesh',state:'Uttarakhand',tags:['spiritual','adventure','nature'],budget_range:[5000,15000],best_months:['february','march','april','may','september','october','november'],weather:'moderate',coords:[30.09,78.27],highlights:['Ram Jhula','Rafting','Triveni Ghat']},
    {name:'Leh Ladakh',state:'Ladakh',tags:['adventure','nature'],budget_range:[15000,50000],best_months:['june','july','august','september'],weather:'cold',coords:[34.15,77.58],highlights:['Pangong Lake','Nubra Valley','Khardung La']},
    {name:'Ooty',state:'Tamil Nadu',tags:['nature','food'],budget_range:[5000,15000],best_months:['march','april','may','october','november'],weather:'cold',coords:[11.41,76.70],highlights:['Botanical Garden','Ooty Lake','Nilgiri Mountain Railway']},
    {name:'Kodaikanal',state:'Tamil Nadu',tags:['nature','adventure'],budget_range:[5000,15000],best_months:['march','april','may','september','october'],weather:'cold',coords:[10.24,77.49],highlights:['Kodai Lake','Coakers Walk','Pillar Rocks']},
    {name:'Amritsar',state:'Punjab',tags:['spiritual','food','history','culture'],budget_range:[5000,15000],best_months:['october','november','december','january','february','march'],weather:'moderate',coords:[31.63,74.87],highlights:['Golden Temple','Wagah Border','Jallianwala Bagh']},
    {name:'Mahabalipuram',state:'Tamil Nadu',tags:['beach','history','culture'],budget_range:[3000,10000],best_months:['november','december','january','february','march'],weather:'warm',coords:[12.62,80.20],highlights:["Shore Temple","Pancha Rathas","Arjuna's Penance"]},
  ]
  
  return destinations.filter(d => {
    if (d.budget_range[0] > budget) return false
    if (preferences.length && !preferences.some(p => d.tags.includes(p))) return false
    return true
  }).map(d => ({
    ...d, estimatedCost: Math.round(d.budget_range[0] + (d.budget_range[1]-d.budget_range[0])*(duration/7)),
    matchScore: preferences.filter(p => d.tags.includes(p)).length / Math.max(preferences.length, 1) * 100,
  })).sort((a,b) => b.matchScore - a.matchScore).slice(0, 8)
}

// ============================================
// TRIP COMPARISON ENGINE
// ============================================
function compareTrips(trips: any[]): any {
  if (!trips.length) return {}
  return trips.map(t => ({
    destination: t.destination,
    days: t.days,
    totalCost: t.totalCost,
    budget: t.budget,
    budgetUtilization: Math.round(t.totalCost / t.budget * 100),
    activitiesCount: t.days_data?.reduce((s: number,d: any) => s + (d.activities?.length||0), 0) || 0,
    avgCrowd: Math.round((t.days_data?.flatMap((d: any) => d.activities||[]).reduce((s: number,a: any) => s + (a.crowd_level||50), 0) || 0) / Math.max(t.days_data?.flatMap((d: any) => d.activities||[]).length||1, 1)),
    rainyDays: (t.weather||[]).filter((w: any) => w.risk_level === 'high').length,
    weatherQuality: Math.round(((t.weather||[]).filter((w: any) => w.risk_level !== 'high').length / Math.max((t.weather||[]).length, 1)) * 100),
  }))
}

// ============================================
// API ROUTES
// ============================================

// Health check
app.get('/api/health', (c) => c.json({status:'ok',agents:7,version:'4.0',engine:'SmartRoute SRMIST Agentic AI + RL',features:['mcts','q-learning','bayesian-thompson','pomdp','naive-bayes','dense-sparse-rewards','multi-city','packing','atlas','journal','comparison','emergency-contacts','safety-tips','currency','collab']}))

// Generate Trip
app.post('/api/generate-trip', async (c) => {
  const body = await c.req.json()
  const { destination, origin='', duration=3, budget=15000, persona='solo', startDate='' } = body
  
  if (!destination) return c.json({error:'Destination required'}, 400)
  
  const [destGeo, originGeo] = await Promise.all([
    geocode(destination),
    origin ? geocode(origin) : Promise.resolve({lat:13.08,lon:80.27,name:'Chennai'})
  ])
  
  // Use the resolved city name for attraction lookup (e.g., "SRM Trichy" → "Trichy")
  const resolvedDest = destGeo.resolvedCity || destination
  
  const [attractions, weather] = await Promise.all([
    fetchAttractions(destGeo.lat, destGeo.lon, resolvedDest, duration),
    fetchWeather(destGeo.lat, destGeo.lon, duration)
  ])
  
  // Fetch photos in parallel (up to 12)
  const topPlaces = attractions.slice(0, 12)
  const photos = await Promise.all(topPlaces.map(p => fetchWikiPhoto(p.wikiTitle || p.name)))
  topPlaces.forEach((p, i) => { if (photos[i]) p.photo = photos[i] })
  
  const itinerary = buildItinerary(attractions, weather, duration, budget, destGeo.name || resolvedDest, persona, origin, originGeo)
  const langTips = getLanguageTips(resolvedDest)
  const packingList = generatePackingList(duration, weather, persona)
  const restaurants = generateRestaurants(resolvedDest, destGeo.lat, destGeo.lon)
  const emergencyContacts = getEmergencyContacts(resolvedDest)
  const safetyTips = getSafetyTips(resolvedDest, persona)
  
  return c.json({
    success: true, itinerary, languageTips: langTips, packingList, restaurants,
    photos: topPlaces.filter(p=>p.photo).map(p=>({name:p.name,url:p.photo})),
    emergencyContacts, safetyTips,
  })
})

// Multi-City Trip (from TripSage concept)
app.post('/api/generate-multi-city', async (c) => {
  const { cities, daysPerCity, budget=30000, persona='solo', origin='' } = await c.req.json()
  if (!cities?.length) return c.json({error:'At least one city required'}, 400)
  try {
    const result = await buildMultiCityTrip(cities, daysPerCity || cities.map(() => 2), budget, persona, origin)
    return c.json({success:true, ...result})
  } catch(e: any) {
    return c.json({error: e.message || 'Multi-city trip generation failed'}, 500)
  }
})

// Rate Activity — Updates Bayesian + POMDP + Q-Learning
app.post('/api/rate', async (c) => {
  const { activity, rating, category='cultural', destination='', day=1 } = await c.req.json()
  bayesianUpdate(category, rating)
  
  // POMDP observations based on rating
  if (rating >= 4) pomdpUpdate('high_rating')
  else if (rating >= 3) pomdpUpdate('mid')
  else pomdpUpdate('low_rating')
  
  // Q-Learning update with rating-based reward
  const stateKey = `${destination}|d${day}|${category}|rated`
  const action = rating >= 4 ? 'keep_plan' : rating >= 3 ? 'adjust_budget' : 'swap_activity'
  const denseR = computeDenseReward({
    rating, budgetAdherence: 0.8, weatherSafety: 0.7,
    crowdLevel: 50, timeEfficiency: 0.8, diversityBonus: 0.6
  })
  const qlResult = qUpdate(stateKey, action, denseR)
  
  return c.json({
    success: true, 
    reward: denseR,
    tdError: qlResult.tdError,
    bayesian: aiState.bayesian, 
    dirichlet: aiState.dirichlet, 
    pomdpBelief: aiState.pomdpBelief, 
    denseRewards: aiState.denseRewards.slice(-30),
    sparseRewards: aiState.sparseRewards.slice(-20),
    totalRewards: aiState.totalRewards.slice(-30),
    epsilon: aiState.epsilon,
    episode: aiState.episode,
    totalSteps: aiState.totalSteps,
    thompsonPrefs: getThompsonPreferences(),
  })
})

// Search Flights
app.post('/api/search-flights', async (c) => {
  const { origin, destination, date } = await c.req.json()
  return c.json({success:true, flights: generateFlights(origin||'Chennai', destination||'Delhi', date||'')})
})

// Search Trains
app.post('/api/search-trains', async (c) => {
  const { origin, destination } = await c.req.json()
  return c.json({success:true, trains: generateTrains(origin||'Chennai', destination||'Delhi')})
})

// Search Hotels
app.post('/api/search-hotels', async (c) => {
  const { city, days, persona } = await c.req.json()
  return c.json({success:true, hotels: generateHotels(city||'Delhi', days||3, persona||'solo')})
})

// Search Cabs
app.post('/api/search-cabs', async (c) => {
  const { city } = await c.req.json()
  return c.json({success:true, cabs: generateCabs(city||'Delhi')})
})

// Get Recommendations
app.post('/api/recommendations', async (c) => {
  const { budget=20000, duration=3, preferences=[], currentLocation='' } = await c.req.json()
  return c.json({success:true, destinations: getRecommendations(budget, duration, preferences, currentLocation)})
})

// Compare Trips
app.post('/api/compare-trips', async (c) => {
  const { trips } = await c.req.json()
  return c.json({success:true, comparison: compareTrips(trips || [])})
})

// Emergency Contacts
app.get('/api/emergency-contacts', async (c) => {
  const city = c.req.query('city') || 'delhi'
  return c.json({success:true, contacts: getEmergencyContacts(city)})
})

// Safety Tips
app.get('/api/safety-tips', async (c) => {
  const city = c.req.query('city') || 'delhi'
  const persona = c.req.query('persona') || 'solo'
  return c.json({success:true, tips: getSafetyTips(city, persona)})
})

// Emergency Replan — intelligent replanning with actual alternatives
app.post('/api/replan', async (c) => {
  const { itinerary, reason='delay', day=1, delayHours=4, weatherRisk='rain', crowdLevel='high' } = await c.req.json()
  if (!itinerary?.days_data) return c.json({error:'No itinerary to replan'}, 400)
  
  const dayData = itinerary.days_data[day-1]
  if (!dayData) return c.json({error:'Invalid day'}, 400)
  
  const replanLog: string[] = []
  
  if (reason === 'delay') {
    const trimCount = Math.ceil(delayHours / 2)
    const removed = dayData.activities.splice(-trimCount)
    replanLog.push(`Removed ${removed.length} activities due to ${delayHours}h delay: ${removed.map((a:any)=>a.name).join(', ')}`)
    // Adjust remaining timings
    let h = 9 + delayHours
    dayData.activities.forEach((a: any) => { 
      a.time = `${String(Math.floor(h)).padStart(2,'0')}:${h%1>=0.5?'30':'00'}`
      h += parseFloat(a.duration) + 0.5 
    })
    replanLog.push('Adjusted timings for remaining activities')
  } else if (reason === 'weather') {
    const outdoorTypes = ['beach','park','viewpoint','garden','nature_reserve','hiking','trekking']
    const outdoorActs = dayData.activities.filter((a:any) => outdoorTypes.some(t => (a.type||'').toLowerCase().includes(t)))
    const indoorActs = dayData.activities.filter((a:any) => !outdoorTypes.some(t => (a.type||'').toLowerCase().includes(t)))
    
    if (outdoorActs.length > 0) {
      // Replace outdoor with indoor alternatives
      const indoorAlts = [
        {name:'Local Museum Visit',type:'museum',description:'Explore local history and art in an indoor museum',cost:200,duration:'2h'},
        {name:'Cultural Workshop',type:'cultural',description:'Attend a local cooking or craft workshop',cost:500,duration:'2h'},
        {name:'Shopping District',type:'market',description:'Explore local markets and shopping areas',cost:300,duration:'1.5h'},
        {name:'Indoor Food Tour',type:'food',description:'Sample local cuisine at popular restaurants',cost:400,duration:'1.5h'},
        {name:'Temple/Heritage Visit',type:'temple',description:'Visit indoor heritage sites and temples',cost:100,duration:'1.5h'},
        {name:'Spa & Wellness',type:'relaxation',description:'Relax at a local spa or wellness center',cost:800,duration:'2h'},
      ]
      let h = 9
      const newActivities = indoorActs.map((a:any) => { 
        a.weather_warning = ''
        a.time = `${String(Math.floor(h)).padStart(2,'0')}:${h%1>=0.5?'30':'00'}`
        h += parseFloat(a.duration) + 0.5
        return a
      })
      for (let i = 0; i < outdoorActs.length && i < indoorAlts.length; i++) {
        const alt = indoorAlts[i]
        newActivities.push({
          ...outdoorActs[i], name: alt.name, type: alt.type, description: alt.description,
          cost: alt.cost, duration: alt.duration, weather_safe: true,
          weather_warning: `✅ Replanned (was: ${outdoorActs[i].name})`,
          time: `${String(Math.floor(h)).padStart(2,'0')}:${h%1>=0.5?'30':'00'}`,
          crowd_level: 30 + Math.floor(Math.random()*20),
        })
        h += parseFloat(alt.duration) + 0.5
      }
      dayData.activities = newActivities
      replanLog.push(`Replaced ${outdoorActs.length} outdoor activities with indoor alternatives`)
    } else {
      dayData.activities.forEach((a: any) => { a.weather_warning = '⚠️ Check weather before heading out' })
      replanLog.push('No outdoor activities found; added weather warnings to all')
    }
  } else if (reason === 'crowd') {
    // Reverse order: visit popular spots at off-peak times
    dayData.activities.reverse()
    let h = 7 // Start earlier to avoid crowds
    dayData.activities.forEach((a: any) => {
      a.time = `${String(Math.floor(h)).padStart(2,'0')}:${h%1>=0.5?'30':'00'}`
      a.crowd_level = Math.max(10, a.crowd_level - 25)
      h += parseFloat(a.duration) + 0.5
    })
    replanLog.push('Reordered activities to avoid peak crowd hours (starting at 7 AM)')
  }
  
  // Update day budget
  dayData.dayBudget = dayData.activities.reduce((s:number,a:any) => s + (a.cost||0), 0)
  itinerary.days_data[day-1] = dayData
  
  return c.json({success:true, itinerary, replanLog, reason, day})
})

// Nearby Places — quality search with real POIs
app.get('/api/nearby', async (c) => {
  const lat = parseFloat(c.req.query('lat')||'13.08')
  const lon = parseFloat(c.req.query('lon')||'80.27')
  const radius = parseInt(c.req.query('radius')||'3000')
  const category = c.req.query('category') || 'all'
  
  try {
    // Better Overpass query — only named places with minimum quality
    const query = `[out:json][timeout:20];(
      node(around:${radius},${lat},${lon})[tourism~"^(attraction|museum|gallery|viewpoint|zoo|theme_park|artwork|hotel|hostel)$"]["name"];
      node(around:${radius},${lat},${lon})[historic~"^(monument|memorial|castle|fort|ruins|archaeological_site|palace)$"]["name"];
      node(around:${radius},${lat},${lon})[amenity~"^(restaurant|cafe|hospital|pharmacy|bank|police)$"]["name"];
      node(around:${radius},${lat},${lon})[leisure~"^(park|garden|nature_reserve)$"]["name"];
      node(around:${radius},${lat},${lon})[shop~"^(mall|supermarket|department_store)$"]["name"];
    );out 40;`
    const r = await fetch('https://overpass-api.de/api/interpreter', {method:'POST', body:`data=${encodeURIComponent(query)}`, headers:{'Content-Type':'application/x-www-form-urlencoded','User-Agent':'SmartRouteSRMIST/4.0'}})
    const d: any = await r.json()
    
    // Filter and sort by relevance
    const places = (d.elements||[])
      .filter((e:any) => e.tags?.name && e.tags.name.length > 3)
      .map((e:any) => {
        const tags = e.tags || {}
        const ptype = tags.tourism || tags.historic || tags.amenity || tags.leisure || tags.shop || 'place'
        // Calculate distance for sorting
        const dLat = (e.lat - lat) * 111320
        const dLon = (e.lon - lon) * 111320 * Math.cos(lat * Math.PI/180)
        const dist = Math.round(Math.sqrt(dLat*dLat + dLon*dLon))
        return {
          name: tags['name:en'] || tags.name,
          lat: e.lat, lon: e.lon,
          type: ptype,
          description: tags.description || tags['description:en'] || `${ptype.replace(/_/g,' ')} nearby`,
          phone: tags.phone || tags['contact:phone'] || '',
          website: tags.website || tags['contact:website'] || '',
          opening_hours: tags.opening_hours || '',
          rating: tags.stars ? parseFloat(tags.stars) : (3.5 + Math.random()*1.5),
          distance: dist,
          address: tags['addr:street'] ? `${tags['addr:street']}${tags['addr:housenumber']?', '+tags['addr:housenumber']:''}` : '',
        }
      })
      .sort((a:any,b:any) => a.distance - b.distance)
      .slice(0, 25)
    
    return c.json({success:true, places, count: places.length})
  } catch(e) {
    // Fallback: try OpenTripMap
    try {
      const r2 = await fetch(`https://api.opentripmap.com/0.1/en/places/radius?radius=${radius}&lon=${lon}&lat=${lat}&kinds=interesting_places,cultural,historic,natural,foods&format=json&limit=20&rate=2&apikey=5ae2e3f221c38a28845f05b6aec53ea2b07e9e48b7f89b38bd76ca73`)
      const otm: any = await r2.json()
      const places = (otm||[]).filter((p:any) => p.name && p.name.length > 3).map((p:any) => ({
        name: p.name, lat: p.point?.lat||lat, lon: p.point?.lon||lon,
        type: p.kinds?.split(',')[0]?.replace(/_/g,' ') || 'place',
        description: `${p.kinds?.split(',')[0]?.replace(/_/g,' ')||'attraction'} nearby`,
        rating: p.rate || 3.5, distance: Math.round(p.dist || 0),
      }))
      return c.json({success:true, places, count: places.length})
    } catch(e2) {}
    return c.json({success:true, places:[], count:0})
  }
})

// AI State — Full RL state
app.get('/api/ai-state', (c) => c.json({
  bayesian: aiState.bayesian, dirichlet: aiState.dirichlet,
  pomdpBelief: aiState.pomdpBelief, 
  denseRewards: aiState.denseRewards.slice(-30),
  sparseRewards: aiState.sparseRewards.slice(-20),
  totalRewards: aiState.totalRewards.slice(-30),
  cumulativeReward: aiState.cumulativeReward,
  qTableSize: Object.keys(aiState.qTable).length, 
  epsilon: aiState.epsilon,
  episode: aiState.episode,
  totalSteps: aiState.totalSteps,
  alpha: aiState.alpha,
  gamma: aiState.gamma,
  thompsonPrefs: getThompsonPreferences(),
  agentDecisions: aiState.agentDecisions.slice(-10),
}))

// Chatbot — Smart context-aware AI assistant
app.post('/api/chat', async (c) => {
  const { message, context } = await c.req.json()
  if (!message?.trim()) return c.json({success:false, response:'Please type a message!'})
  
  const lower = message.toLowerCase().trim()
  const dest = context?.destination || ''
  const origin = context?.origin || ''
  const budgetCtx = context?.budget || 15000
  let response = ''
  
  // 1. Trip planning intent — detailed response with actionable steps
  if (/plan\s+(a\s+)?trip\s+to|travel\s+to|visit\s+to|going\s+to|trip\s+for|want\s+to\s+go|take\s+me\s+to|itinerary\s+for/i.test(lower)) {
    const match = lower.match(/(?:to|for|visit|go)\s+([a-z\s]+?)(?:\s+for|\s+in|\s+with|\s*$)/i)
    const place = match?.[1]?.trim() || ''
    if (place) {
      const cap = place.charAt(0).toUpperCase() + place.slice(1)
      response = `🗺️ **Planning your trip to ${cap}!**\n\nHere's what to do:\n1. Enter **"${place}"** in the Destination field\n2. Set your budget and number of days\n3. Click **Generate AI Trip**\n\nMy 7 AI agents will create a personalized itinerary with:\n- 🏛️ Top attractions ranked by your preferences\n- 🌦️ Weather-adjusted scheduling\n- 💰 Budget-optimized activities\n- 🔗 Booking links for flights, trains & hotels\n\n**Pro tip:** After generating, use the ⚡ Smart Automations panel to optimize your route, balance budget, and avoid crowds automatically!`
    } else {
      response = `🗺️ I'd love to help you plan! Where do you want to go?\n\nTry:\n- "Plan a trip to Jaipur"\n- "Plan a trip to Manali for 5 days"\n- "Plan a trip to SRM Trichy campus"\n\nOr click **Help Me Choose** for AI destination recommendations!`
    }
  }
  // 2. Greetings — concise, helpful
  else if (/^(hello|hi|hey|namaste|howdy|sup)\b|good\s+(morning|afternoon|evening)/i.test(lower)) {
    response = `👋 **Hey there!** How can I help you today?\n\nQuick options:\n- 🗺️ "Plan a trip to [city]"\n- ✈️ "Search flights to Delhi"\n- 🌦️ "Weather in ${dest || 'Goa'}"\n- 💰 "Budget tips"\n- 🍽️ "Food recommendations"\n- 🛡️ "Safety tips"\n\nJust ask away! 😊`
  }
  // 3. Specific question about current trip
  else if (dest && /what|how|tell|show|give|suggest|recommend/i.test(lower) && /my\s+trip|itinerary|plan|schedule/i.test(lower)) {
    if (state_summary()) {
      response = state_summary()
    } else {
      response = `📋 Your current trip to **${dest}** is active. Here are things I can help with:\n\n- "Optimize my route" — reduce travel time\n- "Balance my budget" — even spending across days\n- "Avoid crowds" — reorder for low-crowd times\n- "Add food stops" — insert meal breaks\n- "Emergency replan" — handle weather/delays\n\nUse the **Smart Automations** panel on the right for one-click optimizations!`
    }
  }
  // 4. Weather queries
  else if (/weather|forecast|rain|temperature|hot|cold|humid/i.test(lower)) {
    const target = dest || extractCity(lower) || 'your destination'
    response = `🌦️ **Weather for ${target}:**\n\nI use the **OpenMeteo API** with **Naive Bayes classification** to analyze:\n- 🌡️ Temperature range (min/max)\n- 💧 Precipitation probability\n- 💨 Wind speed\n- ☀️ UV index\n\n**Risk Levels:** 🟢 Low · 🟡 Medium · 🔴 High\n\n${dest ? 'Check your itinerary — each day shows weather forecasts with risk indicators.' : 'Generate a trip to see day-by-day weather analysis!'}\n\n💡 If bad weather is detected, use **Weather Swap** in Smart Automations to auto-replace outdoor activities!`
  }
  // 5. Budget
  else if (/budget|cheap|save|money|cost|expensive|afford|price/i.test(lower)) {
    response = `💰 **Budget Tips${dest ? ' for '+dest : ''}:**\n\n**Money-Saving Strategies:**\n🚂 Book trains 30+ days ahead on IRCTC\n🏨 OYO/Treebo for ₹600-1500/night stays\n🍽️ Local dhabas & street food (₹50-150/meal)\n🚌 Use public transport & shared cabs\n🆓 Free: temples, parks, beaches, ghats\n\n**Smart Budget Split:**\n🏨 30% Stay | 🍽️ 20% Food | 🎯 25% Activities | 🚗 15% Transport | 🆘 10% Emergency\n\n${dest ? `Your ₹${budgetCtx.toLocaleString()} budget is being optimized by the AI Budget Agent using MDP reward functions.` : 'Generate a trip to see AI-optimized budget allocation!'}\n\n💡 Click **Balance Budget** in Smart Automations to auto-optimize spending!`
  }
  // 6. Food
  else if (/food|restaurant|eat|cuisine|dine|hungry|lunch|dinner|breakfast|snack/i.test(lower)) {
    response = `🍽️ **Food Guide${dest ? ' for '+dest : ''}:**\n\n**Recommendations:**\n🥘 Try local specialties & regional dishes\n🛕 Temple food (prasadam) — free & authentic\n🍜 Street food at busy stalls — follow the locals\n☕ Regional drinks: filter coffee, lassi, chai\n\n**Booking:**\n- [Zomato](https://zomato.com) — reviews + delivery\n- [Swiggy](https://swiggy.com) — quick delivery\n- [Dineout](https://dineout.co.in) — table reservations\n\n💡 Use **Add Food Stops** in Smart Automations to auto-insert meal breaks into your itinerary!`
  }
  // 7. Safety
  else if (/safe|danger|security|emergency|help|police|hospital/i.test(lower)) {
    response = `🛡️ **Emergency Numbers (India):**\n\n🚨 **112** — Universal Emergency\n🚔 **100** — Police\n🚑 **108** — Ambulance\n🚒 **101** — Fire\n👩 **1091** — Women Helpline\n🏛️ **1363** — Tourist Helpline\n\n**Safety Tips:**\n• Share itinerary with family\n• Use only registered taxis (Ola/Uber)\n• Download offline maps\n• Keep document copies\n• Stay in well-lit areas at night\n\n${dest ? `Check the **Emergency Contacts** panel in the sidebar for ${dest}-specific numbers.` : ''}`
  }
  // 8. Hidden gems
  else if (/hidden|gem|secret|offbeat|unexplored|unique|unusual/i.test(lower)) {
    response = `💎 **Hidden Gems${dest ? ' near '+dest : ''}:**\n\nOur AI discovers gems using:\n1. **Overpass API** — finds lesser-known spots\n2. **Crowd Analysis** — places <40% crowd\n3. **MCTS** — 50 iterations for unique combos\n\nLook for 💎 tagged activities with low crowd levels in your itinerary.\n\n**Check the Hidden Gems tab** in the Discovery section after generating your trip!\n\n💡 Rate activities ⭐⭐⭐⭐⭐ to train the AI — it'll find similar gems in future trips!`
  }
  // 9. Flights
  else if (/flight|fly|plane|airport|airline/i.test(lower)) {
    const fromCity = origin || extractCity(lower.replace(/flight|fly|plane|from|to/gi, '')) || 'your city'
    const toCity = dest || extractCityAfter(lower, 'to') || 'your destination'
    response = `✈️ **Flight Search: ${fromCity} → ${toCity}**\n\n**Airlines:** IndiGo, Air India, Vistara, SpiceJet, AirAsia, Akasa Air\n\n**Platforms with pre-filled details:**\n🔗 Google Flights — price comparison\n🔗 MakeMyTrip — bundled deals\n🔗 Skyscanner — global search\n🔗 ixigo — budget focus\n\n**Tips:**\n✅ Book 2-4 weeks ahead\n✅ Tue/Wed flights cheapest\n✅ Use incognito mode\n\n${dest ? 'Click **✈️ Flights** in the Booking Wizard below your itinerary!' : 'Generate a trip first, then use the Booking Wizard!'}`
  }
  // 10. Trains
  else if (/train|railway|irctc|rail/i.test(lower)) {
    response = `🚂 **Train Booking Guide:**\n\n**Types:** Vande Bharat (fastest) · Rajdhani · Shatabdi · Duronto · Superfast\n\n**Book on:** IRCTC (official) · ConfirmTkt · RailYatri · ixigo Trains\n\n**Tips:**\n✅ Book 120 days in advance\n✅ Tatkal: 10 AM (AC) / 11 AM (Non-AC)\n✅ Use "Alternate Trains" feature\n\n${dest ? 'Click **🚂 Trains** in the Booking Wizard!' : 'Generate your trip first!'}`
  }
  // 11. Hotels
  else if (/hotel|stay|accommodation|hostel|resort|lodge|oyo|airbnb/i.test(lower)) {
    response = `🏨 **Accommodation${dest ? ' in '+dest : ''}:**\n\n**Budget (₹500-1500):** OYO, Treebo, Zostel\n**Mid-Range (₹1500-5000):** Lemon Tree, Radisson\n**Luxury (₹5000+):** Taj, ITC, Oberoi\n\n**Platforms:** Booking.com · MakeMyTrip · Goibibo · Agoda · Trivago · OYO\n\n${dest ? 'Click **🏨 Hotels** in the Booking Wizard!' : 'Generate your trip first!'}`
  }
  // 12. AI explanation
  else if (/how.*work|algorithm|ai|machine\s*learning|reinforcement|q.?learn|explain/i.test(lower)) {
    response = `🧠 **How SmartRoute AI Works:**\n\n**7 Agents:**\n1. 🗺️ Planner — MCTS (50 iterations)\n2. 🌦️ Weather — Naive Bayes\n3. 👥 Crowd — Time-based prediction\n4. 💰 Budget — MDP optimization\n5. ❤️ Preference — Bayesian Beta sampling\n6. 🎫 Booking — Multi-platform search\n7. 🧠 Explainer — POMDP belief state\n\n**RL:** Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]\n**Dense Reward:** rating + budget + weather − crowd + time + diversity\n**Exploration:** ε-greedy + Thompson Sampling\n\nRate activities ⭐ to train the AI in real-time!`
  }
  // 13. Packing
  else if (/pack|luggage|carry|bring|clothes|bag|suitcase/i.test(lower)) {
    response = `🧳 **Smart Packing:**\n\nYour AI packing list adapts to:\n📅 Trip duration\n🌦️ Weather forecast\n👤 Travel persona\n\nGo to the **Packing** tab to see your personalized checklist!\n\nUse **Pre-Trip Checklist** in Smart Automations for a complete preparation guide.`
  }
  // 14. Multi-city
  else if (/multi.?city|multiple\s+cities|road\s*trip|circuit/i.test(lower)) {
    response = `🗺️ **Multi-City Trips:**\n\nClick **Multi-City Trip** in the planning panel!\n\n**Popular Routes:**\n🏰 Golden Triangle: Delhi → Agra → Jaipur\n🏖️ South India: Chennai → Pondicherry → Madurai → Kochi\n⛰️ Himalayan: Delhi → Shimla → Manali\n\nAI optimizes transit between cities using nearest-neighbor TSP!`
  }
  // 15. Nearby
  else if (/nearby|around\s*me|close\s*to|near\s*here/i.test(lower)) {
    response = `📍 **Nearby Places:**\n\nClick the **Nearby** button in the header!\n\nSearches for: attractions, restaurants, hospitals, parks, markets within 3-5km radius using the Overpass API.\n\n${dest ? `Or use the itinerary coordinates to discover hidden spots around ${dest}.` : 'Allow location access or generate a trip first!'}`
  }
  // 16. Campus
  else if (/campus|srm|university|college|institute/i.test(lower)) {
    response = `🏫 **Campus Trip Planning:**\n\n**Supported Campuses:**\n- SRM Kattankulathur (Chennai)\n- SRM Trichy\n- SRM NCR (Greater Noida)\n- SRM Andhra (Amaravati)\n- SRM Sikkim (Gangtok)\n- IIT Madras, IIT Delhi, IIT Bombay\n- VIT Vellore, BITS Pilani, NIT Trichy\n\nJust type the campus name as destination! e.g., "SRM Trichy campus"`
  }
  // 17. Compare
  else if (/compare|versus|vs|which.*better/i.test(lower)) {
    response = `📊 **Trip Comparison:**\n\nGenerate multiple trips, then click **Compare Trips** to see side-by-side metrics: cost, activities, weather, crowd levels, and budget utilization.\n\nAll generated trips are auto-saved for comparison.`
  }
  // 18. Thanks
  else if (/thank|thanks|thx|great|awesome|nice|good|cool|perfect|amazing/i.test(lower)) {
    response = `😊 Glad I could help! Remember to:\n• Rate ⭐ activities to improve AI\n• Use Smart Automations for optimization\n• Check Packing tab before your trip\n\nHave an amazing journey! 🌟`
  }
  // 19. What can you do / help
  else if (/what.*can|what.*do|help me|capabilities|features/i.test(lower)) {
    response = `🤖 **I can help you with:**\n\n🗺️ Trip planning — "Plan a trip to Jaipur"\n✈️ Flight search — "Flights to Delhi"\n🚂 Train booking — "Trains to Mumbai"\n🏨 Hotels — "Hotels in Goa"\n🌦️ Weather — "Weather in Manali"\n💰 Budget — "Budget tips"\n🍽️ Food — "Best food in Chennai"\n🛡️ Safety — "Safety tips"\n💎 Hidden gems — "Secret spots"\n🏫 Campus trips — "SRM Trichy campus"\n🧳 Packing — "What to pack"\n📊 Compare — "Compare my trips"\n🗺️ Multi-city — "Multi-city route"\n\nJust ask! 😊`
  }
  // Default — helpful, not a long intro
  else {
    response = `I'm not sure I understood that. Try asking:\n\n🗺️ "Plan a trip to [city]"\n✈️ "Flights to [city]"\n🌦️ "Weather in [city]"\n💰 "Budget tips"\n🍽️ "Food recommendations"\n🛡️ "Safety tips"\n\nOr use the suggestion buttons below! 👇`
  }
  
  return c.json({success:true, response})
})

// Helper functions for chatbot
function extractCity(text: string): string {
  const cities = Object.keys(CITY_COORDS)
  for (const city of cities) {
    if (text.toLowerCase().includes(city)) return city.charAt(0).toUpperCase() + city.slice(1)
  }
  return ''
}

function extractCityAfter(text: string, keyword: string): string {
  const idx = text.toLowerCase().indexOf(keyword)
  if (idx < 0) return ''
  const after = text.substring(idx + keyword.length).trim()
  return extractCity(after) || after.split(/\s+/)[0] || ''
}

function state_summary(): string {
  return ''
}

// ============================================
// SERVE FRONTEND
// ============================================
app.get('/', (c) => {
  return c.redirect('/static/index.html')
})

export default app

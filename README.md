# 🚀 SmartRoute SRMIST — Intelligent Adaptive Route Optimization
> **Created by: devcrazy AKA Abhay Goyal**

![Campus Navigation](https://img.shields.io/badge/Navigation-AI_Powered-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Production_Ready-success)

Welcome to **SmartRoute SRMIST**, an advanced campus navigation and route optimization system tailored specifically for the SRM Institute of Science and Technology (SRMIST KTR) campus. This is a highly scalable, AI-powered platform designed to provide the absolute best routing experience for a massive user base!

---

## 🌟 The Vision

> *"To create the most intelligent, adaptive, and visually stunning navigation experience for the SRMIST community, combining cutting-edge AI with an intuitive dashboard interface."* — **devcrazy AKA Abhay Goyal**

---

## 🔥 Key Features

### 🧠 Core Artificial Intelligence
- **Q-Learning Reinforcement Learning:** The system learns from your feedback! Using the full Bellman equation, the AI gets smarter over time.
- **A\* Pathfinding Algorithm:** Optimized with the Haversine (great-circle) distance heuristic using real GPS coordinates.
- **4-Factor Cost Function:** A sophisticated decision engine weighting:
  - 📏 Distance
  - ⏱️ Time estimating
  - 🚦 Dynamic Traffic
  - 🤖 RL Preferences

### 🎨 Stunning Interface & User Experience
- **Interactive Multi-Tab Dashboard:** Seamless navigation between Planner, SRMIST Campus Info, and Neural Core Dashboard.
- **Dynamic Theming:** Beautiful Pastel-Blue Light Mode and Cyberpunk Dark Mode.
- **Live Traffic Simulation:** True-to-life peak hour traffic multipliers (Morning Rush, Lunch Hour, Evening Rush).
- **Explainable AI (XAI):** See exactly *why* the AI chose your route with a per-segment breakdown.
- **⚡ Emergency Replan & 🧠 Quick FAB:** Innovative shortcuts for instant route correction.

### 🏛️ SRMIST KTR Localization
We've meticulously mapped the Kattankulathur campus with **20 precise nodes** and **39 connected paths**, featuring real GPS coordinates for exact accuracy:
- **Academic Zones:** University Building, Tech Park, Architecture Block, Main Block, Mechanical Block, Biotech Dept.
- **Landmarks:** SRM Main Gate, Central Library, T.P. Ganesan Auditorium, Java Green, Clock Tower.
- **Hostels & Facilities:** SRM Hospital, Potheri Railway Station, Nelson Mandela, NRI Hostel, Women's Hostel, Abode Valley, Estancia.

---

## 🛠️ Architecture Overview

The system is built with a decoupled frontend/backend architecture, making it ready for future mobile app expansion!

```text
Smart_Route-main/
├── app/
│   └── api.py              # FastAPI server & REST endpoints
├── core/
│   ├── graph.py            # Graph logic & Haversine distance
│   ├── rl_agent.py         # Q-Learning AI module
│   └── routing.py          # A* pathfinding and 4-factor cost engine
├── data/
│   ├── chennai_map.json    # The 20-node, 39-edge mapped campus
│   └── q_table.json        # Persistent AI memory
├── public/                 # Vanilla Web Engine (Mobile-Ready)
│   ├── index.html          
│   ├── style.css           
│   └── script.js           
├── utils/
│   └── logger.py           
└── README.md
```

---

## ⚙️ How The Intelligence Works

### 1. Multi-Factor Decision Making
Routes aren't just about distance. The AI considers a combined cost:
`Cost = (Distance × w1) + (Time × w2) + (Traffic × w3) + (RL_Bias × w4)`

You can choose from **4 AI Strategies**:
1. 🏎️ **Fastest:** Prioritizes time (55% weight).
2. 📏 **Shortest:** Prioritizes pure geographical distance (60% weight).
3. 🍃 **Low Traffic:** Prioritizes roads with the least congestion (60% weight).
4. ⚖️ **Balanced:** An equal split of all factors.

### 2. Reinforcement Learning
Using Q-Learning, the agent tracks state (current node) and action (next node). By providing positive or negative feedback, the AI updates its memory using the Bellman Update Equation, persistently learning the best paths around campus.

---

## 🚀 Getting Started

Want to run the SmartRoute engine locally?

### Prerequisites
- Python 3.8+
- PIP package manager

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic
```

### 2. Start the AI Server
```bash
# Run from the root directory
python -m uvicorn app.api:app --reload --port 8000
```
*The server will start at `http://127.0.0.1:8000/`. Open this in your browser!*

### 3. Verify System Health
Run the comprehensive 32-suite route verification test:
```bash
python test_routes.py
```

---

## 📱 Future Roadmap
- Integration with live traffic APIs.
- Epsilon-greedy implementation for the RL core.
- **Full Mobile Phone Application Expansion.**

---

## 📜 License & Authorship

**© 2026 devcrazy AKA Abhay Goyal**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Designed and architected by devcrazy AKA Abhay Goyal. All rights reserved.

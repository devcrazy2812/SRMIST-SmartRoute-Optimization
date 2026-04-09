# Intelligent Adaptive Route Optimization System (SRMIST Edition)

An advanced, research-grade route optimization engine incorporating Reinforcement Learning (Q-Learning) and multi-factor decision-making. Designed explicitly for navigating the SRM Institute of Science and Technology (SRMIST) Kattankulathur campus.

## Problem Statement

Traditional maps rely strictly on shortest-path algorithms (`cost = distance`), ignoring critical real-world factors like user preferences, current traffic, and abstract time penalties. 
Furthermore, static maps do not *learn*. This project solves these limitations by upgrading basic routing into a dynamic, learning agent. 

## Key Features

1. **Multi-Factor Cost Function:** Replaces basic distance with dynamically weighted values: `cost = (distance * w1) + (time * w2) + (traffic * w3) + (preference_bias)`.
2. **Reinforcement Learning (Q-Learning):**
   - **State:** Current node in the graph.
   - **Action:** Edge selected (next node).
   - **Reward:** User feedback on the provided route (+1 for accepted, -1 for rejected).
3. **Explainable Output:** Instead of simply presenting a route, the system tells you *why* (e.g., "avoided traffic", "learned from previous feedback").
4. **SRMIST Localization:** Features a customized node map including real campus locations like *SRM Main Gate, Tech Park, UB, Potheri Railway Station, and Estancia*.
5. **Interactive Feedback Loop:** Run the simulator to train the Q-table in real-time.

## Project Architecture

The codebase has been completely refactored into a clean Python intelligence engine:

```text
/
├── app/
│   └── main.py              # CLI Simulation Loop & User Interaction
├── core/
│   ├── graph.py             # Parses campus map and computes multi-factor logic
│   ├── routing.py           # Modified Dijkstra search biased by the Q-table
│   └── rl_agent.py          # State/Action/Reward Q-Learning engine
├── data/
│   ├── chennai_map.json     # SRMIST KTR Nodes, Edges, Traffic, Time
│   └── q_table.json         # Persistent RL brain
└── utils/
    └── logger.py            # Debugging and Step Logs
```

## How the Reinforcement Learning Works

The RL agent uses Q-Learning to augment the raw routing cost:
- Standard paths have an abstract baseline cost.
- As you use the system and provide `y/n` feedback, the `q_table.json` updates using the Bellman TD equation.
- `effective_cost = base_cost - (RL_weight * q_value)`. 
- Positive reinforcement reduces the apparent mathematical cost of those edges in future A*/Dijkstra calculations.
- Over time, the shortest mathematical path might be abandoned if users continually reject it due to unaccounted real-world properties (e.g., heavily flooded road, bad scenery).

## Running the Project

### Prerequisites
- Python 3.8+

### Quick Start
```bash
python app/main.py
```

1. Enter your Start and End nodes (e.g., `Potheri_Station` to `Tech_Park`).
2. Provide your preference (`fastest`, `shortest`, `low_traffic`, `balanced`).
3. Read the AI's transparent reasoning.
4. Provide `y/n` feedback at the prompt.
5. (Optional) Rerun the same route query after providing negative feedback to watch the A* algorithm pivot and select a different path based on Q-values!

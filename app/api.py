"""
SmartRoute SRMIST API
Created by: devcrazy AKA Abhay Goyal
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import RoutingGraph
from core.rl_agent import RLAgent
from core.routing import Router

# ── Application ────────────────────────────────────────
app = FastAPI(
    title="SmartRoute SRMIST",
    description="Intelligent Adaptive Route Optimization with Reinforcement Learning",
    version="2.0.0",
)

graph = RoutingGraph("data/chennai_map.json")
rl_agent = RLAgent("data/q_table.json")
router = Router(graph, rl_agent)


# ── Models ─────────────────────────────────────────────
class RouteRequest(BaseModel):
    startNode: str
    endNode: str
    preference: str


class FeedbackRequest(BaseModel):
    path: list
    reward: int


# ── Endpoints ──────────────────────────────────────────
@app.get("/api/nodes")
def get_nodes():
    """All nodes, preferences, and raw map data for the frontend."""
    return {
        "nodes": graph.nodes,
        "preferences": list(graph.preferences.keys()),
        "map_data": graph.get_raw_data(),
    }


@app.post("/api/route")
def calculate_route(req: RouteRequest):
    """Compute optimal route via A* with 4-factor cost + RL bias."""
    try:
        result, err = router.find_route(req.startNode, req.endNode, req.preference)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if err:
        raise HTTPException(status_code=400, detail=err)
    return result


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Apply RL reward (+1/-1) along the route using full Bellman update."""
    if req.reward not in (1, -1):
        raise HTTPException(status_code=400, detail="Reward must be +1 or -1.")
    rl_agent.apply_feedback_to_route(req.path, req.reward)
    return {"status": "ok", "message": "Q-Table updated via Bellman equation."}


@app.get("/api/qtable")
def get_qtable():
    """Live Q-Table snapshot for analytics panel."""
    return rl_agent.q_table


@app.get("/api/stats")
def get_stats():
    """RL agent learning statistics."""
    return rl_agent.get_stats()


# ── Static Files ───────────────────────────────────────
app.mount("/", StaticFiles(directory="public", html=True), name="static")

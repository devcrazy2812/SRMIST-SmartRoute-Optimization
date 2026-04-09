from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import os
import json

# Ensure core modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import RoutingGraph
from core.rl_agent import RLAgent
from core.routing import Router

app = FastAPI(title="SmartRoute SRMIST Intelligent System")

# Initialize AI Modules
graph = RoutingGraph("data/chennai_map.json")
rl_agent = RLAgent("data/q_table.json")
router = Router(graph, rl_agent, rl_weight=5.0)

# Models
class RouteRequest(BaseModel):
    startNode: str
    endNode: str
    preference: str

class FeedbackRequest(BaseModel):
    path: list
    reward: int

@app.get("/api/nodes")
def get_nodes():
    """Returns available locations and full metadata for the frontend."""
    return {
        "nodes": graph.nodes,
        "nodes_data": graph.nodes, # For mapping compatibility if needed
        "preferences": list(graph.preferences.keys()),
        # Pass coordinates directly from raw source
        "map_data": json.load(open("data/chennai_map.json", "r"))
    }

@app.post("/api/route")
def calculate_route(req: RouteRequest):
    """Calculates the best route utilizing multi-factor costs and Q-Learning biases."""
    result, err = router.find_route(req.startNode, req.endNode, req.preference)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return result

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Accepts +1 / -1 reward and applies the TD Bellman update to all transitions in path."""
    if req.reward not in [1, -1]:
        raise HTTPException(status_code=400, detail="Reward must be 1 or -1")
    
    rl_agent.apply_feedback_to_route(req.path, req.reward)
    return {"status": "success", "message": "RL Q-Table Updated Successfully!"}

@app.get("/api/qtable")
def get_qtable():
    """Sends current state of QTable for visualization."""
    return rl_agent.q_table

# Mount the static frontend directly onto the FastAPI server
# This will serve index.html when visiting the root automatically if named properly
app.mount("/", StaticFiles(directory="public", html=True), name="public")

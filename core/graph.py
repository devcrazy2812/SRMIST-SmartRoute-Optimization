import json
import os
import datetime
from collections import defaultdict
from utils.logger import logger

class RoutingGraph:
    def __init__(self, map_file_path="data/chennai_map.json"):
        self.nodes = {}
        self.edges = defaultdict(list)
        self.load_graph(map_file_path)
        
        # Default weights if not specified map to w1 (distance), w2 (time), w3 (traffic)
        # We will expose a way to scale user preference in routing or RL.
        self.preferences = {
            "fastest": {"w_dist": 0.2, "w_time": 0.6, "w_traffic": 0.2},
            "shortest": {"w_dist": 0.7, "w_time": 0.2, "w_traffic": 0.1},
            "low_traffic": {"w_dist": 0.1, "w_time": 0.2, "w_traffic": 0.7},
            "balanced": {"w_dist": 0.33, "w_time": 0.33, "w_traffic": 0.34}
        }

    def load_graph(self, path):
        if not os.path.exists(path):
            logger.error(f"Map file not found: {path}")
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        for node in data.get("nodes", []):
            self.nodes[node["id"]] = node["name"]
            
        for edge in data.get("edges", []):
            u = edge["from"]
            v = edge["to"]
            dist = edge.get("distance", 1.0)
            time = edge.get("time", 1.0)
            traffic = edge.get("traffic", 0.0)
            
            # Assuming undirected graph for campus routes
            self.edges[u].append({"node": v, "distance": dist, "time": time, "traffic": traffic})
            self.edges[v].append({"node": u, "distance": dist, "time": time, "traffic": traffic})
            
        logger.info(f"Loaded graph with {len(self.nodes)} nodes and {sum(len(v) for v in self.edges.values())//2} edges.")

    def calculate_cost(self, u, v, preference="balanced"):
        """
        Calculates the multi-factor cost to travel between adjacent nodes u and v.
        cost = (distance * w1) + (time * w2) + (traffic * w3)
        User preference acts to bias these weights. We handle RL user preference separately 
        in the routing algorithm using Q-values.
        """
        weights = self.preferences.get(preference, self.preferences["balanced"])
        w_dist = weights["w_dist"]
        w_time = weights["w_time"]
        w_traf = weights["w_traffic"]
        
        edge_data = None
        for edge in self.edges.get(u, []):
            if edge["node"] == v:
                edge_data = edge
                break
                
        if not edge_data:
            return float('inf') # No direct path
            
        # Dynamic Time-based Traffic Simulator
        # Peak hours (e.g., 8-10 AM or 4-6 PM) spike traffic significantly
        current_hour = datetime.datetime.now().hour
        traffic_multiplier = 1.0
        is_rush_hour = False
        
        # 8 AM to 10 AM, and 4 PM to 6 PM are rush hours on campus
        if (8 <= current_hour <= 10) or (16 <= current_hour <= 18):
            traffic_multiplier = 2.5
            is_rush_hour = True

        # Normalize somewhat: distances are ~0.1-2.0, time is ~2-15, traffic is 0.0-1.0
        norm_dist = edge_data["distance"] * 10
        norm_time = edge_data["time"]
        norm_traf = min(1.0, edge_data["traffic"] * traffic_multiplier) * 10 
        
        base_cost = (norm_dist * w_dist) + (norm_time * w_time) + (norm_traf * w_traf)
        
        # Attach rush hour status to edge data so the router can explain it
        edge_data["is_rush_hour"] = is_rush_hour
        
        return base_cost, edge_data

    def get_neighbors(self, u):
        return [edge["node"] for edge in self.edges.get(u, [])]

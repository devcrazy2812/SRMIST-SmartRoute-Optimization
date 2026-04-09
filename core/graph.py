"""
SmartRoute SRMIST - Graph module
Created by: devcrazy AKA Abhay Goyal
"""
import json
import os
import datetime
from collections import defaultdict
from utils.logger import logger


class RoutingGraph:
    """
    Campus navigation graph with 4-factor cost calculation:
        cost = (distance * w1) + (time * w2) + (traffic * w3) + (rl_preference * w4)

    Includes dynamic time-based traffic simulation for peak campus hours.
    """

    def __init__(self, map_file_path="data/chennai_map.json"):
        self.nodes = {}        # id -> display name
        self.node_data = {}    # id -> full dict with lat/lon
        self.edges = defaultdict(list)
        self.map_file_path = map_file_path
        self.load_graph(map_file_path)

        # 4-factor weight profiles: w_dist, w_time, w_traffic, w_rl
        self.preferences = {
            "fastest":     {"w_dist": 0.15, "w_time": 0.55, "w_traffic": 0.15, "w_rl": 0.15},
            "shortest":    {"w_dist": 0.60, "w_time": 0.15, "w_traffic": 0.10, "w_rl": 0.15},
            "low_traffic": {"w_dist": 0.10, "w_time": 0.15, "w_traffic": 0.60, "w_rl": 0.15},
            "balanced":    {"w_dist": 0.25, "w_time": 0.25, "w_traffic": 0.25, "w_rl": 0.25},
        }

    def load_graph(self, path):
        if not os.path.exists(path):
            logger.error(f"Map file not found: {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            self.nodes[node["id"]] = node["name"]
            self.node_data[node["id"]] = node

        for edge in data.get("edges", []):
            u, v = edge["from"], edge["to"]
            payload = {
                "node": v,
                "distance": edge.get("distance", 1.0),
                "time": edge.get("time", 1.0),
                "traffic": edge.get("traffic", 0.0),
            }
            self.edges[u].append(payload)
            self.edges[v].append(dict(payload, node=u))

        logger.info(
            f"Graph loaded: {len(self.nodes)} nodes, "
            f"{sum(len(v) for v in self.edges.values()) // 2} edges."
        )

    def calculate_cost(self, u, v, preference="balanced", q_value=0.0):
        """
        4-factor cost function:
          cost = (dist_norm * w1) + (time_norm * w2) + (traffic_norm * w3) + (rl_norm * w4)

        q_value is passed in by the router so the cost function itself
        incorporates RL preference as the 4th explicit factor.

        Returns (cost, edge_info_copy) or (inf, None).
        """
        weights = self.preferences.get(preference, self.preferences["balanced"])

        edge_data = None
        for e in self.edges.get(u, []):
            if e["node"] == v:
                edge_data = e
                break

        if edge_data is None:
            return float("inf"), None

        # --- Dynamic time-of-day traffic ---
        hour = datetime.datetime.now().hour
        traffic_mult = 1.0
        is_rush = False

        # Peak campus hours: morning classes & evening rush
        if 8 <= hour <= 10:
            traffic_mult = 2.5
            is_rush = True
        elif 16 <= hour <= 18:
            traffic_mult = 2.0
            is_rush = True
        elif 12 <= hour <= 13:   # lunch hour moderate spike
            traffic_mult = 1.5

        # Normalize all factors to roughly 0-10 range
        norm_dist = edge_data["distance"] * 10        # 0.2km -> 2, 1.2km -> 12
        norm_time = edge_data["time"]                 # already in minutes (3-15)
        norm_traf = min(1.0, edge_data["traffic"] * traffic_mult) * 10
        # RL factor: negative Q = penalized (high cost), positive Q = rewarded (low cost)
        # Map Q from [-5, +5] to [10, 0] — positive Q reduces cost
        norm_rl = max(0.0, 5.0 - q_value) * 2        # q=+5 -> 0, q=0 -> 10, q=-5 -> 20

        cost = (
            norm_dist * weights["w_dist"]
            + norm_time * weights["w_time"]
            + norm_traf * weights["w_traffic"]
            + norm_rl  * weights["w_rl"]
        )

        info = dict(edge_data, is_rush_hour=is_rush, traffic_mult=traffic_mult)

        logger.debug(
            f"  Cost({u}->{v}): dist={norm_dist:.1f}*{weights['w_dist']:.2f} "
            f"+ time={norm_time:.1f}*{weights['w_time']:.2f} "
            f"+ traf={norm_traf:.1f}*{weights['w_traffic']:.2f} "
            f"+ rl={norm_rl:.1f}*{weights['w_rl']:.2f} = {cost:.2f}"
        )

        return cost, info

    def get_neighbors(self, u):
        return [e["node"] for e in self.edges.get(u, [])]

    def get_raw_data(self):
        """Read map JSON safely for the API."""
        with open(self.map_file_path, "r") as f:
            return json.load(f)

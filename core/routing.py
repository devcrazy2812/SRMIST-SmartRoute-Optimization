"""
SmartRoute SRMIST - Routing module
Created by: devcrazy AKA Abhay Goyal
"""
import heapq
import math
from utils.logger import logger


class Router:
    """
    A* pathfinding engine with:
    - Haversine heuristic (admissible, never overestimates)
    - 4-factor edge cost: distance, time, traffic, RL preference
    - Per-segment explainability
    - Route statistics aggregation
    """

    def __init__(self, graph, rl_agent):
        self.graph = graph
        self.rl_agent = rl_agent

    # ── Haversine heuristic ─────────────────────────────
    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = la2 - la1, lo2 - lo1
        a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def _h(self, nid, goal):
        try:
            a, b = self.graph.node_data[nid], self.graph.node_data[goal]
            return self._haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        except KeyError:
            return 0.0

    # ── A* Search ───────────────────────────────────────
    def find_route(self, start, end, preference="balanced"):
        if start not in self.graph.nodes or end not in self.graph.nodes:
            return None, "Invalid start or end node."

        logger.info(
            f"A* search: {self.graph.nodes[start]} -> {self.graph.nodes[end]} "
            f"[{preference}]"
        )

        counter = 0
        h0 = self._h(start, end)
        # (f_cost, tiebreaker, g_cost, node, path, breakdown)
        pq = [(h0, counter, 0.0, start, [start], [])]
        visited = set()

        while pq:
            _f, _t, g, cur, path, steps = heapq.heappop(pq)

            if cur == end:
                return self._build_result(path, steps, preference), None

            if cur in visited:
                continue
            visited.add(cur)

            for nb in self.graph.get_neighbors(cur):
                if nb in visited:
                    continue

                # Get Q-value and pass it into the 4-factor cost function
                q_val = self.rl_agent.get_q_value(cur, nb)
                q_capped = max(-5.0, min(5.0, q_val))

                base_cost, info = self.graph.calculate_cost(cur, nb, preference, q_capped)
                if info is None:
                    continue

                eff = max(0.1, base_cost)
                new_g = g + eff
                counter += 1

                step = {
                    "from": cur,
                    "from_name": self.graph.nodes.get(cur, cur),
                    "to": nb,
                    "to_name": self.graph.nodes.get(nb, nb),
                    "distance": info["distance"],
                    "time": info["time"],
                    "traffic": info["traffic"],
                    "base_cost": round(base_cost, 3),
                    "q_value": round(q_val, 4),
                    "effective_cost": round(eff, 3),
                    "is_rush_hour": info.get("is_rush_hour", False),
                    "traffic_mult": info.get("traffic_mult", 1.0),
                }

                heapq.heappush(
                    pq,
                    (new_g + self._h(nb, end), counter, new_g, nb, path + [nb], steps + [step]),
                )

        logger.warning(f"No route: {start} -> {end}")
        return None, "No path exists between these locations."

    # ── Build rich response ─────────────────────────────
    def _build_result(self, path, steps, preference):
        total_cost = sum(s["base_cost"] for s in steps)
        total_dist = sum(s["distance"] for s in steps)
        total_time = sum(s["time"] for s in steps)
        avg_traffic = sum(s["traffic"] for s in steps) / max(len(steps), 1)

        logger.info(f"Route found! Cost={total_cost:.2f}, Dist={total_dist:.1f}km, Time={total_time}min")

        return {
            "path": path,
            "path_names": [self.graph.nodes[n] for n in path],
            "total_cost": round(total_cost, 2),
            "total_distance_km": round(total_dist, 2),
            "total_time_min": total_time,
            "avg_traffic": round(avg_traffic, 2),
            "num_segments": len(steps),
            "breakdown": steps,
            "explanation": self._explain(steps, preference),
        }

    # ── Explainability ──────────────────────────────────
    def _explain(self, steps, preference):
        parts = []

        # 1. Strategy
        parts.append(f"Strategy: '{preference}'.")

        # 2. Route stats
        d = sum(s["distance"] for s in steps)
        t = sum(s["time"] for s in steps)
        parts.append(f"Total: {d:.1f} km, ~{t} min, {len(steps)} segments.")

        # 3. Cost formula used
        w = self.graph.preferences.get(preference, {})
        parts.append(
            f"Weights: distance={w.get('w_dist',0):.0%}, time={w.get('w_time',0):.0%}, "
            f"traffic={w.get('w_traffic',0):.0%}, RL={w.get('w_rl',0):.0%}."
        )

        # 4. RL influence
        rl_pos = sum(1 for s in steps if s["q_value"] > 0)
        rl_neg = sum(1 for s in steps if s["q_value"] < 0)
        if rl_pos:
            parts.append(f"AI reinforced {rl_pos} segment(s) from positive feedback history.")
        if rl_neg:
            parts.append(f"AI penalized {rl_neg} segment(s) from negative feedback.")
        if not rl_pos and not rl_neg:
            parts.append("No RL bias yet -- provide feedback to train the model!")

        # 5. Rush hour
        rush = [s for s in steps if s.get("is_rush_hour")]
        if rush:
            mult = rush[0].get("traffic_mult", 2.0)
            parts.append(f"Rush-hour detected: traffic multiplier {mult}x active on {len(rush)} segment(s).")

        # 6. Traffic summary
        heavy = [s for s in steps if s["traffic"] > 0.6]
        if heavy and preference == "low_traffic":
            parts.append(f"Warning: {len(heavy)} high-traffic segment(s) unavoidable on this path.")
        elif not heavy and preference == "low_traffic":
            parts.append("Successfully routed around all high-traffic areas.")

        return " ".join(parts)

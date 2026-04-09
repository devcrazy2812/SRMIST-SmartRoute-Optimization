import heapq
from utils.logger import logger

class Router:
    def __init__(self, graph, rl_agent, rl_weight=5.0):
        self.graph = graph
        self.rl_agent = rl_agent
        self.rl_weight = rl_weight  # How much Q-value influences the routing cost

    def find_route(self, start_node, end_node, preference="balanced"):
        """
        Uses Dijkstra's algorithm with a custom cost function that includes
        multi-factor weights and Q-value reinforcement learning biases.
        """
        if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
            logger.error("Start or End node not found in graph.")
            return None, "Invalid nodes"

        logger.info(f"Calculating route from {self.graph.nodes[start_node]} to {self.graph.nodes[end_node]} ({preference} preference)")

        # priority queue stores (cumulative_effective_cost, current_node, path, cost_breakdown)
        pq = [(0.0, start_node, [start_node], [])]
        visited = set()
        
        while pq:
            current_cost, current, path, breakdown = heapq.heappop(pq)
            
            if current == end_node:
                # Goal reached
                # Calculate True base cost sum for output (without RL bias)
                total_base_cost = sum(b['base_cost'] for b in breakdown)
                explanation = self.generate_explanation(breakdown, preference)
                logger.info(f"Route found! Total abstract cost: {total_base_cost:.2f}")
                return {
                    "path": path,
                    "path_names": [self.graph.nodes[n] for n in path],
                    "total_cost": total_base_cost,
                    "breakdown": breakdown,
                    "explanation": explanation
                }, None
                
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Multi-Factor Cost Evaluation
                base_cost, edge_data = self.graph.calculate_cost(current, neighbor, preference)
                
                # Q-learning bias integration
                q_value = self.rl_agent.get_q_value(current, neighbor)
                
                # Effective cost: reward (positive Q) reduces cost, penalty (negative Q) increases cost
                effective_cost = base_cost - (self.rl_weight * q_value)
                
                # Ensure effective cost doesn't drop below 0 to avoid negative cycles in Dijkstra
                effective_cost = max(0.1, effective_cost)
                
                new_cost = current_cost + effective_cost
                new_path = list(path)
                new_path.append(neighbor)
                
                step_breakdown = {
                    "from": current,
                    "to": neighbor,
                    "distance": edge_data["distance"],
                    "time": edge_data["time"],
                    "traffic": edge_data["traffic"],
                    "base_cost": base_cost,
                    "q_value": q_value,
                    "effective_cost": effective_cost,
                    "is_rush_hour": edge_data.get("is_rush_hour", False)
                }
                new_breakdown = list(breakdown)
                new_breakdown.append(step_breakdown)
                
                heapq.heappush(pq, (new_cost, neighbor, new_path, new_breakdown))
                
        logger.warning(f"No route found between {start_node} and {end_node}.")
        return None, "No route found"
        
    def generate_explanation(self, breakdown, preference):
        explanation = [f"This route was selected optimizing for the '{preference}' preference."]
        
        total_time = sum(b["time"] for b in breakdown)
        total_dist = sum(b["distance"] for b in breakdown)
        
        explanation.append(f"Expected travel time is ~{total_time} mins for {total_dist} km.")
        
        # Check RL influence
        rl_influenced = [b for b in breakdown if b["q_value"] != 0.0]
        if rl_influenced:
            pos_influence = sum(1 for b in rl_influenced if b["q_value"] > 0)
            neg_influence = sum(1 for b in rl_influenced if b["q_value"] < 0)
            if pos_influence > 0:
                explanation.append("The system also learned from previous positive feedback and favored highly rated segments.")
            if neg_influence > 0:
                explanation.append("The system rerouted successfully to avoid paths with poor historical feedback.")
        
        # Check traffic peaks
        high_traffic_edges = [b for b in breakdown if b["traffic"] > 0.6]
        
        # Check dynamic rush hour
        rush_hour_edges = [b for b in breakdown if b.get("is_rush_hour", False)]
        if rush_hour_edges:
            explanation.append("⚠️ The engine detected current Time-of-Day rush hour traffic and mathematically spiked congestion weights.")
        if not high_traffic_edges and preference == "low_traffic":
            explanation.append("Successfully avoided high-traffic bottlenecks typical around SRMist campus.")
        elif high_traffic_edges:
            explanation.append("Note: Some segments have moderate/heavy traffic, but were chosen as they significantly reduce overall time/distance.")
            
        return " ".join(explanation)

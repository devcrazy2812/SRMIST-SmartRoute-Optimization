import sys
import os

# Append project root to path for imports to work nicely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import RoutingGraph
from core.rl_agent import RLAgent
from core.routing import Router
from utils.logger import logger

def display_menu(nodes):
    print("\n" + "="*50)
    print(" SRMIST Intelligent Route Optimization System ")
    print("="*50)
    print("\nAvailable Locations:")
    for k, v in nodes.items():
        print(f" - {k} ({v})")
    print("\nAvailable Preferences: fastest, shortest, low_traffic, balanced")
    print("="*50)

def main():
    graph = RoutingGraph("data/chennai_map.json")
    if not graph.nodes:
        print("Failed to load map data. Exiting...")
        return
        
    rl_agent = RLAgent("data/q_table.json")
    router = Router(graph, rl_agent, rl_weight=5.0)
    
    while True:
        display_menu(graph.nodes)
        
        start = input("\nEnter starting location ID (or 'quit' to exit): ").strip()
        if start.lower() == 'quit':
            break
            
        end = input("Enter destination ID: ").strip()
        pref = input("Enter preference (fastest/shortest/low_traffic/balanced): ").strip()
        
        if pref not in graph.preferences:
            pref = "balanced"
            print("Invalid preference. Using 'balanced'.")
            
        result, err = router.find_route(start, end, preference=pref)
        
        if err:
            print(f"\nError: {err}")
            continue
            
        print("\n--- RECOMMENDED ROUTE ---")
        print("Path: " + " -> ".join(result["path_names"]))
        print(f"Total Abstract Cost: {result['total_cost']:.2f}")
        print("\nExplanation:")
        print(result["explanation"])
        print("-" * 25)
        
        # User Feedback Loop
        feedback = input("\nDo you accept this route? (y/n) [Provides Reinforcement Reward]: ").strip().lower()
        reward = 0
        if feedback == 'y':
            reward = 1
            print("Feedback (+1) received! Modifying Q-table to favor these choices.")
        elif feedback == 'n':
            reward = -1
            print("Feedback (-1) received! Modifying Q-table to penalize these choices.")
            
        if reward != 0:
            rl_agent.apply_feedback_to_route(result["path"], reward)
            
        print("\nSystem Learned Successfully! Returning to main menu...")

if __name__ == "__main__":
    main()

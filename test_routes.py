"""
SmartRoute SRMIST — Verification for expanded 20-node graph.
"""
import sys, os
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

from core.graph import RoutingGraph
from core.rl_agent import RLAgent
from core.routing import Router

g = RoutingGraph("data/chennai_map.json")
a = RLAgent("data/q_table.json")
r = Router(g, a)

print(f"Graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
print(f"Node IDs: {list(g.nodes.keys())}")

# Test all 4 strategies on key pairs
pairs = [
    ("Potheri_Station", "Estancia"),
    ("Potheri_Station", "Tech_Park"),
    ("Mens_Hostel", "UB"),
    ("Womens_Hostel", "Java_Green"),
    ("SRM_Main_Gate", "Architecture_Block"),
    ("Clock_Tower", "Abode_Valley"),
    ("Main_Block", "SRM_Hospital"),
    ("Admin_Block", "Estancia"),
]
passes = 0
fails = 0
for s, e in pairs:
    for mode in ["fastest", "shortest", "low_traffic", "balanced"]:
        result, err = r.find_route(s, e, mode)
        if err:
            print(f"  FAIL: {s}->{e} [{mode}]: {err}")
            fails += 1
        else:
            path = " → ".join(result["path_names"])
            print(f"  OK [{mode:11s}] {path} (cost={result['total_cost']:.1f})")
            passes += 1

print(f"\n--- Results: {passes} passed, {fails} failed ---")

# Reset Q-table
a.q_table = {}
a.save_q_table()

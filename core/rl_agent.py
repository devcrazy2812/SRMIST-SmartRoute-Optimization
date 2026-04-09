"""
SmartRoute SRMIST - RL Agent module
Created by: devcrazy AKA Abhay Goyal
"""
import json
import os
from utils.logger import logger


class RLAgent:
    """
    Q-Learning Reinforcement Learning Agent.

    Implements the FULL Bellman update equation:
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))

    State  = current node (location)
    Action = next node (route choice)
    Reward = +1 (user accepted route) / -1 (user rejected route)

    The Q-table is persisted to JSON so the agent retains knowledge across restarts.
    """

    def __init__(self, q_table_path="data/q_table.json", alpha=0.1, gamma=0.9):
        self.q_table_path = q_table_path
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor — how much future reward matters
        self.q_table = {}
        self.load_q_table()

    # ── Persistence ──────────────────────────────────────
    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, "r") as f:
                    self.q_table = json.load(f)
                logger.info(f"Q-table loaded: {sum(len(v) for v in self.q_table.values())} entries.")
            except Exception as e:
                logger.error(f"Q-table load error: {e}")
                self.q_table = {}
        else:
            logger.info("Initializing empty Q-table.")

    def save_q_table(self):
        try:
            with open(self.q_table_path, "w") as f:
                json.dump(self.q_table, f, indent=2)
        except Exception as e:
            logger.error(f"Q-table save error: {e}")

    # ── Q-value access ───────────────────────────────────
    def get_q_value(self, state, action):
        """Q(state, action) — returns 0.0 for unseen pairs."""
        return self.q_table.get(state, {}).get(action, 0.0)

    def get_max_q(self, state):
        """max over a' of Q(state, a').  Returns 0.0 if state unseen."""
        actions = self.q_table.get(state, {})
        return max(actions.values()) if actions else 0.0

    # ── Core update ──────────────────────────────────────
    def update_q_value(self, state, action, reward, next_state=None):
        """
        Full Bellman update:
          Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a'(Q(s', a')) - Q(s, a))

        When next_state is None (terminal transition), gamma term is 0.
        Q-values are clamped to [-5, +5] to prevent runaway from spam feedback.
        """
        if state not in self.q_table:
            self.q_table[state] = {}

        old_q = self.get_q_value(state, action)

        # Future reward estimate
        future_q = self.get_max_q(next_state) if next_state else 0.0

        # Bellman equation
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

        # Clamp to prevent extreme values
        new_q = max(-5.0, min(5.0, new_q))

        self.q_table[state][action] = round(new_q, 6)

        logger.debug(
            f"Q({state}->{action}): {old_q:.4f} -> {new_q:.4f}  "
            f"[r={reward}, gamma*maxQ'={self.gamma * future_q:.4f}]"
        )

    # ── Trajectory feedback ──────────────────────────────
    def apply_feedback_to_route(self, route, reward):
        """
        Apply reward to every (state, action) transition in the route.
        Uses full Bellman: each transition knows its next_state so the
        discount factor properly propagates future value.

        route: [nodeA, nodeB, nodeC, nodeD]
        reward: +1 or -1
        """
        logger.info(f"Applying reward={reward} to {len(route)-1} transitions.")
        for i in range(len(route) - 1):
            state = route[i]
            action = route[i + 1]
            # next_state is the state we land in after taking this action
            next_state = route[i + 1] if i + 1 < len(route) - 1 else None
            self.update_q_value(state, action, reward, next_state)

        self.save_q_table()

    # ── Statistics for frontend ──────────────────────────
    def get_stats(self):
        """Summary stats for the analytics panel."""
        total_entries = sum(len(v) for v in self.q_table.values())
        positive = sum(1 for s in self.q_table.values() for v in s.values() if v > 0)
        negative = sum(1 for s in self.q_table.values() for v in s.values() if v < 0)
        return {
            "total_entries": total_entries,
            "positive_biases": positive,
            "negative_biases": negative,
            "states_explored": len(self.q_table),
        }

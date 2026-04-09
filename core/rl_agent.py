import json
import os
from utils.logger import logger

class RLAgent:
    def __init__(self, q_table_path="data/q_table.json", alpha=0.1, gamma=0.9):
        self.q_table_path = q_table_path
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as f:
                    self.q_table = json.load(f)
                logger.info("Loaded external Q-table successfully.")
            except Exception as e:
                logger.error(f"Error loading Q-table: {e}")
        else:
            logger.info("Initializing new empty Q-table.")
            
    def save_q_table(self):
        try:
            with open(self.q_table_path, 'w') as f:
                json.dump(self.q_table, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")

    def get_q_value(self, state, action):
        """
        State: current location (node id)
        Action: next node (route choice)
        """
        if state not in self.q_table:
            return 0.0
        return self.q_table[state].get(action, 0.0)

    def update_q_value(self, state, action, reward, next_state=None):
        """
        Update rule:
        Q[state][action] = Q[state][action] + alpha * (reward + gamma*max_next_q - Q[state][action])
        Since user feedback happens at the end of the route, we can just apply simple TD updates or 
        distribute the reward along the executed trajectory.
        Wait, prompt specifies:
        Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
        This is a simplified variant (gamma=0 basically), perfectly suited for straightforward trajectory reward.
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        
        old_q = self.get_q_value(state, action)
        
        # Simplified update rule per prompt
        new_q = old_q + self.alpha * (reward - old_q)
        
        self.q_table[state][action] = new_q
        logger.debug(f"Q-value updated for {state} -> {action}: {old_q:.3f} -> {new_q:.3f} (Reward: {reward})")
        self.save_q_table()

    def apply_feedback_to_route(self, route, reward):
        """
        Route: list of nodes selected e.g. [A, B, C, D]
        If user accepted, reward=+1. We apply this to all (state, action) transitions in the route.
        """
        logger.info(f"Applying reward={reward} to route: {' -> '.join(route)}")
        for i in range(len(route) - 1):
            state = route[i]
            action = route[i+1]
            self.update_q_value(state, action, reward)

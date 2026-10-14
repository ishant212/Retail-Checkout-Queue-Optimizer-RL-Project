import numpy as np
from collections import defaultdict
from .base import BaseQueueAgent

class ContextualBanditAgent(BaseQueueAgent):
    """
    Multi-Armed Bandit with UCB1 (Upper Confidence Bound) and Contextual Arm Payoffs.
    Arms: 41 discrete actions across 10 checkout lanes.
    Decision rule:
       a = argmax_a [ mu(s, a) + c * sqrt(ln(N(s) + 1) / (N(s, a) + 1)) ]
    Serves as an adaptive, zero-discount (gamma=0) immediate-reward baseline.
    Compatible with 10-lane environment, 35 tabular features, and 41 discrete actions.
    """
    def __init__(self, num_counters=10, max_queue_len=6, c_param=1.2, epsilon=0.1, num_actions=41):
        super().__init__("Multi-Armed Bandit", "bandit", num_counters, max_queue_len, num_actions)
        self.c_param = c_param
        self.epsilon = epsilon
        self.feature_dim = 35

        # State-action counts and mean estimated rewards
        self.counts = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.int32))
        self.rewards = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float64))

        # Global arm statistics
        self.global_counts = np.zeros(self.num_actions, dtype=np.int32)
        self.global_rewards = np.zeros(self.num_actions, dtype=np.float64)

    def state_to_context(self, state):
        """Derives documented context tuple from the 35 tabular features."""
        return self.state_to_key(state)

    def choose_action(self, state, explore=True):
        c = self.state_to_context(state)
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.choice(self.num_actions))

        state_counts = self.counts[c]
        total_n = np.sum(state_counts)
        if total_n == 0:
            # Cold-start state: prioritize open counter with shortest queue or action 0
            if hasattr(state, "queue_lengths"):
                return int(np.argmin(state.queue_lengths))
            elif isinstance(state, (tuple, list)) and len(state) >= 3:
                return int(np.argmin(state[:3]))
            return 0

        # UCB1 score for each arm
        ucb_values = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            n_a = state_counts[a]
            if n_a == 0:
                ucb_values[a] = 1e6
            else:
                bonus = self.c_param * np.sqrt(np.log(total_n + 1.0) / float(n_a))
                ucb_values[a] = self.rewards[c][a] + bonus

        return int(np.argmax(ucb_values))

    def update(self, state, action, reward, next_state=None, done=False):
        c = self.state_to_context(state)
        self.counts[c][action] += 1
        n = self.counts[c][action]
        self.rewards[c][action] += (reward - self.rewards[c][action]) / float(n)

        self.global_counts[action] += 1
        gn = self.global_counts[action]
        self.global_rewards[action] += (reward - self.global_rewards[action]) / float(gn)

    def get_action_values(self, state):
        c = self.state_to_context(state)
        state_counts = self.counts[c]
        total_n = np.sum(state_counts)
        vals = []
        for a in range(self.num_actions):
            n_a = state_counts[a]
            if n_a == 0:
                vals.append(float(self.global_rewards[a]))
            else:
                bonus = self.c_param * np.sqrt(np.log(total_n + 1.0) / float(n_a))
                vals.append(float(self.rewards[c][a] + bonus))
        return vals

import numpy as np
from collections import defaultdict
from .base import BaseQueueAgent

class SarsaAgent(BaseQueueAgent):
    """
    SARSA (On-Policy 1-Step Temporal Difference Control).
    Update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
    where A' is chosen using the agent's current epsilon-greedy policy.
    Compatible with 10-lane environment, 35 tabular features, and 41 discrete actions.
    """
    def __init__(self, num_counters=10, max_queue_len=6, gamma=0.95, alpha=0.1,
                 epsilon=0.4, epsilon_decay=0.995, min_epsilon=0.05, num_actions=41):
        super().__init__("SARSA", "value_based", num_counters, max_queue_len, num_actions)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.feature_dim = 35

        # Dynamic Q-table indexed by discretized state key
        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float64))

    def choose_action(self, state, explore=True):
        key = self.state_to_key(state)
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.choice(self.num_actions))
        return int(np.argmax(self.Q[key]))

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        s_key = self.state_to_key(state)
        s_next_key = self.state_to_key(next_state)

        if done:
            td_target = reward
        else:
            if next_action is None:
                next_action = self.choose_action(next_state, explore=True)
            td_target = reward + self.gamma * self.Q[s_next_key][next_action]

        td_error = td_target - self.Q[s_key][action]
        self.Q[s_key][action] += self.alpha * td_error
        return next_action

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_action_values(self, state):
        key = self.state_to_key(state)
        return [float(v) for v in self.Q[key]]

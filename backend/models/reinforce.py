import numpy as np
from .base import BaseQueueAgent
from .neural_net import MLP

class ReinforceAgent(BaseQueueAgent):
    """
    REINFORCE with State-Value Baseline (Episodic Monte Carlo Policy Gradient).
    - Policy Network (Actor): MLP parameterized policy pi_theta(a|s) with Softmax activation.
    - Baseline Network (Critic): MLP parameterized state value function V_w(s).
    - Continuous State: Operates directly on the complete 84-dimensional continuous normalized feature vector.
    - Standard Sutton & Barto (2018) episodic policy gradient with gamma^t discounting.
    """
    def __init__(self, num_counters=10, max_queue_len=6, gamma=0.95,
                 alpha_pi=0.005, alpha_v=0.01, num_actions=41):
        super().__init__("REINFORCE (w/ Baseline)", "policy_gradient", num_counters, max_queue_len, num_actions)
        self.gamma = gamma
        self.alpha_pi = alpha_pi
        self.alpha_v = alpha_v

        input_dim = self.feature_dim
        self.policy_net = MLP(input_dim=input_dim, hidden_dim=64, output_dim=self.num_actions, lr=alpha_pi, activation_out="softmax")
        self.value_net = MLP(input_dim=input_dim, hidden_dim=64, output_dim=1, lr=alpha_v, activation_out="linear")

        # Episodic buffer to store trajectory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def get_action_probabilities(self, state):
        feat = self.state_features(state)
        probs = self.policy_net.forward(feat)[0]
        return probs.tolist()

    def choose_action(self, state, explore=True):
        probs = np.array(self.get_action_probabilities(state), dtype=np.float64)
        probs = probs / np.sum(probs)
        if explore:
            return int(np.random.choice(self.num_actions, p=probs))
        return int(np.argmax(probs))

    def store_transition(self, state, action, reward, done=False):
        feat = self.state_features(state)
        self.episode_states.append(feat)
        self.episode_actions.append(int(action))
        self.episode_rewards.append(float(reward))

    def update_episode(self):
        """Perform Monte Carlo policy gradient update at the end of an episode."""
        T = len(self.episode_rewards)
        if T == 0:
            return

        # Compute discounted returns G_t
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = self.episode_rewards[t] + self.gamma * G
            returns[t] = G

        states = np.array(self.episode_states, dtype=np.float32)
        baseline_vals = self.value_net.forward(states).reshape(-1)
        deltas = returns - baseline_vals

        # 1. Update baseline network V(s) using MSE loss
        grad_v = (baseline_vals - returns).reshape(-1, 1)
        self.value_net.backward(grad_v)

        # 2. Update policy network pi(a|s)
        probs = self.policy_net.forward(states)
        grad_policy = np.zeros_like(probs)
        for t in range(T):
            a = self.episode_actions[t]
            delta = deltas[t]
            discount = self.gamma ** t
            # Gradient of negative log-likelihood w.r.t logits z: -(1_{a=A_t} - pi(a|s)) * gamma^t * delta
            grad = np.copy(probs[t])
            grad[a] -= 1.0
            grad *= discount * delta
            grad_policy[t] = grad

        self.policy_net.backward(grad_policy)

        # Clear buffer
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def update(self, state, action, reward, next_state=None, done=False):
        self.store_transition(state, action, reward, done=done)

    def get_action_values(self, state):
        feat = self.state_features(state)
        probs = self.policy_net.forward(feat)[0]
        val = float(self.value_net.forward(feat)[0, 0])
        return [float(val + np.log(max(1e-8, p))) for p in probs]

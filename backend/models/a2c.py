import numpy as np
from .base import BaseQueueAgent
from .neural_net import MLP

class A2CAgent(BaseQueueAgent):
    """
    Advantage Actor-Critic (A2C - 1-Step Synchronous Actor-Critic).
    - Actor Network: MLP parameterized policy pi_theta(a|s) with Softmax activation.
    - Critic Network: MLP parameterized state-value function V_w(s).
    - Continuous State: Operates directly on the complete 84-dimensional continuous normalized feature vector.
    - 1-Step TD Advantage: delta = R + gamma * (1 - done) * V(s') - V(s).
    - Mathematically rigorous entropy gradient through policy logits:
        grad_entropy = pi * (ln(pi) + H(pi)).
    """
    def __init__(self, num_counters=10, max_queue_len=6, gamma=0.95,
                 alpha_actor=0.003, alpha_critic=0.008, entropy_coef=0.01, num_actions=41):
        super().__init__("A2C (Advantage Actor-Critic)", "actor_critic", num_counters, max_queue_len, num_actions)
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.entropy_coef = entropy_coef

        input_dim = self.feature_dim
        self.actor = MLP(input_dim=input_dim, hidden_dim=64, output_dim=self.num_actions, lr=alpha_actor, activation_out="softmax")
        self.critic = MLP(input_dim=input_dim, hidden_dim=64, output_dim=1, lr=alpha_critic, activation_out="linear")

    def get_action_probabilities(self, state):
        feat = self.state_features(state)
        probs = self.actor.forward(feat)[0]
        return probs.tolist()

    def choose_action(self, state, explore=True):
        probs = np.array(self.get_action_probabilities(state), dtype=np.float64)
        probs = probs / np.sum(probs)
        if explore:
            return int(np.random.choice(self.num_actions, p=probs))
        return int(np.argmax(probs))

    def update(self, state, action, reward, next_state, done=False):
        feat = self.state_features(state)
        next_feat = self.state_features(next_state)

        # 1. Critic evaluation & 1-step TD target with terminal masking
        v_s = float(self.critic.forward(feat)[0, 0])
        v_next = 0.0 if done else float(self.critic.forward(next_feat)[0, 0])
        td_target = reward + self.gamma * v_next
        delta = td_target - v_s

        # 2. Update Critic (MSE loss: (V(s) - td_target)^2)
        critic_grad = np.array([[v_s - td_target]], dtype=np.float32)
        self.critic.backward(critic_grad)

        # 3. Update Actor (Policy Gradient + Entropy regularization w.r.t logits)
        probs = self.actor.forward(feat)[0]
        grad_policy = np.copy(probs)
        grad_policy[int(action)] -= 1.0
        grad_policy *= delta  # (probs - e_a) * delta is loss grad w.r.t logits

        # Exact entropy gradient w.r.t logits:
        # H(pi) = - sum(pi * ln(pi))
        # dH / dz_i = - pi_i * (ln(pi_i) + H(pi))
        # Loss contains - beta_ent * H(pi), so dLoss_ent / dz_i = beta_ent * pi_i * (ln(pi_i) + H(pi))
        log_probs = np.log(np.maximum(probs, 1e-8))
        entropy = -np.sum(probs * log_probs)
        grad_entropy = probs * (log_probs + entropy)

        actor_grad = (grad_policy + self.entropy_coef * grad_entropy).reshape(1, -1)
        self.actor.backward(actor_grad)

    def get_action_values(self, state):
        feat = self.state_features(state)
        probs = self.actor.forward(feat)[0]
        val = float(self.critic.forward(feat)[0, 0])
        return [float(val + np.log(max(1e-8, p))) for p in probs]

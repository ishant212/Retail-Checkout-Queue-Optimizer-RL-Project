import numpy as np
from .base import BaseQueueAgent
from .neural_net import MLP, ReplayBuffer

class DDPGAgent(BaseQueueAgent):
    """
    DDPG (Deep Deterministic Policy Gradient - Lillicrap et al. 2015).
    Continuous Action Simplex / Softmax Relaxation for Discrete Queue Management:
    - Deterministic Actor Network: mu(s) producing continuous action probabilities on simplex.
    - Critic Network: Q(s, a) evaluating state-action quality with continuous action inputs.
    - Differentiable Policy Gradient: dJ/d_theta = dQ/da * da/dz * dz/d_theta chained end-to-end.
    - Target Networks with Polyak soft parameter updates (tau) and terminal masking (1 - done).
    - Exploration via Gaussian logit perturbation noise.
    """
    def __init__(self, num_counters=10, max_queue_len=6, gamma=0.95,
                 lr_actor=0.003, lr_critic=0.006, tau=0.02, noise_std=0.2,
                 batch_size=32, num_actions=41):
        super().__init__("DDPG (Deep Deterministic PG)", "actor_critic", num_counters, max_queue_len, num_actions)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.min_noise_std = 0.05
        self.noise_decay = 0.995
        self.batch_size = batch_size

        input_dim = self.feature_dim
        # Actor outputs continuous action distribution over the 41 actions via Softmax
        self.actor = MLP(input_dim=input_dim, hidden_dim=64, output_dim=self.num_actions, lr=lr_actor, activation_out="softmax")
        self.actor_target = MLP(input_dim=input_dim, hidden_dim=64, output_dim=self.num_actions, lr=lr_actor, activation_out="softmax")
        self.actor_target.copy_weights_from(self.actor, tau=1.0)

        # Critic inputs: state_features (84) + action distribution (41) = 125 dims -> scalar Q(s, a)
        self.critic = MLP(input_dim=input_dim + self.num_actions, hidden_dim=64, output_dim=1, lr=lr_critic, activation_out="linear")
        self.critic_target = MLP(input_dim=input_dim + self.num_actions, hidden_dim=64, output_dim=1, lr=lr_critic, activation_out="linear")
        self.critic_target.copy_weights_from(self.critic, tau=1.0)

        self.replay_buffer = ReplayBuffer(capacity=2500)

    def _to_one_hot(self, actions):
        if np.isscalar(actions):
            oh = np.zeros(self.num_actions, dtype=np.float32)
            oh[int(actions)] = 1.0
            return oh
        oh = np.zeros((len(actions), self.num_actions), dtype=np.float32)
        for i, a in enumerate(actions):
            oh[i, int(a)] = 1.0
        return oh

    def choose_action(self, state, explore=True):
        feat = self.state_features(state)
        probs = self.actor.forward(feat)[0]
        if explore:
            noisy_logits = np.log(np.maximum(probs, 1e-8)) + np.random.normal(0, self.noise_std, size=self.num_actions)
            return int(np.argmax(noisy_logits))
        return int(np.argmax(probs))

    def update(self, state, action, reward, next_state, done=False):
        feat = self.state_features(state)
        next_feat = self.state_features(next_state)
        self.replay_buffer.push(feat, action, reward, next_feat, done=done)

        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # 1. Target Q calculation with target policy and terminal masking:
            # y = r + gamma * (1 - done) * Q_target(s', mu_target(s'))
            target_probs = self.actor_target.forward(next_states)
            next_sa = np.hstack([next_states, target_probs])
            target_q = self.critic_target.forward(next_sa).reshape(-1)
            y = rewards + self.gamma * (1.0 - dones.astype(np.float32)) * target_q

            # 2. Update Critic: minimize MSE (Q(s, a) - y)^2
            actions_oh = self._to_one_hot(actions)
            sa = np.hstack([states, actions_oh])
            pred_q = self.critic.forward(sa).reshape(-1)
            critic_grad = (pred_q - y).reshape(-1, 1)
            self.critic.backward(critic_grad)

            # 3. Update Actor using genuine Deterministic Policy Gradient:
            # Evaluate Q(s, mu(s)) and backpropagate through Critic into Actor
            current_probs = self.actor.forward(states)
            sa_actor = np.hstack([states, current_probs])
            self.critic.forward(sa_actor)

            # Gradient of Q w.r.t input (states and actions)
            ones_grad = np.ones((self.batch_size, 1), dtype=np.float32)
            d_sa = self.critic.backward(ones_grad, return_grad_input=True)
            dq_da = d_sa[:, self.feature_dim:]  # (batch_size, num_actions)

            # Softmax Jacobian chain rule:
            # dQ / dz_i = mu_i * ( dQ/da_i - sum_k mu_k * dQ/da_k )
            dot_prod = np.sum(current_probs * dq_da, axis=1, keepdims=True)
            dq_dz = current_probs * (dq_da - dot_prod)

            # Gradient ascent on J => minimize loss -J
            actor_grad = -dq_dz
            self.actor.backward(actor_grad)

            # 4. Soft target updates (Polyak averaging)
            self.actor_target.copy_weights_from(self.actor, tau=self.tau)
            self.critic_target.copy_weights_from(self.critic, tau=self.tau)

    def decay_epsilon(self):
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)

    def get_action_values(self, state):
        feat = self.state_features(state)
        vals = []
        for a in range(self.num_actions):
            a_oh = self._to_one_hot(a)
            sa = np.hstack([feat.reshape(1, -1), a_oh.reshape(1, -1)])
            q_val = float(self.critic.forward(sa)[0, 0])
            vals.append(q_val)
        return vals

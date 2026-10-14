import numpy as np
from .base import BaseQueueAgent
from .neural_net import MLP

class PPOAgent(BaseQueueAgent):
    """
    Proximal Policy Optimization (PPO - Schulman et al. 2017).
    - Actor Network: MLP parameterized policy pi_theta(a|s) with Softmax activation.
    - Critic Network: MLP parameterized state-value function V_w(s).
    - Continuous State: Operates directly on the complete 84-dimensional continuous normalized feature vector.
    - Generalized Advantage Estimation (GAE-lambda) with terminal masking.
    - Clipped Surrogate Objective with multi-epoch optimization and mini-batch shuffling.
    """
    def __init__(self, num_counters=10, max_queue_len=6, gamma=0.95, gae_lambda=0.95,
                 clip_eps=0.2, lr_actor=0.003, lr_critic=0.006, epochs=4,
                 batch_size=16, entropy_coef=0.01, num_actions=41):
        super().__init__("PPO (Proximal Policy Optimization)", "actor_critic", num_counters, max_queue_len, num_actions)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        input_dim = self.feature_dim
        self.actor = MLP(input_dim=input_dim, hidden_dim=64, output_dim=self.num_actions, lr=lr_actor, activation_out="softmax")
        self.critic = MLP(input_dim=input_dim, hidden_dim=64, output_dim=1, lr=lr_critic, activation_out="linear")

        # Rollout buffer
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_next_states = []
        self.buffer_old_probs = []
        self.buffer_dones = []

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

    def store_step(self, state, action, reward, next_state, done=False):
        feat = self.state_features(state)
        next_feat = self.state_features(next_state)
        probs = self.actor.forward(feat)[0]
        old_prob = float(probs[int(action)])

        self.buffer_states.append(feat)
        self.buffer_actions.append(int(action))
        self.buffer_rewards.append(float(reward))
        self.buffer_next_states.append(next_feat)
        self.buffer_old_probs.append(old_prob)
        self.buffer_dones.append(bool(done))

    def update(self, state, action, reward, next_state, done=False):
        self.store_step(state, action, reward, next_state, done=done)

    def optimize_batch(self):
        """Runs PPO clipped surrogate optimization over the collected trajectory."""
        n = len(self.buffer_rewards)
        if n == 0:
            return

        states = np.array(self.buffer_states, dtype=np.float32)
        next_states = np.array(self.buffer_next_states, dtype=np.float32)
        actions = np.array(self.buffer_actions, dtype=np.int32)
        rewards = np.array(self.buffer_rewards, dtype=np.float32)
        old_probs = np.array(self.buffer_old_probs, dtype=np.float32)
        dones = np.array(self.buffer_dones, dtype=np.bool_)

        # 1. Compute GAE and returns with terminal masking
        v_s = self.critic.forward(states).reshape(-1)
        v_next = self.critic.forward(next_states).reshape(-1)

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * non_terminal * v_next[t] - v_s[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + v_s[t]

        # Normalize advantages
        adv_std = np.std(advantages)
        if adv_std > 1e-8:
            norm_advantages = (advantages - np.mean(advantages)) / (adv_std + 1e-8)
        else:
            norm_advantages = advantages

        # 2. PPO Multi-Epoch Updates with shuffled mini-batches
        indices = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(indices)

            # Mini-batch loop
            for start_idx in range(0, n, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_probs = old_probs[batch_idx]
                b_adv = norm_advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Update Critic on mini-batch
                b_v = self.critic.forward(b_states).reshape(-1)
                critic_grad = (b_v - b_returns).reshape(-1, 1)
                self.critic.backward(critic_grad)

                # Update Actor with Clipped Surrogate gradient
                probs = self.actor.forward(b_states)
                grad_actor = np.zeros_like(probs)

                for i in range(len(batch_idx)):
                    a = b_actions[i]
                    adv = b_adv[i]
                    p_current = probs[i, a]
                    p_old = b_old_probs[i]

                    ratio = p_current / max(1e-8, p_old)
                    surr1 = ratio * adv
                    clipped_ratio = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    surr2 = clipped_ratio * adv

                    # Check if clipping is active
                    # When unclipped (surr1 <= surr2):
                    # d/dz [-ratio * adv] = adv / p_old * p_current * (probs[i] - e_a) = ratio * adv * (probs[i] - e_a)
                    if surr1 <= surr2:
                        grad = np.copy(probs[i])
                        grad[a] -= 1.0
                        grad_actor[i] = grad * (ratio * adv)
                    else:
                        # Clipped region: gradient is 0
                        grad_actor[i] = 0.0

                self.actor.backward(grad_actor)

        # Clear buffer
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_next_states.clear()
        self.buffer_old_probs.clear()
        self.buffer_dones.clear()

    def get_action_values(self, state):
        feat = self.state_features(state)
        probs = self.actor.forward(feat)[0]
        val = float(self.critic.forward(feat)[0, 0])
        return [float(val + np.log(max(1e-8, p))) for p in probs]

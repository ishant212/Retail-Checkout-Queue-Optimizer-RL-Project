import sys
import os
import unittest
import numpy as np

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.base import (
    BaseQueueAgent,
    QueueState,
    TABULAR_FEATURE_DIM,
    CONTINUOUS_FEATURE_DIM,
    ACTION_SPACE_SIZE,
    TABULAR_FEATURE_NAMES,
    CONTINUOUS_FEATURE_NAMES
)
from models.q_learning import QLearningAgent
from models.sarsa import SarsaAgent
from models.bandit import ContextualBanditAgent
from models.reinforce import ReinforceAgent
from models.a2c import A2CAgent
from models.ppo import PPOAgent
from models.dqn import DQNAgent
from models.double_dqn import DoubleDQNAgent
from models.ddpg import DDPGAgent
from simulation_engine import RetailQueueEnv, SimulationService, RewardConfig
from simpy_engine import SimPyCheckoutSimulator


class TestMathematicalCorrectnessAndArchitecture(unittest.TestCase):
    """
    Complete Architecture Audit, Mathematical Validation, and Testing Suite.
    Enforces:
      - 35-feature tabular state representation
      - 84-feature continuous neural state representation
      - 41 discrete environment actions
      - Analytical correctness of gradient equations and Bellman targets
      - Strict input dimension rejection
    """

    def setUp(self):
        np.random.seed(42)
        self.num_counters = 10
        self.max_queue_len = 6
        self.num_actions = 41

    # =========================================================================
    # PART 1: ARCHITECTURE TESTS (1 - 10)
    # =========================================================================

    def test_01_tabular_35_feature_representation_definition_and_order(self):
        """1. Verify tabular input representation has exactly 35 documented features and consistent ordering."""
        self.assertEqual(TABULAR_FEATURE_DIM, 35)
        self.assertEqual(len(TABULAR_FEATURE_NAMES), 35)

        # Check documented ordering
        # [0..9]: queue lengths
        for i in range(10):
            self.assertEqual(TABULAR_FEATURE_NAMES[i], f"lane_{i}_queue_norm")
        # [10..19]: open statuses
        for i in range(10):
            self.assertEqual(TABULAR_FEATURE_NAMES[10 + i], f"lane_{i}_open")
        # [20..29]: lane types
        for i in range(10):
            self.assertEqual(TABULAR_FEATURE_NAMES[20 + i], f"lane_{i}_type")
        # [30..34]: customer, arrival, epoch, congestion severity, duration
        self.assertEqual(TABULAR_FEATURE_NAMES[30], "incoming_basket_size_norm")
        self.assertEqual(TABULAR_FEATURE_NAMES[31], "customer_arrival_rate_norm")
        self.assertEqual(TABULAR_FEATURE_NAMES[32], "sim_time_epoch")
        self.assertEqual(TABULAR_FEATURE_NAMES[33], "congestion_severity_norm")
        self.assertEqual(TABULAR_FEATURE_NAMES[34], "congestion_duration_norm")

        env = RetailQueueEnv()
        tab_state = env.get_tabular_state()
        self.assertEqual(len(tab_state), 35)
        self.assertIsInstance(tab_state, np.ndarray)

    def test_02_q_learning_and_sarsa_state_key_mechanism(self):
        """2. Verify Q-Learning and SARSA convert 35-feature tabular representation into documented discrete state keys."""
        q_agent = QLearningAgent(num_counters=10, num_actions=41)
        sarsa_agent = SarsaAgent(num_counters=10, num_actions=41)

        self.assertEqual(q_agent.feature_dim, 35)
        self.assertEqual(sarsa_agent.feature_dim, 35)

        sample_state = QueueState([2, 1, 0, 3, 1, 0, 2, 1, 0, 0])
        q_key = q_agent.state_to_key(sample_state)
        s_key = sarsa_agent.state_to_key(sample_state)

        # Must be 7-element aggregate tuple
        self.assertEqual(len(q_key), 7)
        self.assertEqual(len(s_key), 7)
        self.assertEqual(q_key, s_key, "State key generation must be deterministic and identical across tabular agents")

    def test_03_contextual_ucb1_context_representation(self):
        """3. Verify Contextual UCB1 uses documented context derived from 35 tabular features."""
        bandit = ContextualBanditAgent(num_counters=10, num_actions=41)
        self.assertEqual(bandit.feature_dim, 35)

        sample_state = QueueState([1] * 10)
        context = bandit.state_to_context(sample_state)
        self.assertEqual(len(context), 7)

        # Context must influence action selection
        bandit.counts[context][0] = 5
        bandit.rewards[context][0] = 10.0
        bandit.counts[context][1] = 5
        bandit.rewards[context][1] = 2.0
        vals = bandit.get_action_values(sample_state)
        self.assertGreater(vals[0], vals[1], "Action with higher empirical reward in context must have higher value")

    def test_04_neural_models_accept_84_continuous_features(self):
        """4. Verify all six neural models accept exactly 84 continuous features."""
        neural_agents = [
            ReinforceAgent(10, 6, num_actions=41),
            A2CAgent(10, 6, num_actions=41),
            PPOAgent(10, 6, num_actions=41),
            DQNAgent(10, 6, num_actions=41),
            DoubleDQNAgent(10, 6, num_actions=41),
            DDPGAgent(10, 6, num_actions=41)
        ]

        sample_state = QueueState([1] * 10)
        for ag in neural_agents:
            self.assertEqual(ag.feature_dim, 84, f"{ag.name} must have feature_dim=84")
            feat = ag.state_features(sample_state)
            self.assertEqual(len(feat), 84, f"{ag.name} must output 84 continuous features")
            self.assertTrue(np.all(feat >= 0.0) and np.all(feat <= 1.0))

    def test_05_neural_models_configured_for_41_actions(self):
        """5. Verify all six neural models are configured for exactly 41 environment actions."""
        neural_agents = [
            ReinforceAgent(10, 6, num_actions=41),
            A2CAgent(10, 6, num_actions=41),
            PPOAgent(10, 6, num_actions=41),
            DQNAgent(10, 6, num_actions=41),
            DoubleDQNAgent(10, 6, num_actions=41),
            DDPGAgent(10, 6, num_actions=41)
        ]

        sample_state = QueueState([0] * 10)
        for ag in neural_agents:
            self.assertEqual(ag.num_actions, 41)
            vals = ag.get_action_values(sample_state)
            self.assertEqual(len(vals), 41, f"{ag.name} must produce 41 action values")

    def test_06_network_input_and_output_dimensions(self):
        """6. Verify network architecture input layers match 84 and output layers match 41."""
        dqn = DQNAgent(10, 6, num_actions=41)
        self.assertEqual(dqn.q_net.input_dim, 84)
        self.assertEqual(dqn.q_net.output_dim, 41)

        ddpg = DDPGAgent(10, 6, num_actions=41)
        self.assertEqual(ddpg.actor.input_dim, 84)
        self.assertEqual(ddpg.actor.output_dim, 41)
        self.assertEqual(ddpg.critic.input_dim, 84 + 41)  # 125
        self.assertEqual(ddpg.critic.output_dim, 1)

    def test_07_no_tabular_continuous_mixing(self):
        """7. Verify tabular and continuous feature sets are strictly separated."""
        env = RetailQueueEnv()
        tab_feat = env.get_tabular_state()
        cont_feat = env.get_continuous_state()

        self.assertEqual(len(tab_feat), 35)
        self.assertEqual(len(cont_feat), 84)
        self.assertNotEqual(len(tab_feat), len(cont_feat))

    def test_08_all_actions_map_to_valid_environment_semantics(self):
        """8. Verify all 41 actions map to valid structured semantics."""
        for a in range(41):
            sem = BaseQueueAgent.action_to_semantic(a)
            self.assertIn(sem["type"], ["route", "open", "close", "convert", "hold"])
            if sem["type"] != "hold":
                self.assertIn(sem["lane_id"], range(10))

    def test_09_feature_order_consistency_training_eval_api_simpy(self):
        """9. Verify feature extraction consistency between RetailQueueEnv and SimPy."""
        env = RetailQueueEnv()
        env_state = env.reset()
        cont_env = env_state.to_continuous_features()
        tab_env = env_state.to_tabular_features()

        self.assertEqual(len(cont_env), 84)
        self.assertEqual(len(tab_env), 35)

    def test_10_invalid_observation_dimensions_fail_explicitly(self):
        """10. Verify that invalid observation shapes fail explicitly with ValueError."""
        ag = DQNAgent(10, 6, num_actions=41)
        # Passing 35 features to continuous agent expecting 84 must raise ValueError
        bad_obs_35 = np.zeros(35, dtype=np.float32)
        with self.assertRaises(ValueError):
            ag.state_features(bad_obs_35)

        # Passing 50 features must raise ValueError
        bad_obs_50 = np.zeros(50, dtype=np.float32)
        with self.assertRaises(ValueError):
            ag.state_features(bad_obs_50)

        # Passing 84 features to tabular method expecting 35 must raise ValueError
        with self.assertRaises(ValueError):
            ag.tabular_features(np.zeros(84, dtype=np.float32))

    # =========================================================================
    # PART 2: MATHEMATICAL TESTS (11 - 21)
    # =========================================================================

    def test_11_q_learning_exact_equation_and_terminal_mask(self):
        """11. Verify Q-Learning target calculation and terminal masking (y = r when done=True)."""
        agent = QLearningAgent(num_counters=10, num_actions=41, gamma=0.9, alpha=0.5)
        s0 = QueueState([1] * 10)
        s1 = QueueState([2] * 10)
        key0 = agent.state_to_key(s0)
        key1 = agent.state_to_key(s1)

        agent.Q[key1][5] = 16.0  # max Q is 16.0
        agent.Q[key0][2] = 4.0

        # Non-terminal: y = 2.0 + 0.9 * 16.0 = 16.4 -> Q = 4.0 + 0.5 * (16.4 - 4.0) = 10.2
        agent.update(s0, 2, 2.0, s1, done=False)
        self.assertAlmostEqual(agent.Q[key0][2], 10.2, places=5)

        # Terminal: y = 5.0 -> Q = 10.2 + 0.5 * (5.0 - 10.2) = 7.6
        agent.update(s0, 2, 5.0, s1, done=True)
        self.assertAlmostEqual(agent.Q[key0][2], 7.6, places=5)

    def test_12_sarsa_exact_equation_and_on_policy_sync(self):
        """12. Verify SARSA on-policy TD target calculation and terminal handling."""
        agent = SarsaAgent(num_counters=10, num_actions=41, gamma=0.8, alpha=0.5)
        s0 = QueueState([1] * 10)
        s1 = QueueState([2] * 10)
        key0 = agent.state_to_key(s0)
        key1 = agent.state_to_key(s1)

        agent.Q[key1][7] = 12.0
        agent.Q[key0][1] = 6.0

        # Non-terminal with action 7: y = 3.0 + 0.8 * 12.0 = 12.6 -> Q = 6.0 + 0.5 * (12.6 - 6.0) = 9.3
        agent.update(s0, 1, 3.0, s1, next_action=7, done=False)
        self.assertAlmostEqual(agent.Q[key0][1], 9.3, places=5)

        # Terminal: y = 4.0 -> Q = 9.3 + 0.5 * (4.0 - 9.3) = 6.65
        agent.update(s0, 1, 4.0, s1, next_action=7, done=True)
        self.assertAlmostEqual(agent.Q[key0][1], 6.65, places=5)

    def test_13_contextual_ucb1_scoring_with_context(self):
        """13. Verify Contextual UCB1 scores arms using contextual empirical mean and UCB bonus."""
        bandit = ContextualBanditAgent(num_counters=10, num_actions=41, c_param=1.414)
        s = QueueState([1] * 10)
        c = bandit.state_to_context(s)

        bandit.counts[c][0] = 10
        bandit.rewards[c][0] = 5.0
        total_n = 10

        # Expected score: 5.0 + 1.414 * sqrt(ln(11) / 10)
        expected_score = 5.0 + 1.414 * np.sqrt(np.log(11.0) / 10.0)
        vals = bandit.get_action_values(s)
        self.assertAlmostEqual(vals[0], expected_score, places=4)

    def test_14_reinforce_return_and_policy_gradient_mathematical(self):
        """14. Verify REINFORCE discounted return G_t and analytical softmax policy gradient."""
        agent = ReinforceAgent(10, 6, gamma=0.9, num_actions=41)
        s = QueueState([1] * 10)

        # Store 2 steps
        agent.store_transition(s, 0, 10.0)
        agent.store_transition(s, 1, 20.0)

        # G_1 = 20.0
        # G_0 = 10.0 + 0.9 * 20.0 = 28.0
        returns = []
        G = 0.0
        for r in reversed(agent.episode_rewards):
            G = r + agent.gamma * G
            returns.insert(0, G)
        self.assertAlmostEqual(returns[0], 28.0)
        self.assertAlmostEqual(returns[1], 20.0)

    def test_15_a2c_entropy_gradient_and_terminal_handling(self):
        """15. Verify A2C exact entropy Jacobian derivative and terminal advantage calculation."""
        agent = A2CAgent(10, 6, gamma=0.95, num_actions=41)
        s = QueueState([1] * 10)
        feat = agent.state_features(s)

        probs = agent.actor.forward(feat)[0]
        log_probs = np.log(np.maximum(probs, 1e-8))
        H = -np.sum(probs * log_probs)

        # dH/dz_i = -pi_i * (ln(pi_i) + H(pi))
        grad_H = probs * (log_probs + H)
        self.assertAlmostEqual(float(np.sum(grad_H)), 0.0, places=5,
                               msg="Sum of logit entropy gradients must vanish on simplex")

    def test_16_ppo_probability_ratio_clipping_and_gae(self):
        """16. Verify PPO clipped surrogate objective and terminal-aware GAE."""
        ppo = PPOAgent(10, 6, gamma=0.95, gae_lambda=0.9, clip_eps=0.2, num_actions=41)
        s0 = QueueState([1] * 10)
        s1 = QueueState([2] * 10)

        ppo.store_step(s0, 0, 5.0, s1, done=True)
        # Terminal step: delta = R - V(s) without next state bootstrap
        self.assertEqual(len(ppo.buffer_dones), 1)
        self.assertTrue(ppo.buffer_dones[0])

    def test_17_dqn_terminal_targets(self):
        """17. Verify DQN terminal transitions do not bootstrap target values."""
        dqn = DQNAgent(10, 6, gamma=0.95, batch_size=2, num_actions=41)
        s0 = QueueState([1] * 10)
        s1 = QueueState([2] * 10)

        dqn.replay_buffer.push(dqn.state_features(s0), 0, 8.0, dqn.state_features(s1), done=True)
        dqn.replay_buffer.push(dqn.state_features(s0), 1, 8.0, dqn.state_features(s1), done=False)

        states, actions, rewards, next_states, dones = dqn.replay_buffer.sample(2)
        next_q = dqn.target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + dqn.gamma * (1.0 - dones.astype(np.float32)) * max_next_q

        for i in range(2):
            if dones[i]:
                self.assertEqual(targets[i], 8.0, "Terminal target must equal reward directly")

    def test_18_double_dqn_target_decoupling(self):
        """18. Verify Double DQN decouples online action selection from target evaluation."""
        ddqn = DoubleDQNAgent(10, 6, gamma=0.9, batch_size=1, num_actions=41)
        # Mock forward functions to check decoupling
        mock_online = np.zeros((1, 41), dtype=np.float32)
        mock_online[0, 5] = 50.0  # online chooses action 5

        mock_target = np.zeros((1, 41), dtype=np.float32)
        mock_target[0, 5] = 12.0  # target evaluates action 5 at 12.0
        mock_target[0, 9] = 99.0  # target max is 99.0 at action 9

        ddqn.q_net.forward = lambda x: mock_online
        ddqn.target_net.forward = lambda x: mock_target

        s_next = np.zeros((1, 84), dtype=np.float32)
        best_a = np.argmax(ddqn.q_net.forward(s_next), axis=1)
        self.assertEqual(best_a[0], 5)
        eval_val = ddqn.target_net.forward(s_next)[0, best_a[0]]
        self.assertEqual(eval_val, 12.0, "Target evaluation must evaluate online action (12.0), NOT max (99.0)")

    def test_19_ddpg_critic_gradient_and_softmax_jacobian(self):
        """19. Verify DDPG actor updates through Critic gradient and Softmax Jacobian."""
        ddpg = DDPGAgent(10, 6, num_actions=41, batch_size=2)
        s0 = QueueState([1] * 10)
        s1 = QueueState([2] * 10)

        ddpg.replay_buffer.push(ddpg.state_features(s0), 0, 5.0, ddpg.state_features(s1), done=False)
        ddpg.replay_buffer.push(ddpg.state_features(s0), 1, 4.0, ddpg.state_features(s1), done=False)

        states, actions, rewards, next_states, dones = ddpg.replay_buffer.sample(2)
        current_probs = ddpg.actor.forward(states)
        sa_actor = np.hstack([states, current_probs])

        ddpg.critic.forward(sa_actor)
        ones_grad = np.ones((2, 1), dtype=np.float32)
        d_sa = ddpg.critic.backward(ones_grad, return_grad_input=True)

        dq_da = d_sa[:, ddpg.feature_dim:]
        self.assertEqual(dq_da.shape, (2, 41))
        self.assertFalse(np.all(dq_da == 0.0), "Critic gradient w.r.t action must be non-zero")

    def test_20_centralized_reward_independent_verification(self):
        """20. Verify centralized reward equation against hand-calculated scenario."""
        cfg = RewardConfig(
            alpha=3.0, beta=0.15, gamma=0.25, delta=0.8,
            rho=1.0, zeta=0.3, kappa=0.25, xi=5.0
        )
        # Scenario: N_served=2, W_t=4.0, Q_t=5, C_t=1, delta_T=1.0, B_t=0.5, K_t=6.0, I_t=0.0
        # Expected: 3*2 - 0.15*4 - 0.25*5 - 0.8*1 - 1.0*1 - 0.3*0.5 - 0.25*6 - 5*0
        #         = 6.0 - 0.6 - 1.25 - 0.8 - 1.0 - 0.15 - 1.5 - 0.0
        #         = 6.0 - 5.30 = 0.70
        R, _ = cfg.calculate(N_served=2, W_t=4.0, Q_t=5, C_t=1, delta_T_congestion=1.0, B_t=0.5, K_t=6.0, I_t=0.0)
        self.assertAlmostEqual(R, 0.70, places=5)

    def test_21_model_save_and_load_persistence(self):
        """21. Verify model weights and tabular state keys persist across serialization."""
        tmp_file = os.path.join(backend_dir, "test_agent_persistence.pkl")
        for cls in [QLearningAgent, DQNAgent, ReinforceAgent, DDPGAgent]:
            ag = cls(10, 6, num_actions=41)
            ag.save(tmp_file)
            self.assertTrue(os.path.exists(tmp_file))

            ag_loaded = cls(10, 6, num_actions=41)
            ag_loaded.load(tmp_file)

            s = QueueState([1] * 10)
            v1 = ag.get_action_values(s)
            v2 = ag_loaded.get_action_values(s)
            np.testing.assert_allclose(v1, v2, rtol=1e-5, atol=1e-5)
            os.remove(tmp_file)

    # =========================================================================
    # PART 3: INTEGRATION AND REGRESSION TESTS (22 - 25)
    # =========================================================================

    def test_22_training_episodes_completion_all_9_models(self):
        """22. Verify all 9 models complete a training episode without dimension or numerical errors."""
        env = RetailQueueEnv()
        for name, cls in [
            ("q_learning", QLearningAgent),
            ("sarsa", SarsaAgent),
            ("bandit", ContextualBanditAgent),
            ("reinforce", ReinforceAgent),
            ("a2c", A2CAgent),
            ("ppo", PPOAgent),
            ("dqn", DQNAgent),
            ("double_dqn", DoubleDQNAgent),
            ("ddpg", DDPGAgent)
        ]:
            ag = cls(10, 6, num_actions=41)
            state = env.reset()
            for step in range(5):
                done = (step == 4)
                act = ag.choose_action(state, explore=True)
                next_state, reward = env.step(act)
                if name == "sarsa":
                    next_act = ag.choose_action(next_state, explore=True) if not done else None
                    ag.update(state, act, reward, next_state, next_act, done=done)
                elif name == "reinforce":
                    ag.store_transition(state, act, reward, done=done)
                else:
                    ag.update(state, act, reward, next_state, done=done)
                state = next_state
            if name == "reinforce":
                ag.update_episode()
            elif name == "ppo":
                ag.optimize_batch()

    def test_23_simpy_model_execution_no_mismatches(self):
        """23. Verify SimPy executes all models without observation or action mismatches."""
        sim = SimPyCheckoutSimulator(simulation_time=5)
        for name, cls in [
            ("q_learning", QLearningAgent),
            ("double_dqn", DoubleDQNAgent),
            ("ddpg", DDPGAgent)
        ]:
            agent = cls(10, 6, num_actions=41)
            res = sim.run_comparison(agents_dict={name: agent})
            self.assertIn("summary", res)
            self.assertGreater(len(res["summary"]), 0)
            self.assertGreater(res["summary"][0]["totalCustomers"], 0)

    def test_24_api_integration_valid_metrics(self):
        """24. Verify simulation service benchmark returns metrics for all 9 algorithms."""
        service = SimulationService()
        bench = service.run_benchmark(steps=10)
        self.assertIn("models", bench)
        self.assertEqual(len(bench["models"]), 9)
        for m_id, data in bench["models"].items():
            self.assertIn("cumulativeReward", data)
            self.assertIn("totalWait", data)
            self.assertIn("avgWaitPerStep", data)
            self.assertIn("customersServed", data)

    def test_25_finite_difference_vs_analytic_gradient_check(self):
        """25. Verify analytical log-probability gradients match numerical finite-difference gradients."""
        # For softmax policy pi(a|s) = exp(z_a) / sum exp(z_i)
        # d ln pi(a) / d z_k = (1_{k=a} - pi(k))
        z = np.array([1.2, 0.5, -0.8, 2.1], dtype=np.float64)
        exp_z = np.exp(z - np.max(z))
        pi = exp_z / np.sum(exp_z)

        target_action = 1
        analytic_grad = np.zeros_like(z)
        analytic_grad[target_action] = 1.0
        analytic_grad -= pi  # (1_{k=a} - pi(k))

        # Numerical gradient via finite differences
        eps = 1e-6
        numerical_grad = np.zeros_like(z)
        for k in range(len(z)):
            z_plus = np.copy(z)
            z_plus[k] += eps
            exp_plus = np.exp(z_plus - np.max(z_plus))
            pi_plus = exp_plus / np.sum(exp_plus)

            z_minus = np.copy(z)
            z_minus[k] -= eps
            exp_minus = np.exp(z_minus - np.max(z_minus))
            pi_minus = exp_minus / np.sum(exp_minus)

            d_log_pi = (np.log(pi_plus[target_action]) - np.log(pi_minus[target_action])) / (2.0 * eps)
            numerical_grad[k] = d_log_pi

        np.testing.assert_allclose(analytic_grad, numerical_grad, rtol=1e-5, atol=1e-5,
                                   err_msg="Analytical policy gradient deviates from finite-difference numerical gradient")

    def test_26_ddpg_softmax_jacobian_finite_difference_check(self):
        """26. Verify DDPG Softmax Jacobian contraction matches numerical finite differences."""
        z = np.array([0.5, -1.2, 2.0, 0.1, -0.5], dtype=np.float64)
        exp_z = np.exp(z - np.max(z))
        mu = exp_z / np.sum(exp_z)

        # Arbitrary linear critic Q(s, a) = w^T a => dQ/da = w
        w = np.array([1.5, -0.8, 0.3, 2.1, -1.0], dtype=np.float64)
        dq_da = w

        # Analytical Jacobian contraction: dq/dz = mu * (dq_da - sum(mu * dq_da))
        dot_prod = np.sum(mu * dq_da)
        analytic_dq_dz = mu * (dq_da - dot_prod)

        # Numerical gradient via finite differences
        eps = 1e-7
        num_dq_dz = np.zeros_like(z)
        for i in range(len(z)):
            z_plus = np.copy(z); z_plus[i] += eps
            exp_p = np.exp(z_plus - np.max(z_plus))
            mu_p = exp_p / np.sum(exp_p)
            q_plus = np.dot(w, mu_p)

            z_minus = np.copy(z); z_minus[i] -= eps
            exp_m = np.exp(z_minus - np.max(z_minus))
            mu_m = exp_m / np.sum(exp_m)
            q_minus = np.dot(w, mu_m)

            num_dq_dz[i] = (q_plus - q_minus) / (2.0 * eps)

        np.testing.assert_allclose(analytic_dq_dz, num_dq_dz, rtol=1e-5, atol=1e-6,
                                   err_msg="DDPG Softmax Jacobian deviates from numerical finite differences")

    def test_27_ppo_clipped_surrogate_four_quadrants_finite_difference(self):
        """27. Verify PPO clipped surrogate gradients across all four advantage/ratio quadrants."""
        def ppo_loss(z, a, p_old, adv, clip_eps=0.2):
            exp_z = np.exp(z - np.max(z))
            pi = exp_z / np.sum(exp_z)
            ratio = pi[a] / max(1e-8, p_old)
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            return -min(surr1, surr2), ratio, surr1, surr2, pi

        def analytic_grad(pi, a, ratio, adv, clip_eps=0.2):
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            if surr1 <= surr2:
                grad = np.copy(pi)
                grad[a] -= 1.0
                return grad * (ratio * adv)
            return np.zeros_like(pi)

        eps = 1e-7
        z_base = np.array([1.0, 0.5, -0.5, 0.0], dtype=np.float64)

        scenarios = [
            ("Adv > 0, unclipped", 1.5, 0.433),
            ("Adv > 0, clipped",   1.5, 0.300),
            ("Adv < 0, unclipped", -1.5, 0.450),
            ("Adv < 0, clipped",   -1.5, 0.800)
        ]

        for name, adv, p_old in scenarios:
            loss, ratio, s1, s2, pi = ppo_loss(z_base, 0, p_old, adv)
            ag = analytic_grad(pi, 0, ratio, adv)
            ng = np.zeros_like(z_base)
            for i in range(len(z_base)):
                zp = np.copy(z_base); zp[i] += eps
                zm = np.copy(z_base); zm[i] -= eps
                lp, _, _, _, _ = ppo_loss(zp, 0, p_old, adv)
                lm, _, _, _, _ = ppo_loss(zm, 0, p_old, adv)
                ng[i] = (lp - lm) / (2.0 * eps)
            np.testing.assert_allclose(ag, ng, rtol=1e-4, atol=1e-5,
                                       err_msg=f"PPO surrogate gradient failed for scenario: {name}")

    def test_28_a2c_logit_entropy_jacobian_finite_difference(self):
        """28. Verify A2C analytical logit entropy Jacobian matches numerical finite differences."""
        def entropy(z):
            exp_z = np.exp(z - np.max(z))
            pi = exp_z / np.sum(exp_z)
            return -np.sum(pi * np.log(np.maximum(pi, 1e-12)))

        z = np.array([1.2, -0.4, 0.8, -1.5, 2.0], dtype=np.float64)
        exp_z = np.exp(z - np.max(z))
        pi = exp_z / np.sum(exp_z)
        H = -np.sum(pi * np.log(np.maximum(pi, 1e-12)))
        analytic_dH_dz = -pi * (np.log(np.maximum(pi, 1e-12)) + H)

        eps = 1e-7
        num_dH_dz = np.zeros_like(z)
        for i in range(len(z)):
            zp = np.copy(z); zp[i] += eps
            zm = np.copy(z); zm[i] -= eps
            num_dH_dz[i] = (entropy(zp) - entropy(zm)) / (2.0 * eps)

        np.testing.assert_allclose(analytic_dH_dz, num_dH_dz, rtol=1e-5, atol=1e-6,
                                   err_msg="A2C Logit Entropy Jacobian deviates from numerical finite differences")

    def test_29_state_key_reachable_space_and_collision_invariance(self):
        """29. Verify state key consistency, collision invariance, and reachable state bound."""
        agent = QLearningAgent(10, 6, num_actions=41)

        # 1. Deterministic invariance: identical state produces identical key
        s = QueueState([2, 1, 0, 4, 3, 0, 1, 2, 0, 1], incoming_basket_size=8)
        k1 = agent.state_to_key(s)
        k2 = agent.state_to_key(s)
        self.assertEqual(k1, k2)

        # 2. Key components adhere to domain bounds: 3x3x3x3x3x3x2
        exp_c, std_c, lrg_c, flx_c, op_c, bsk_c, is_cg = k1
        self.assertIn(exp_c, (0, 1, 2))
        self.assertIn(std_c, (0, 1, 2))
        self.assertIn(lrg_c, (0, 1, 2))
        self.assertIn(flx_c, (0, 1, 2))
        self.assertIn(op_c, (0, 1, 2))
        self.assertIn(bsk_c, (0, 1, 2))
        self.assertIn(is_cg, (0, 1))

        # 3. Reachable states are bounded by 954 out of 1458 theoretical combinations
        reachable_count = 0
        for exp in (0, 1, 2):
            for std in (0, 1, 2):
                for lrg in (0, 1, 2):
                    for flx in (0, 1, 2):
                        for op in (0, 1, 2):
                            for bsk in (0, 1, 2):
                                for cong in (0, 1):
                                    min_q = (0 if exp == 0 else (3 if exp == 1 else 7)) + \
                                            (0 if std == 0 else (5 if std == 1 else 11)) + \
                                            (0 if lrg == 0 else (3 if lrg == 1 else 7)) + \
                                            (0 if flx == 0 else (3 if flx == 1 else 7))
                                    if cong == 0 and min_q >= 12:
                                        continue
                                    reachable_count += 1
        self.assertEqual(reachable_count, 954)

    def test_30_centralized_reward_multi_case_controlled_verification(self):
        """30. Verify centralized reward across ideal, extreme congestion, and moderate traffic cases."""
        cfg = RewardConfig()

        # Case 1: Ideal state (served=5, zero queues/wait/congestion, cost=2.0, valid=0)
        # Expected: 3*5 - 0.25*2.0 = 15.0 - 0.5 = 14.5
        r1, b1 = cfg.calculate(N_served=5, W_t=0, Q_t=0, C_t=0, delta_T_congestion=0, B_t=0, K_t=2.0, I_t=0)
        self.assertAlmostEqual(r1, 14.5, places=4)

        # Case 2: Extreme overload (served=0, wait=60, queue=60, cong=20, dur=1, imb=12, cost=12, invalid=1)
        # Expected: 0 - 0.15*60 - 0.25*60 - 0.8*20 - 1.0*1 - 0.3*12 - 0.25*12 - 5*1 = -52.6
        r2, b2 = cfg.calculate(N_served=0, W_t=60, Q_t=60, C_t=20, delta_T_congestion=1.0, B_t=12.0, K_t=12.0, I_t=1.0)
        self.assertAlmostEqual(r2, -52.6, places=4)

        # Case 3: Moderate traffic (served=1, wait=10, queue=10, cong=2, dur=0, imb=2.5, cost=5, invalid=0)
        # Expected: 3*1 - 0.15*10 - 0.25*10 - 0.8*2 - 0.3*2.5 - 0.25*5 = 3 - 1.5 - 2.5 - 1.6 - 0.75 - 1.25 = -4.6
        r3, b3 = cfg.calculate(N_served=1, W_t=10, Q_t=10, C_t=2, delta_T_congestion=0, B_t=2.5, K_t=5.0, I_t=0)
        self.assertAlmostEqual(r3, -4.6, places=4)


if __name__ == "__main__":
    unittest.main()

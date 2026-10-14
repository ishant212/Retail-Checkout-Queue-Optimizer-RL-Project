import sys
import os
import unittest
import numpy as np

# Ensure backend directory is in python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from simulation_engine import RetailQueueEnv, SimulationService, RewardConfig, Lane
from models import (
    AVAILABLE_MODELS,
    BaseQueueAgent,
    QLearningAgent,
    SarsaAgent,
    ContextualBanditAgent,
    ReinforceAgent,
    A2CAgent,
    PPOAgent,
    DQNAgent,
    DoubleDQNAgent,
    DDPGAgent
)
from models.base import QueueState
from app import app


class TestRLModelsArchitecture10Lanes(unittest.TestCase):
    """Unit tests for each of the 9 Reinforcement Learning model architectures on 10 lanes."""

    def setUp(self):
        self.num_counters = 10
        self.max_queue_len = 6
        self.num_actions = 41
        self.feature_dim = 84
        self.state_test_samples = [
            QueueState([0] * 10),
            QueueState([1, 2, 0, 3, 1, 0, 4, 2, 1, 0]),
            QueueState([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        ]

    def test_01_model_registry_contains_all_9_models(self):
        expected_keys = [
            "q_learning", "sarsa", "bandit", "reinforce",
            "a2c", "ppo", "dqn", "double_dqn", "ddpg"
        ]
        for key in expected_keys:
            self.assertIn(key, AVAILABLE_MODELS, f"Model {key} missing in AVAILABLE_MODELS registry")
            self.assertIn("name", AVAILABLE_MODELS[key])
            self.assertIn("family", AVAILABLE_MODELS[key])
            self.assertIn("tag", AVAILABLE_MODELS[key])

    def _test_generic_agent(self, agent, agent_name):
        # 1. Action selection within valid bounds [0..40]
        for s in self.state_test_samples:
            action = agent.choose_action(s, explore=False)
            self.assertIn(action, range(self.num_actions), f"{agent_name}: Action {action} out of bounds [0..40] for state {s}")

        # 2. Action values return exactly 41 finite values
        for s in self.state_test_samples:
            vals = agent.get_action_values(s)
            self.assertEqual(len(vals), 41, f"{agent_name}: Action values must have length 41")
            for v in vals:
                self.assertFalse(np.isnan(v), f"{agent_name}: NaN action value encountered in state {s}")
                self.assertFalse(np.isinf(v), f"{agent_name}: Inf action value encountered in state {s}")

        # 3. Policy decision matrix generates grid
        matrix = agent.get_policy_matrix(fixed_lane2=0)
        self.assertEqual(len(matrix), self.max_queue_len + 1, f"{agent_name}: Policy matrix row count mismatch")
        for row in matrix:
            self.assertEqual(len(row), self.max_queue_len + 1, f"{agent_name}: Policy matrix col count mismatch")
            for cell in row:
                self.assertIn(cell, range(self.num_actions), f"{agent_name}: Action in policy matrix must be in [0..40]")

        # 4. State feature vector dimension must be 84 across all 10 states
        for s in self.state_test_samples:
            feat = agent.state_features(s)
            self.assertEqual(len(feat), self.feature_dim, f"{agent_name}: Feature dimension must be 84")
            self.assertFalse(np.any(np.isnan(feat)), f"{agent_name}: NaN in feature vector")

    def test_02_tabular_q_learning(self):
        agent = QLearningAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10))
        self._test_generic_agent(agent, "Q-Learning")

    def test_03_sarsa(self):
        agent = SarsaAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10), 1)
        self._test_generic_agent(agent, "SARSA")

    def test_04_contextual_bandit(self):
        agent = ContextualBanditAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0)
        self._test_generic_agent(agent, "Contextual Bandit (UCB1)")

    def test_05_reinforce_policy_gradient(self):
        agent = ReinforceAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.store_transition(QueueState([1] * 10), 0, -1.0)
        agent.update_episode()
        self._test_generic_agent(agent, "REINFORCE")

    def test_06_advantage_actor_critic(self):
        agent = A2CAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10))
        self._test_generic_agent(agent, "A2C")

    def test_07_proximal_policy_optimization(self):
        agent = PPOAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10))
        agent.optimize_batch()
        self._test_generic_agent(agent, "PPO")

    def test_08_deep_q_network(self):
        agent = DQNAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10))
        self._test_generic_agent(agent, "DQN")

    def test_09_double_dqn_proposed(self):
        agent = DoubleDQNAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10))
        self._test_generic_agent(agent, "Double DQN (Proposed)")

    def test_10_deep_deterministic_policy_gradient(self):
        agent = DDPGAgent(self.num_counters, self.max_queue_len, num_actions=self.num_actions)
        agent.update(QueueState([1] * 10), 0, -1.0, QueueState([1] * 10))
        self._test_generic_agent(agent, "DDPG")


class TestSimulationEngineCore10Lanes(unittest.TestCase):
    """Integration tests for 10-lane SimulationService and Environment dynamics."""

    def test_11_environment_10_lanes_initialization(self):
        env = RetailQueueEnv()
        state = env.reset()
        self.assertEqual(len(env.lanes), 10, "Environment must have exactly 10 lanes")
        self.assertEqual(len(state.queue_lengths), 10, "Queue lengths must have length 10")
        self.assertEqual(len(state.open_status), 10, "Open status must have length 10")
        self.assertEqual(len(state.lane_types), 10, "Lane types must have length 10")
        self.assertEqual(len(state.capacities), 10, "Capacities must have length 10")

    def test_12_express_and_standard_lane_validation(self):
        env = RetailQueueEnv()
        env.reset()

        # Lane 0 is express (max 5 items)
        env.lanes[0].is_open = True
        env.lanes[0].queue.clear()

        # Basket size 3 is valid on Express Lane 0
        env.incoming_basket_size = 3
        valid, penalty, reason = env.validate_action(0)
        self.assertTrue(valid, "Basket <= 5 should be accepted by Express Lane 0")
        self.assertEqual(penalty, 0.0)

        # Basket size 12 should be rejected on Express Lane 0
        env.incoming_basket_size = 12
        valid, penalty, reason = env.validate_action(0)
        self.assertFalse(valid, "Basket > 5 should be rejected by Express Lane 0")
        self.assertEqual(penalty, 1.0)

        # Basket size 12 should be accepted by Standard Lane 2
        env.lanes[2].is_open = True
        env.lanes[2].queue.clear()
        valid, penalty, reason = env.validate_action(2)
        self.assertTrue(valid, "Basket <= 15 should be accepted by Standard Lane 2")

    def test_13_lane_capacity_limit_enforcement(self):
        env = RetailQueueEnv()
        env.reset()

        lane = env.lanes[2]
        lane.is_open = True
        lane.queue = [{"id": i, "basket": 5, "arr_t": 0} for i in range(lane.capacity)]
        env.incoming_basket_size = 5

        valid, penalty, reason = env.validate_action(2)
        self.assertFalse(valid, "Routing to full lane must be rejected")
        self.assertEqual(penalty, 1.0)
        self.assertIn("full", reason.lower())

    def test_14_safe_lane_closing_invariant(self):
        env = RetailQueueEnv()
        env.reset()

        # Lane 4 is closed initially
        env.lanes[4].is_open = False
        valid, penalty, reason = env.validate_action(24)  # Action 20 + 4 = 24 (close lane 4)
        self.assertFalse(valid, "Closing an already closed lane must be rejected")

        # Open Lane 4 and add a customer
        env.lanes[4].is_open = True
        env.lanes[4].queue = [{"id": 99, "basket": 5, "arr_t": 0}]
        valid, penalty, reason = env.validate_action(24)
        self.assertFalse(valid, "Closing a lane with waiting customers must be rejected to prevent stranding")
        self.assertEqual(penalty, 1.0)

        # Clear queue -> closing is now safe
        env.lanes[4].queue.clear()
        env.lanes[4].current_serving = None
        valid, penalty, reason = env.validate_action(24)
        self.assertTrue(valid, "Closing empty idle lane must be permitted")
        self.assertEqual(penalty, 0.0)

    def test_15_lane_type_conversion_invariant(self):
        env = RetailQueueEnv()
        env.reset()

        # Action 38: Convert Lane 8 (Flexible)
        lane8 = env.lanes[8]
        lane8.queue.clear()
        lane8.current_serving = None
        orig_type = lane8.lane_type

        # Conversion of empty idle lane should succeed
        valid, penalty, reason = env.validate_action(38)
        self.assertTrue(valid, "Converting empty idle lane should succeed")

        # Step conversion
        next_s, r = env.step(38)
        self.assertNotEqual(lane8.lane_type, orig_type, "Lane type should cycle")

        # Conversion with customers in queue should be rejected
        lane8.queue = [{"id": 1, "basket": 5, "arr_t": 0}]
        valid, penalty, reason = env.validate_action(38)
        self.assertFalse(valid, "Converting lane with queued customers must be rejected")
        self.assertEqual(penalty, 1.0)

    def test_16_hold_action_is_always_valid(self):
        env = RetailQueueEnv()
        env.reset()
        valid, penalty, reason = env.validate_action(40)
        self.assertTrue(valid, "Hold action (40) must always be valid")
        self.assertEqual(penalty, 0.0)

        next_s, r = env.step(40)
        self.assertTrue(isinstance(r, float))

    def test_17_centralized_reward_calculation(self):
        rc = RewardConfig(alpha=3.0, beta=0.15, gamma=0.25, delta=0.8, rho=1.0, zeta=0.3, kappa=0.25, xi=5.0)
        r, bd = rc.calculate(
            N_served=2,
            W_t=10.0,
            Q_t=10,
            C_t=2,
            delta_T_congestion=1.0,
            B_t=1.5,
            K_t=8.0,
            I_t=0.0
        )
        # 3.0*2 - 0.15*10 - 0.25*10 - 0.8*2 - 1.0*1 - 0.3*1.5 - 0.25*8 - 0
        expected = 6.0 - 1.5 - 2.5 - 1.6 - 1.0 - 0.45 - 2.0
        self.assertAlmostEqual(r, expected, places=2)
        self.assertIn("N_served", bd)
        self.assertIn("total_reward", bd)

    def test_18_simulation_service_100_step_benchmark(self):
        sim = SimulationService()
        bench = sim.run_benchmark(steps=20)
        self.assertIn("leaderboard", bench)
        self.assertIn("randomTotalWait", bench)
        self.assertEqual(len(bench["leaderboard"]), 10)  # 9 RL models + Shortest Queue
        for m in bench["leaderboard"]:
            self.assertIn("name", m)
            self.assertIn("totalWait", m)
            self.assertIn("improvement", m)

    def test_19_interactive_step_and_semantic_actions(self):
        sim = SimulationService()
        reset_res = sim.reset_interactive()
        self.assertEqual(len(reset_res["state"]), 10)
        self.assertIn("laneTypes", reset_res)
        self.assertIn("capacities", reset_res)

        step_res = sim.step_interactive(action=40)
        self.assertEqual(step_res["action"], 40)
        self.assertEqual(step_res["actionSemantic"]["type"], "hold")
        self.assertTrue(step_res["actionValid"])

    def test_20_comparative_analysis_table_iv(self):
        sim = SimulationService()
        res = sim.run_comparative_analysis("normal")
        self.assertIn("tableIV", res)
        self.assertGreaterEqual(len(res["tableIV"]), 10)
        row0 = res["tableIV"][0]
        self.assertIn("strategy", row0)
        self.assertIn("avgWaitTime", row0)
        self.assertIn("hourlyStaffCost", row0)


class TestFlaskAPI10Lanes(unittest.TestCase):
    """End-to-end integration tests for Flask REST API endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_21_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["numCounters"], 10)
        self.assertIn("modelEpisodes", data)
        self.assertEqual(len(data["models"]), 9)

    def test_22_api_models_and_select(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("models", data)

        sel_res = self.client.post("/api/models/select", json={"modelId": "double_dqn"})
        self.assertEqual(sel_res.status_code, 200)
        sel_data = sel_res.get_json()
        self.assertTrue(sel_data["success"])
        self.assertEqual(sel_data["activeModelId"], "double_dqn")

    def test_23_api_interactive_step_and_reset(self):
        res_rst = self.client.post("/api/interactive/reset", json={"modelId": "q_learning"})
        self.assertEqual(res_rst.status_code, 200)
        rst_data = res_rst.get_json()
        self.assertEqual(len(rst_data["state"]), 10)

        res_stp = self.client.post("/api/interactive/step", json={"action": 40})
        self.assertEqual(res_stp.status_code, 200)
        stp_data = res_stp.get_json()
        self.assertEqual(len(stp_data["state"]), 10)
        self.assertEqual(stp_data["action"], 40)
        self.assertEqual(stp_data["actionSemantic"]["type"], "hold")

    def test_24_api_lanes_get_and_update(self):
        res = self.client.get("/api/lanes")
        self.assertEqual(res.status_code, 200)
        lanes = res.get_json()
        self.assertEqual(len(lanes), 10)
        self.assertEqual(lanes[0]["id"], 0)

        # Update lane 8 capacity and service rate
        up_res = self.client.post("/api/lanes/update", json={
            "laneId": 8,
            "capacity": 8,
            "serviceRate": 0.65
        })
        self.assertEqual(up_res.status_code, 200)
        up_data = up_res.get_json()
        self.assertTrue(up_data["success"])

    def test_25_api_simpy_run(self):
        res = self.client.post("/api/simpy/run", json={"arrivalRate": 1.0, "simulationTime": 30.0})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("summary", data)
        self.assertIn("bestPolicy", data)
        self.assertGreater(len(data["summary"]), 0)


if __name__ == "__main__":
    unittest.main()

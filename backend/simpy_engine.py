import simpy
import random
import numpy as np
from models.base import QueueState

POLICY_METADATA = {
    "random": {
        "label": "Baseline (Random Routing)",
        "tag": "Zero Intelligence",
        "color": "#94a3b8"
    },
    "shortest_queue": {
        "label": "Heuristic (Shortest Queue)",
        "tag": "Greedy Heuristic",
        "color": "#3b82f6"
    },
    "q_learning": {
        "label": "Q-Learning (Off-Policy TD)",
        "tag": "1-Step TD Control",
        "color": "#10b981"
    },
    "sarsa": {
        "label": "SARSA (On-Policy TD)",
        "tag": "Safe On-Policy TD",
        "color": "#06b6d4"
    },
    "bandit": {
        "label": "Multi-Armed Bandit (UCB1)",
        "tag": "Adaptive Arms",
        "color": "#f59e0b"
    },
    "reinforce": {
        "label": "REINFORCE (w/ Baseline)",
        "tag": "Policy Gradient",
        "color": "#ec4899"
    },
    "a2c": {
        "label": "A2C (Actor-Critic)",
        "tag": "Advantage Actor-Critic",
        "color": "#8b5cf6"
    },
    "ppo": {
        "label": "PPO (Proximal Policy Opt)",
        "tag": "Clipped Surrogate",
        "color": "#6366f1"
    },
    "dqn": {
        "label": "DQN (Deep Q-Network)",
        "tag": "Experience Replay",
        "color": "#14b8a6"
    },
    "double_dqn": {
        "label": "Double DQN",
        "tag": "Decoupled Q-Target",
        "color": "#0ea5e9"
    },
    "ddpg": {
        "label": "DDPG (Deep Deterministic PG)",
        "tag": "Continuous Actor-Critic",
        "color": "#a855f7"
    }
}

DEFAULT_SIMPY_SERVICE_RATES = [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5]
DEFAULT_SIMPY_LANE_TYPES = [
    "express", "express", "standard", "standard", "standard",
    "standard", "large_basket", "large_basket", "flexible", "flexible"
]
DEFAULT_SIMPY_INITIAL_OPEN = [True, True, True, True, False, False, True, False, True, False]


class SimPyCheckoutSimulator:
    """
    10-Lane Continuous-Time Discrete-Event Simulation Arena powered by SimPy.
    Models stochastic customer arrivals, dynamic cashier servers, queue capacities,
    and basket size eligibility across all 10 lanes.
    """
    def __init__(self, service_rates=None, arrival_rate=1.0, simulation_time=120.0, seed=42):
        self.service_rates = list(service_rates) if service_rates else list(DEFAULT_SIMPY_SERVICE_RATES)
        # Pad to 10 if fewer provided
        while len(self.service_rates) < 10:
            self.service_rates.append(0.5)

        self.arrival_rate = float(arrival_rate)
        self.simulation_time = float(simulation_time)
        self.seed = int(seed)
        self.num_lanes = len(self.service_rates)

    def run_comparison(self, agents_dict=None, q_learning_agent=None, max_queue_len=6):
        all_agents = {}
        if isinstance(agents_dict, dict):
            all_agents.update(agents_dict)
        elif q_learning_agent is not None:
            all_agents["q_learning"] = q_learning_agent

        # Base policies
        policies = ["random", "shortest_queue"]

        # Add available RL models
        for mid in ["q_learning", "sarsa", "bandit", "reinforce", "a2c", "ppo", "dqn", "double_dqn", "ddpg"]:
            if mid in all_agents:
                policies.append(mid)

        results = {}
        for policy in policies:
            results[policy] = self._run_single_policy(policy, all_agents.get(policy), max_queue_len)

        # Baseline random average wait for reduction comparison
        rand_avg_wait = results.get("random", {}).get("avgWaitTime", 1.0)
        rand_avg_wait = max(rand_avg_wait, 0.001)

        summary = []
        for policy in policies:
            res = results[policy]
            meta = POLICY_METADATA.get(policy, {
                "label": policy.title(),
                "tag": "Policy",
                "color": "#64748b"
            })
            avg_wait = res["avgWaitTime"]
            reduction = ((rand_avg_wait - avg_wait) / rand_avg_wait) * 100

            summary.append({
                "policy": policy,
                "label": meta["label"],
                "tag": meta["tag"],
                "color": meta["color"],
                "customersServed": res["totalServed"],
                "totalCustomers": res["totalCustomers"],
                "avgWaitTime": round(avg_wait, 2),
                "maxWaitTime": round(res["maxWaitTime"], 2),
                "avgTotalQueue": round(res["avgTotalQueue"], 2),
                "waitReductionPct": round(reduction, 2),
                "utilization": [round(u * 100, 1) for u in res["laneUtilization"]],
                "lanesOpened": res.get("lanesOpened", 0),
                "lanesClosed": res.get("lanesClosed", 0),
                "lanesConverted": res.get("lanesConverted", 0)
            })

        summary.sort(key=lambda x: x["avgWaitTime"])
        best_policy = summary[0]["label"] if summary else "N/A"

        return {
            "simulationTime": self.simulation_time,
            "arrivalRate": self.arrival_rate,
            "serviceRates": self.service_rates,
            "numLanes": self.num_lanes,
            "summary": summary,
            "bestPolicy": best_policy,
            "details": results
        }

    def _run_single_policy(self, policy_name, agent, max_queue_len):
        random.seed(self.seed)
        np.random.seed(self.seed)
        env = simpy.Environment()

        # 10 Cashier resources (1 cashier per lane)
        counters = [simpy.Resource(env, capacity=1) for _ in range(self.num_lanes)]
        queue_lengths = [0] * self.num_lanes
        busy_times = [0.0] * self.num_lanes
        wait_times = []

        open_status = list(DEFAULT_SIMPY_INITIAL_OPEN[:self.num_lanes])
        lane_types = list(DEFAULT_SIMPY_LANE_TYPES[:self.num_lanes])
        capacities = [max_queue_len] * self.num_lanes

        total_customers = 0
        total_served = 0
        opened_cnt = 0
        closed_cnt = 0
        converted_cnt = 0

        sampled_history = []

        def sample_process():
            while True:
                sampled_history.append({
                    "time": round(float(env.now), 1),
                    "lanes": list(queue_lengths),
                    "totalQueue": sum(queue_lengths),
                    "openLanes": sum(1 for o in open_status if o)
                })
                yield env.timeout(3.0)

        env.process(sample_process())

        def customer_generator():
            nonlocal total_customers, opened_cnt, closed_cnt, converted_cnt
            while True:
                inter_arrival = random.expovariate(self.arrival_rate)
                yield env.timeout(inter_arrival)
                total_customers += 1
                cust_id = total_customers

                basket_size = int(np.random.randint(1, 21))
                current_state = QueueState(
                    queues=list(queue_lengths),
                    open_status=list(open_status),
                    lane_types=list(lane_types),
                    capacities=list(capacities),
                    waiting_time={
                        "avg": float(np.mean(wait_times)) if wait_times else 0.0,
                        "max": float(np.max(wait_times)) if wait_times else 0.0,
                        "step": float(sum(queue_lengths))
                    },
                    service_rates=list(self.service_rates),
                    counter_utilization=[float(queue_lengths[i] > 0) for i in range(self.num_lanes)],
                    arrival_rate=float(self.arrival_rate),
                    incoming_basket_size=int(basket_size),
                    sim_time=int(env.now),
                    congestion_severity=max(0, sum(queue_lengths) - 10),
                    congestion_duration={"step": 0.0, "continuous": 0.0, "total": 0.0},
                    operating_cost=float(sum(1.2 for o in open_status if o)),
                    queue_imbalance=float(np.var(queue_lengths)) if len(queue_lengths) > 1 else 0.0
                )

                # Route action selection
                if policy_name == "random":
                    open_eligible = [
                        i for i in range(self.num_lanes)
                        if open_status[i] and queue_lengths[i] < capacities[i]
                        and not (lane_types[i] == "express" and basket_size > 5)
                    ]
                    chosen_lane = random.choice(open_eligible) if open_eligible else int(np.argmin(queue_lengths))

                elif policy_name == "shortest_queue":
                    open_eligible = [
                        i for i in range(self.num_lanes)
                        if open_status[i] and queue_lengths[i] < capacities[i]
                        and not (lane_types[i] == "express" and basket_size > 5)
                    ]
                    chosen_lane = min(open_eligible, key=lambda i: queue_lengths[i]) if open_eligible else int(np.argmin(queue_lengths))

                elif agent is not None and hasattr(agent, "choose_action"):
                    raw_act = int(agent.choose_action(current_state, explore=False))

                    # 1. Routing action [0..9]
                    if 0 <= raw_act < self.num_lanes:
                        if (open_status[raw_act] and queue_lengths[raw_act] < capacities[raw_act]
                                and not (lane_types[raw_act] == "express" and basket_size > 5)):
                            chosen_lane = raw_act
                        else:
                            # Fallback to shortest open eligible
                            open_eligible = [
                                i for i in range(self.num_lanes)
                                if open_status[i] and queue_lengths[i] < capacities[i]
                                and not (lane_types[i] == "express" and basket_size > 5)
                            ]
                            chosen_lane = min(open_eligible, key=lambda i: queue_lengths[i]) if open_eligible else int(np.argmin(queue_lengths))

                    # 2. Lane opening action [10..19]
                    elif 10 <= raw_act < 20:
                        l_idx = raw_act - 10
                        if l_idx < self.num_lanes and not open_status[l_idx]:
                            open_status[l_idx] = True
                            opened_cnt += 1
                        open_eligible = [i for i in range(self.num_lanes) if open_status[i]]
                        chosen_lane = min(open_eligible, key=lambda i: queue_lengths[i]) if open_eligible else 0

                    # 3. Lane closing action [20..29]
                    elif 20 <= raw_act < 30:
                        l_idx = raw_act - 20
                        if l_idx < self.num_lanes and open_status[l_idx] and queue_lengths[l_idx] == 0:
                            if sum(1 for o in open_status if o) > 1:
                                open_status[l_idx] = False
                                closed_cnt += 1
                        open_eligible = [i for i in range(self.num_lanes) if open_status[i]]
                        chosen_lane = min(open_eligible, key=lambda i: queue_lengths[i]) if open_eligible else 0

                    # 4. Lane conversion action [30..39]
                    elif 30 <= raw_act < 40:
                        l_idx = raw_act - 30
                        if l_idx < self.num_lanes and queue_lengths[l_idx] == 0:
                            order = ["express", "standard", "large_basket", "flexible"]
                            cur_t = lane_types[l_idx]
                            idx = order.index(cur_t) if cur_t in order else 1
                            lane_types[l_idx] = order[(idx + 1) % len(order)]
                            converted_cnt += 1
                        open_eligible = [i for i in range(self.num_lanes) if open_status[i]]
                        chosen_lane = min(open_eligible, key=lambda i: queue_lengths[i]) if open_eligible else 0

                    # 5. Hold action [40]
                    else:
                        open_eligible = [i for i in range(self.num_lanes) if open_status[i]]
                        chosen_lane = min(open_eligible, key=lambda i: queue_lengths[i]) if open_eligible else 0

                else:
                    chosen_lane = int(np.argmin(queue_lengths))

                chosen_lane = max(0, min(chosen_lane, self.num_lanes - 1))
                env.process(customer_service(cust_id, chosen_lane, basket_size))

        def customer_service(cid, lane_idx, basket_size):
            nonlocal total_served
            arrival_time = env.now
            queue_lengths[lane_idx] += 1

            counter = counters[lane_idx]
            with counter.request() as req:
                yield req
                service_start_time = env.now
                wait_time = service_start_time - arrival_time
                wait_times.append(wait_time)

                # Service duration proportional to basket size and service rate
                base_rate = self.service_rates[lane_idx]
                effective_rate = base_rate * (10.0 / max(5.0, float(basket_size)))
                service_duration = random.expovariate(max(0.1, effective_rate))
                yield env.timeout(service_duration)

                busy_times[lane_idx] += service_duration
                queue_lengths[lane_idx] -= 1
                total_served += 1

        env.process(customer_generator())
        env.run(until=self.simulation_time)

        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
        max_wait = float(np.max(wait_times)) if wait_times else 0.0
        avg_total_q = float(np.mean([s["totalQueue"] for s in sampled_history])) if sampled_history else 0.0
        utilization = [min(1.0, busy / self.simulation_time) for busy in busy_times]

        return {
            "policy": policy_name,
            "totalCustomers": total_customers,
            "totalServed": total_served,
            "avgWaitTime": avg_wait,
            "maxWaitTime": max_wait,
            "avgTotalQueue": avg_total_q,
            "laneUtilization": utilization,
            "queueHistory": sampled_history[:40],
            "lanesOpened": opened_cnt,
            "lanesClosed": closed_cnt,
            "lanesConverted": converted_cnt
        }

import numpy as np

# Supported Lane Types
LANE_TYPES = ["express", "standard", "large_basket", "flexible"]
LANE_TYPE_MAP = {
    "express": 0,
    "standard": 1,
    "large_basket": 2,
    "flexible": 3
}

DEFAULT_10_LANE_TYPES = [
    "express",       # Lane 0
    "express",       # Lane 1
    "standard",      # Lane 2
    "standard",      # Lane 3
    "standard",      # Lane 4
    "standard",      # Lane 5
    "large_basket",  # Lane 6
    "large_basket",  # Lane 7
    "flexible",      # Lane 8 (configured as standard initially)
    "flexible"       # Lane 9 (configured as flexible initially)
]

DEFAULT_10_SERVICE_RATES = [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5]
DEFAULT_10_CAPACITIES = [6, 6, 8, 8, 8, 8, 8, 8, 8, 6]
DEFAULT_10_OPERATING_COSTS = [1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.2, 1.2]
DEFAULT_10_INITIAL_OPEN = [True, True, True, True, True, True, True, True, True, True]


TABULAR_FEATURE_DIM = 35
CONTINUOUS_FEATURE_DIM = 84
ACTION_SPACE_SIZE = 41

TABULAR_FEATURE_NAMES = [
    f"lane_{i}_queue_norm" for i in range(10)
] + [
    f"lane_{i}_open" for i in range(10)
] + [
    f"lane_{i}_type" for i in range(10)
] + [
    "incoming_basket_size_norm",
    "customer_arrival_rate_norm",
    "sim_time_epoch",
    "congestion_severity_norm",
    "congestion_duration_norm"
]

CONTINUOUS_FEATURE_NAMES = [
    f"lane_{i}_queue_norm" for i in range(10)
] + [
    f"lane_{i}_open" for i in range(10)
] + [
    f"lane_{i}_type" for i in range(10)
] + [
    f"lane_{i}_remaining_capacity" for i in range(10)
] + [
    f"lane_{i}_service_rate" for i in range(10)
] + [
    f"lane_{i}_utilization" for i in range(10)
] + [
    f"lane_{i}_workload" for i in range(10)
] + [
    "waiting_time_avg", "waiting_time_max", "waiting_time_step",
    "basket_size_norm", "is_express_eligible",
    "arrival_rate_norm", "sim_time_epoch",
    "congestion_severity_norm",
    "congestion_duration_step", "congestion_duration_continuous",
    "total_waiting_customers_norm", "total_serving_customers_norm",
    "operating_cost_norm", "queue_imbalance_norm"
]


class QueueState(tuple):
    """
    Rich state container representing all 10 environment states across 10 checkout lanes.
    Maintains backward compatibility by subclassing tuple (unpacking as (q0, q1, ..., q9)
    or legacy q0, q1, q2 = state[:3]).

    Provides dedicated methods for:
      - 35-feature tabular observation representation (.to_tabular_features())
      - 84-feature continuous neural observation representation (.to_continuous_features())
    """
    def __new__(cls, queues, open_status=None, lane_types=None, capacities=None,
                waiting_time=None, service_rates=None, counter_utilization=None,
                estimated_workloads=None, current_serving=None, arrival_rate=1.0,
                incoming_basket_size=5, sim_time=0, congestion_severity=0,
                congestion_duration=None, operating_cost=5.0, queue_imbalance=0.0):
        queues_tuple = tuple(int(x) for x in queues)
        obj = super().__new__(cls, queues_tuple)
        num_c = len(queues_tuple)

        obj.queue_lengths = list(queues_tuple)
        obj.open_status = list(open_status) if open_status is not None else [1] * num_c
        obj.lane_types = list(lane_types) if lane_types is not None else list(DEFAULT_10_LANE_TYPES[:num_c])
        obj.capacities = list(capacities) if capacities is not None else [6] * num_c
        obj.remaining_capacities = [max(0, cap - q) for cap, q in zip(obj.capacities, obj.queue_lengths)]
        obj.waiting_time = waiting_time or {"avg": 0.0, "max": 0.0, "step": 0.0}
        obj.service_rates = list(service_rates) if service_rates is not None else list(DEFAULT_10_SERVICE_RATES[:num_c])
        obj.counter_utilization = list(counter_utilization) if counter_utilization is not None else [0.0] * num_c
        obj.estimated_workloads = list(estimated_workloads) if estimated_workloads is not None else [
            round(float(q) / max(0.01, float(r)), 2) for q, r in zip(obj.queue_lengths, obj.service_rates)
        ]
        obj.current_serving = list(current_serving) if current_serving is not None else [None] * num_c
        obj.arrival_rate = float(arrival_rate)
        obj.incoming_basket_size = int(incoming_basket_size)
        obj.sim_time = int(sim_time)
        obj.congestion_severity = int(congestion_severity)
        obj.congestion_duration = congestion_duration or {"step": 0.0, "continuous": 0.0, "total": 0.0}
        obj.operating_cost = float(operating_cost)
        obj.queue_imbalance = float(queue_imbalance)

        # Derived high-level totals
        obj.total_waiting_customers = int(sum(obj.queue_lengths))
        obj.total_serving_customers = int(sum(1 for s in obj.current_serving if s is not None))
        return obj

    def to_tabular_features(self):
        """Produces documented 35-feature representation for tabular models."""
        feat = np.zeros(TABULAR_FEATURE_DIM, dtype=np.float32)
        for i in range(10):
            cap = max(1.0, float(self.capacities[i]))
            feat[i] = max(0.0, min(float(self.queue_lengths[i]), cap)) / cap
        for i in range(10):
            feat[10 + i] = 1.0 if self.open_status[i] else 0.0
        for i in range(10):
            t_str = str(self.lane_types[i]).lower()
            feat[20 + i] = float(LANE_TYPE_MAP.get(t_str, 1)) / 3.0
        feat[30] = min(1.0, float(self.incoming_basket_size) / 30.0)
        feat[31] = min(1.0, float(self.arrival_rate) / 5.0)
        feat[32] = (float(self.sim_time) % 100.0) / 100.0
        feat[33] = min(1.0, float(self.congestion_severity) / 30.0)
        cont_cd = self.congestion_duration.get("continuous", 0.0) if isinstance(self.congestion_duration, dict) else 0.0
        feat[34] = min(1.0, float(cont_cd) / 50.0)
        return feat

    def to_continuous_features(self):
        """Produces documented 84-feature continuous normalized vector for neural models."""
        feat = np.zeros(CONTINUOUS_FEATURE_DIM, dtype=np.float32)
        for i in range(10):
            cap = max(1.0, float(self.capacities[i]))
            feat[i] = max(0.0, min(float(self.queue_lengths[i]), cap)) / cap
        for i in range(10):
            feat[10 + i] = 1.0 if self.open_status[i] else 0.0
        for i in range(10):
            t_str = str(self.lane_types[i]).lower()
            feat[20 + i] = float(LANE_TYPE_MAP.get(t_str, 1)) / 3.0
        for i in range(10):
            rem = max(0, self.capacities[i] - self.queue_lengths[i])
            feat[30 + i] = float(rem) / max(1.0, float(self.capacities[i]))
        for i in range(10):
            feat[40 + i] = float(self.service_rates[i])
        for i in range(10):
            feat[50 + i] = max(0.0, min(1.0, float(self.counter_utilization[i])))
        for i in range(10):
            feat[60 + i] = min(1.0, float(self.estimated_workloads[i]) / 20.0)
        wt = self.waiting_time if isinstance(self.waiting_time, dict) else {}
        feat[70] = min(1.0, float(wt.get("avg", 0.0)) / 30.0)
        feat[71] = min(1.0, float(wt.get("max", 0.0)) / 60.0)
        feat[72] = min(1.0, float(wt.get("step", 0.0)) / 60.0)
        feat[73] = min(1.0, float(self.incoming_basket_size) / 30.0)
        feat[74] = 1.0 if self.incoming_basket_size <= 5 else 0.0
        feat[75] = min(1.0, float(self.arrival_rate) / 5.0)
        feat[76] = (float(self.sim_time) % 100.0) / 100.0
        feat[77] = min(1.0, float(self.congestion_severity) / 30.0)
        cd = self.congestion_duration if isinstance(self.congestion_duration, dict) else {}
        feat[78] = 1.0 if cd.get("step", 0.0) > 0 else 0.0
        feat[79] = min(1.0, float(cd.get("continuous", 0.0)) / 50.0)
        feat[80] = min(1.0, float(self.total_waiting_customers) / 60.0)
        feat[81] = min(1.0, float(self.total_serving_customers) / 10.0)
        feat[82] = min(1.0, float(self.operating_cost) / 20.0)
        feat[83] = min(1.0, float(self.queue_imbalance) / 10.0)
        return feat

    def to_dict(self):
        return {
            "queue_lengths": self.queue_lengths,
            "open_status": [bool(x) for x in self.open_status],
            "lane_types": self.lane_types,
            "capacities": self.capacities,
            "remaining_capacities": self.remaining_capacities,
            "waiting_time": self.waiting_time,
            "service_rates": self.service_rates,
            "counter_utilization": [round(float(u), 3) for u in self.counter_utilization],
            "estimated_workloads": self.estimated_workloads,
            "current_serving": self.current_serving,
            "customer_arrival_rate": round(float(self.arrival_rate), 2),
            "incoming_basket_size": self.incoming_basket_size,
            "is_express_eligible": bool(self.incoming_basket_size <= 5),
            "current_sim_time": self.sim_time,
            "congestion_severity": self.congestion_severity,
            "congestion_duration": self.congestion_duration,
            "total_waiting_customers": self.total_waiting_customers,
            "total_serving_customers": self.total_serving_customers,
            "operating_cost": round(float(self.operating_cost), 2),
            "queue_imbalance": round(float(self.queue_imbalance), 3)
        }


class BaseQueueAgent:
    """
    Abstract Base Class for Retail Checkout Queue RL Agents.
    Supports 10-Lane environment with 84-dimensional continuous normalized feature vector
    and 41 Discrete Actions:
      - Actions 0..9:   Route customer to Lane 0..9
      - Actions 10..19: Open Lane 0..9
      - Actions 20..29: Close Lane 0..9 safely (requires queue == 0 and idle)
      - Actions 30..39: Convert Lane 0..9 type (cycles Express -> Standard -> Large -> Flexible)
      - Action 40:      Hold / No Action (Cashiers continue servicing queues)
    """
    def __init__(self, name, model_type, num_counters=10, max_queue_len=6, num_actions=41):
        self.name = name
        self.model_type = model_type  # 'value_based', 'bandit', 'policy_gradient', 'actor_critic', 'deep_rl'
        self.num_counters = num_counters
        self.max_queue_len = max_queue_len
        self.num_actions = num_actions
        self.feature_dim = 84
        self.state_shape = tuple([max_queue_len + 1] * min(num_counters, 3))

    @staticmethod
    def action_to_semantic(action):
        """Translates integer action [0..40] into a structured dictionary."""
        a = int(action)
        if 0 <= a < 10:
            return {"type": "route", "lane_id": a, "label": f"Route to Lane {a}"}
        elif 10 <= a < 20:
            return {"type": "open", "lane_id": a - 10, "label": f"Open Lane {a - 10}"}
        elif 20 <= a < 30:
            return {"type": "close", "lane_id": a - 20, "label": f"Close Lane {a - 20}"}
        elif 30 <= a < 40:
            return {"type": "convert", "lane_id": a - 30, "label": f"Convert Lane {a - 30}"}
        elif a == 40:
            return {"type": "hold", "lane_id": None, "label": "Hold / No Action"}
        return {"type": "unknown", "lane_id": None, "label": f"Unknown Action {a}"}

    def tabular_features(self, state):
        """
        Validates and returns the exact 35-feature representation for tabular models.
        Strictly rejects invalid dimensions with a descriptive ValueError.
        """
        if isinstance(state, QueueState):
            return state.to_tabular_features()
        if isinstance(state, np.ndarray):
            if state.size == TABULAR_FEATURE_DIM:
                return state.astype(np.float32).reshape(TABULAR_FEATURE_DIM)
            raise ValueError(f"Invalid tabular feature dimension: received {state.size}, expected exactly {TABULAR_FEATURE_DIM}")
        if isinstance(state, (list, tuple)):
            if len(state) == TABULAR_FEATURE_DIM:
                return np.array(state, dtype=np.float32)
            if len(state) == self.num_counters:
                # Wrap legacy queue lengths into QueueState
                return QueueState(state).to_tabular_features()
            raise ValueError(f"Invalid tabular feature dimension: received {len(state)}, expected exactly {TABULAR_FEATURE_DIM}")
        if isinstance(state, dict):
            qs = QueueState(
                queues=state.get("queue_lengths", [0] * self.num_counters),
                open_status=state.get("open_status", [1] * self.num_counters),
                lane_types=state.get("lane_types", DEFAULT_10_LANE_TYPES),
                incoming_basket_size=state.get("incoming_basket_size", 5),
                arrival_rate=state.get("customer_arrival_rate", 1.0),
                sim_time=state.get("current_sim_time", 0),
                congestion_severity=state.get("congestion_severity", 0),
                congestion_duration=state.get("congestion_duration", {"continuous": 0.0})
            )
            return qs.to_tabular_features()
        raise ValueError(f"Unsupported state type {type(state)} for tabular feature extraction.")

    def state_features(self, state):
        """
        Validates and returns the exact 84-feature representation for continuous/neural models.
        Strictly rejects invalid dimensions with a descriptive ValueError.
        """
        if isinstance(state, QueueState):
            return state.to_continuous_features()
        if isinstance(state, np.ndarray):
            if state.size == CONTINUOUS_FEATURE_DIM:
                return state.astype(np.float32).reshape(CONTINUOUS_FEATURE_DIM)
            raise ValueError(f"Invalid continuous feature dimension: received {state.size}, expected exactly {CONTINUOUS_FEATURE_DIM}")
        if isinstance(state, (list, tuple)):
            if len(state) == CONTINUOUS_FEATURE_DIM:
                return np.array(state, dtype=np.float32)
            if len(state) == self.num_counters:
                # Wrap legacy queue lengths into QueueState
                return QueueState(state).to_continuous_features()
            raise ValueError(f"Invalid continuous feature dimension: received {len(state)}, expected exactly {CONTINUOUS_FEATURE_DIM}")
        if isinstance(state, dict):
            qs = QueueState(
                queues=state.get("queue_lengths", [0] * self.num_counters),
                open_status=state.get("open_status", [1] * self.num_counters),
                lane_types=state.get("lane_types", DEFAULT_10_LANE_TYPES),
                capacities=state.get("capacities", DEFAULT_10_CAPACITIES),
                service_rates=state.get("service_rates", DEFAULT_10_SERVICE_RATES),
                counter_utilization=state.get("counter_utilization", [0.0] * self.num_counters),
                estimated_workloads=state.get("estimated_workloads", [0.0] * self.num_counters),
                waiting_time=state.get("waiting_time", {"avg": 0.0, "max": 0.0, "step": 0.0}),
                arrival_rate=state.get("customer_arrival_rate", 1.0),
                incoming_basket_size=state.get("incoming_basket_size", 5),
                sim_time=state.get("current_sim_time", 0),
                congestion_severity=state.get("congestion_severity", 0),
                congestion_duration=state.get("congestion_duration", {"step": 0.0, "continuous": 0.0}),
                operating_cost=state.get("operating_cost", 5.0),
                queue_imbalance=state.get("queue_imbalance", 0.0)
            )
            return qs.to_continuous_features()
        raise ValueError(f"Unsupported state type {type(state)} for continuous feature extraction.")

    def state_to_key(self, state):
        """
        Discretizes the 35 tabular features into a documented 7-tuple aggregate state key:
          (exp_cat, std_cat, lrg_cat, flx_cat, open_cat, basket_cat, is_congested)
        Total distinct discrete states: 3 * 3 * 3 * 3 * 3 * 3 * 2 = 1458 discrete states.
        """
        tab = self.tabular_features(state)
        # tab[0..9]: normalized queue lengths q_i / cap_i (cap_i = 6)
        # tab[10..19]: open statuses o_i in {0.0, 1.0}
        # tab[20..29]: lane types (Express=0.0, Standard=0.33, Large=0.67, Flexible=1.0)
        # tab[30]: incoming basket size normalized / 30.0
        # tab[33]: congestion severity / 30.0

        q_est = [int(round(tab[i] * 6.0)) for i in range(10)]
        ops = [bool(tab[10 + i] > 0.5) for i in range(10)]
        types = [tab[20 + i] for i in range(10)]
        basket_size = tab[30] * 30.0

        # 1. Express queue load
        exp_q = sum(q_est[i] for i in range(10) if types[i] < 0.15)
        exp_cat = 0 if exp_q < 3 else (1 if exp_q <= 6 else 2)

        # 2. Standard queue load
        std_q = sum(q_est[i] for i in range(10) if 0.15 <= types[i] < 0.5)
        std_cat = 0 if std_q < 5 else (1 if std_q <= 10 else 2)

        # 3. Large Basket queue load
        lrg_q = sum(q_est[i] for i in range(10) if 0.5 <= types[i] < 0.85)
        lrg_cat = 0 if lrg_q < 3 else (1 if lrg_q <= 6 else 2)

        # 4. Flexible queue load
        flx_q = sum(q_est[i] for i in range(10) if types[i] >= 0.85)
        flx_cat = 0 if flx_q < 3 else (1 if flx_q <= 6 else 2)

        # 5. Open count category
        open_count = sum(1 for o in ops if o)
        open_cat = 0 if open_count <= 4 else (1 if open_count <= 7 else 2)

        # 6. Basket category
        basket_cat = 0 if basket_size <= 5 else (1 if basket_size <= 15 else 2)

        # 7. Congestion category
        total_q = sum(q_est)
        is_congested = 1 if (total_q >= 12 or tab[33] * 30.0 > 0) else 0

        return (exp_cat, std_cat, lrg_cat, flx_cat, open_cat, basket_cat, is_congested)

    def choose_action(self, state, explore=True):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_action_values(self, state):
        """Returns 41 float values representing action preferences / Q-values."""
        raise NotImplementedError

    def get_action_probabilities(self, state):
        """Returns action distribution over the 41 discrete actions."""
        vals = np.array(self.get_action_values(state), dtype=np.float64)
        exp_v = np.exp(vals - np.max(vals))
        return (exp_v / np.sum(exp_v)).tolist()

    def get_policy_matrix(self, fixed_lane2=0):
        """Returns 6x6 matrix of optimal action choices for Lane 0 x Lane 1 queues."""
        matrix = []
        for q0 in range(self.max_queue_len + 1):
            row = []
            for q1 in range(self.max_queue_len + 1):
                # Sample state with q0, q1, and fixed_lane2 for lane 2, rest 0
                sample_queues = [q0, q1, fixed_lane2] + [0] * (self.num_counters - 3)
                state = QueueState(sample_queues)
                action = self.choose_action(state, explore=False)
                row.append(int(action))
            matrix.append(row)
        return matrix

    def get_value_matrix(self, fixed_lane2=0):
        """Returns 6x6 matrix of max value / utility for Lane 0 x Lane 1 queues."""
        matrix = []
        for q0 in range(self.max_queue_len + 1):
            row = []
            for q1 in range(self.max_queue_len + 1):
                sample_queues = [q0, q1, fixed_lane2] + [0] * (self.num_counters - 3)
                state = QueueState(sample_queues)
                vals = self.get_action_values(state)
                row.append(round(float(np.max(vals)), 2))
            matrix.append(row)
        return matrix

    def save(self, filepath):
        """Serialize model parameters to disk."""
        import pickle
        state_dict = {"name": self.name}
        for attr in ["epsilon", "noise_std", "global_counts", "global_rewards"]:
            if hasattr(self, attr):
                state_dict[attr] = getattr(self, attr)
        for dict_attr in ["Q", "counts", "rewards"]:
            if hasattr(self, dict_attr):
                state_dict[dict_attr] = dict(getattr(self, dict_attr))
        for net_attr in ["actor", "actor_target", "critic", "critic_target", "q_net", "target_net", "policy_net", "value_net"]:
            if hasattr(self, net_attr):
                net = getattr(self, net_attr)
                if hasattr(net, "get_weights"):
                    state_dict[net_attr] = net.get_weights()
        with open(filepath, "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, filepath):
        """Load model parameters from disk."""
        import pickle
        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)
        for attr, val in state_dict.items():
            if hasattr(self, attr):
                target_attr = getattr(self, attr)
                if hasattr(target_attr, "set_weights"):
                    target_attr.set_weights(val)
                elif isinstance(target_attr, dict):
                    target_attr.clear()
                    target_attr.update(val)
                else:
                    setattr(self, attr, val)

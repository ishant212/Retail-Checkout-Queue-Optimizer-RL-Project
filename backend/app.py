from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation_engine import SimulationService
from simpy_engine import SimPyCheckoutSimulator
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

def make_serializable(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [make_serializable(x) for x in obj.tolist()]
    except Exception:
        pass
    if isinstance(obj, (str, bytes)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_serializable(i) for i in obj]
    return obj


sim_service = SimulationService()


@app.route("/api/status", methods=["GET"])
def get_status():
    active_agent = sim_service.agent
    eps = getattr(active_agent, "epsilon", 0.05)
    gamma = getattr(active_agent, "gamma", 0.95)
    alpha = getattr(active_agent, "alpha", getattr(active_agent, "alpha_pi", getattr(active_agent, "alpha_actor", 0.1)))

    return jsonify({
        "status": "online",
        "activeModelId": sim_service.active_model_id,
        "activeModelName": active_agent.name,
        "activeModelType": active_agent.model_type,
        "trainedEpisodes": sim_service.trained_episodes,
        "modelEpisodes": sim_service.model_episodes,
        "numCounters": sim_service.num_counters,
        "maxQueueLen": sim_service.max_queue_len,
        "serviceRates": sim_service.service_rates,
        "epsilon": round(float(eps), 4),
        "gamma": round(float(gamma), 4),
        "alpha": round(float(alpha), 4),
        "models": sim_service.get_models_overview()
    })

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "activeModelId": sim_service.active_model_id,
        "modelEpisodes": sim_service.model_episodes,
        "models": sim_service.get_models_overview()
    })

@app.route("/api/models/select", methods=["POST"])
def select_model():
    data = request.get_json() or {}
    model_id = data.get("modelId", "q_learning")
    success = sim_service.set_active_model(model_id)
    return jsonify({
        "success": success,
        "activeModelId": sim_service.active_model_id,
        "activeModelName": sim_service.agent.name
    })

@app.route("/api/train", methods=["POST"])
def train():
    data = request.get_json() or {}
    model_type = data.get("modelType", None)  # 'q_learning', 'sarsa', etc. or None/'all'
    model_episodes = data.get("modelEpisodes", None)  # Dict mapping model_id -> episodes
    episodes = int(data.get("episodes", 250))
    steps_per_episode = int(data.get("stepsPerEpisode", 25))
    alpha = float(data.get("alpha", 0.1))
    gamma = float(data.get("gamma", 0.95))
    epsilon = float(data.get("epsilon", 0.4))
    epsilon_decay = float(data.get("epsilonDecay", 0.995))
    min_epsilon = float(data.get("minEpsilon", 0.05))
    service_rates = data.get("serviceRates", None)
    if service_rates:
        service_rates = [float(r) for r in service_rates]

    result = sim_service.train_agent(
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        service_rates=service_rates,
        model_type=model_type,
        model_episodes=model_episodes
    )
    return jsonify(make_serializable(result))

@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    data = request.get_json() or {}
    steps = int(data.get("steps", 100))
    service_rates = data.get("serviceRates", None)
    selected_model_id = data.get("modelId", None)
    if service_rates:
        service_rates = [float(r) for r in service_rates]

    result = sim_service.run_benchmark(
        steps=steps,
        service_rates=service_rates,
        selected_model_id=selected_model_id
    )
    return jsonify(make_serializable(result))

@app.route("/api/interactive/reset", methods=["POST"])
def interactive_reset():
    data = request.get_json() or {}
    model_id = data.get("modelId", None)
    result = sim_service.reset_interactive(model_id=model_id)
    return jsonify(make_serializable(result))

@app.route("/api/interactive/step", methods=["POST"])
def interactive_step():
    data = request.get_json() or {}
    action = data.get("action", None)
    mode = data.get("mode", sim_service.active_model_id)

    result = sim_service.step_interactive(action=action, mode=mode)
    return jsonify(make_serializable(result))

@app.route("/api/policy", methods=["GET"])
def get_policy():
    fixed_lane2 = int(request.args.get("fixedLane2", 0))
    model_id = request.args.get("modelId", sim_service.active_model_id)
    result = sim_service.get_policy_matrix(model_id=model_id, fixed_lane2=fixed_lane2)
    return jsonify(make_serializable(result))

@app.route("/api/q_values", methods=["POST"])
def get_q_values():
    data = request.get_json() or {}
    q0 = int(data.get("q0", 0))
    q1 = int(data.get("q1", 0))
    q2 = int(data.get("q2", 0))
    model_id = data.get("modelId", sim_service.active_model_id)

    result = sim_service.get_state_q_values(q0, q1, q2, model_id=model_id)
    return jsonify(make_serializable(result))

@app.route("/api/simpy/run", methods=["POST"])
def run_simpy():
    data = request.get_json() or {}
    arrival_rate = float(data.get("arrivalRate", 1.0))
    simulation_time = float(data.get("simulationTime", 120.0))
    service_rates = data.get("serviceRates", sim_service.service_rates)
    service_rates = [float(r) for r in service_rates]

    simulator = SimPyCheckoutSimulator(
        service_rates=service_rates,
        arrival_rate=arrival_rate,
        simulation_time=simulation_time
    )

    results = simulator.run_comparison(
        agents_dict=sim_service.models,
        max_queue_len=sim_service.max_queue_len
    )
    return jsonify(make_serializable(results))

@app.route("/api/comparative_analysis", methods=["GET", "POST"])
def get_comparative_analysis():
    scenario = "normal"
    if request.method == "POST":
        data = request.get_json() or {}
        scenario = data.get("scenario", "normal")
    else:
        scenario = request.args.get("scenario", "normal")
    result = sim_service.run_comparative_analysis(scenario_name=scenario)
    return jsonify(make_serializable(result))

@app.route("/api/lanes", methods=["GET"])
def get_lanes():
    lanes_data = [
        {
            "id": l.lane_id,
            "type": l.lane_type,
            "isOpen": l.is_open,
            "capacity": l.capacity,
            "serviceRate": l.service_rate,
            "operatingCost": l.operating_cost,
            "maxBasket": l.max_basket,
            "minBasket": l.min_basket,
            "queueLength": l.queue_length,
            "remainingCapacity": l.remaining_capacity,
            "isIdle": l.is_idle
        }
        for l in sim_service.interactive_env.lanes
    ]
    return jsonify(make_serializable(lanes_data))

@app.route("/api/lanes/update", methods=["POST"])
def update_lane():
    data = request.get_json() or {}
    lane_id = int(data.get("laneId", 0))
    if 0 <= lane_id < sim_service.num_counters:
        lane = sim_service.interactive_env.lanes[lane_id]
        if "type" in data:
            lane.set_type(data["type"])
        if "isOpen" in data:
            lane.is_open = bool(data["isOpen"])
        if "capacity" in data:
            lane.capacity = max(1, min(20, int(data["capacity"])))
        if "serviceRate" in data:
            lane.service_rate = max(0.05, min(2.0, float(data["serviceRate"])))
        if "operatingCost" in data:
            lane.operating_cost = max(0.1, min(10.0, float(data["operatingCost"])))
        return jsonify({"success": True, "laneId": lane_id})
    return jsonify({"success": False, "error": f"Invalid laneId {lane_id}"}), 400



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[OptiQueue AI] Starting Multi-Model Retail Queue Optimizer API on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)

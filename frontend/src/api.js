/**
 * OptiQueue AI - Backend Multi-Model Simulation API Client
 * Connects directly to Flask Backend on http://127.0.0.1:5000/api
 * Supports Q-Learning, SARSA, Multi-Armed Bandits, REINFORCE, A2C, and PPO.
 */

const API_BASE_URL = 'http://127.0.0.1:5000/api';

async function request(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {})
      }
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.warn(`API request to ${endpoint} failed:`, err.message);
    throw err;
  }
}

export const api = {
  async getStatus() {
    try {
      return await request('/status');
    } catch {
      return {
        status: 'offline',
        activeModelId: 'q_learning',
        activeModelName: 'Q-Learning',
        trainedEpisodes: 250,
        modelEpisodes: {
          q_learning: 250,
          sarsa: 250,
          bandit: 250,
          reinforce: 200,
          a2c: 200,
          ppo: 200,
          dqn: 200,
          double_dqn: 250,
          ddpg: 150
        },
        numCounters: 10,
        maxQueueLen: 6,
        serviceRates: [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5],
        epsilon: 0.05,
        gamma: 0.95,
        alpha: 0.1,
        models: [
          { id: 'q_learning', name: 'Q-Learning', family: 'Value-Based (Off-Policy TD)', tag: '1-Step TD Control', isActive: true, isTrained: true },
          { id: 'sarsa', name: 'SARSA', family: 'Value-Based (On-Policy TD)', tag: 'Safe On-Policy TD', isActive: false, isTrained: true },
          { id: 'bandit', name: 'Multi-Armed Bandit', family: 'Bandit / Adaptive Strategy', tag: 'Contextual UCB1', isActive: false, isTrained: true },
          { id: 'reinforce', name: 'REINFORCE (w/ Baseline)', family: 'Policy Gradient', tag: 'Softmax Policy Gradient', isActive: false, isTrained: true },
          { id: 'a2c', name: 'A2C (Advantage Actor-Critic)', family: 'Actor-Critic', tag: 'Synchronous Actor-Critic', isActive: false, isTrained: true },
          { id: 'ppo', name: 'PPO (Proximal Policy Optimization)', family: 'Actor-Critic (Policy Optimization)', tag: 'Clipped Surrogate Objective', isActive: false, isTrained: true },
          { id: 'dqn', name: 'DQN (Deep Q-Network)', family: 'Deep RL (Value-Based)', tag: 'Experience Replay + Target Net', isActive: false, isTrained: true },
          { id: 'double_dqn', name: 'Double DQN', family: 'Deep RL (Value-Based)', tag: 'Decoupled Q-Target', isActive: false, isTrained: true },
          { id: 'ddpg', name: 'DDPG (Deep Deterministic PG)', family: 'Deep RL (Actor-Critic)', tag: 'Deterministic Policy Gradient', isActive: false, isTrained: true }
        ]
      };
    }
  },

  async getModels() {
    return await request('/models');
  },

  async selectModel(modelId) {
    return await request('/models/select', {
      method: 'POST',
      body: JSON.stringify({ modelId })
    });
  },

  async train(params = {}) {
    return await request('/train', {
      method: 'POST',
      body: JSON.stringify(params)
    });
  },

  async runBenchmark(steps = 100, serviceRates = null, modelId = null) {
    return await request('/benchmark', {
      method: 'POST',
      body: JSON.stringify({ steps, serviceRates, modelId })
    });
  },

  async resetInteractive(modelId = null) {
    return await request('/interactive/reset', {
      method: 'POST',
      body: JSON.stringify({ modelId })
    });
  },

  async stepInteractive(action = null, mode = 'q_learning') {
    return await request('/interactive/step', {
      method: 'POST',
      body: JSON.stringify({ action, mode })
    });
  },

  async getPolicy(fixedLane2 = 0, modelId = null) {
    const query = modelId ? `fixedLane2=${fixedLane2}&modelId=${modelId}` : `fixedLane2=${fixedLane2}`;
    return await request(`/policy?${query}`);
  },

  async getQValues(q0, q1, q2, modelId = null) {
    return await request('/q_values', {
      method: 'POST',
      body: JSON.stringify({ q0, q1, q2, modelId })
    });
  },

  async runSimPy(params = {}) {
    return await request('/simpy/run', {
      method: 'POST',
      body: JSON.stringify(params)
    });
  },

  async getComparativeAnalysis(scenario = 'normal') {
    return await request(`/comparative_analysis?scenario=${encodeURIComponent(scenario)}`);
  },

  async getLanes() {
    return await request('/lanes');
  },

  async updateLane(laneData) {
    return await request('/lanes/update', {
      method: 'POST',
      body: JSON.stringify(laneData)
    });
  }
};

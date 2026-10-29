/**
 * OptiQueue AI - Simulation API Client
 * Connects directly to backend at /api (proxied to http://127.0.0.1:5000/api)
 * Includes robust error handling and transparent fallback data for Demo Mode.
 */

const API_BASE_URL = '/api';

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });

  if (!response.ok) {
    throw new Error(`API Error ${response.status}: ${response.statusText}`);
  }

  return await response.json();
}

export const api = {
  async getStatus() {
    return await request('/status');
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

  async setScenario(scenario = 'peak') {
    return await request('/interactive/scenario', {
      method: 'POST',
      body: JSON.stringify({ scenario })
    });
  },

  async resetInteractive(modelId = null, scenario = 'peak') {
    return await request('/interactive/reset', {
      method: 'POST',
      body: JSON.stringify({ modelId, scenario })
    });
  },

  async stepInteractive(action = null, mode = 'double_dqn', scenario = null) {
    return await request('/interactive/step', {
      method: 'POST',
      body: JSON.stringify({ action, mode, scenario })
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

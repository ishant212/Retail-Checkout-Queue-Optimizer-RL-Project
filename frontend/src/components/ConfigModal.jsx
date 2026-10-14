import React, { useState, useEffect } from 'react';
import { Settings, Save, RotateCcw, Brain, CheckCircle2, RefreshCw, Zap, Sliders, AlertCircle, Layers } from 'lucide-react';
import { api } from '../api';

const DEFAULT_EPISODES = {
  q_learning: 250,
  sarsa: 250,
  bandit: 250,
  reinforce: 200,
  a2c: 200,
  ppo: 200,
  dqn: 200,
  double_dqn: 250,
  ddpg: 150
};

const MODEL_SPECS = [
  { id: 'q_learning', name: 'Q-Learning', family: 'Value-Based (Off-Policy TD)', desc: '1-Step off-policy Bellman optimality TD control' },
  { id: 'sarsa', name: 'SARSA', family: 'Value-Based (On-Policy TD)', desc: 'Safe on-policy temporal difference learning' },
  { id: 'bandit', name: 'Multi-Armed Bandit', family: 'Adaptive Arms (UCB1)', desc: 'Contextual upper confidence bound arm selection' },
  { id: 'reinforce', name: 'REINFORCE (w/ Baseline)', family: 'Policy Gradient', desc: 'Monte Carlo policy gradient with state-value baseline' },
  { id: 'a2c', name: 'A2C (Advantage Actor-Critic)', family: 'Actor-Critic', desc: 'Synchronous 1-step TD advantage actor-critic' },
  { id: 'ppo', name: 'PPO (Proximal Policy Opt)', family: 'Actor-Critic', desc: 'Clipped surrogate objective with GAE advantage estimation' },
  { id: 'dqn', name: 'DQN (Deep Q-Network)', family: 'Deep RL (Value-Based)', desc: 'Neural network Q-eval with experience replay buffer' },
  { id: 'double_dqn', name: 'Double DQN', family: 'Deep RL (Value-Based)', desc: 'Decouples greedy action selection and target evaluation' },
  { id: 'ddpg', name: 'DDPG (Deep Deterministic PG)', family: 'Deep RL (Actor-Critic)', desc: 'Deterministic policy gradient with Polyak soft updates' }
];

export default function ConfigModal({ serviceRates = [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5], onUpdateConfig, isTraining, modelEpisodes = {} }) {
  const [modelType, setModelType] = useState('all');

  const [lanes, setLanes] = useState([
    { id: 0, type: 'express', isOpen: true, capacity: 4, serviceRate: 0.8, operatingCost: 1.0 },
    { id: 1, type: 'standard', isOpen: true, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
    { id: 2, type: 'standard', isOpen: true, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
    { id: 3, type: 'large_basket', isOpen: true, capacity: 8, serviceRate: 0.35, operatingCost: 0.9 },
    { id: 4, type: 'standard', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
    { id: 5, type: 'standard', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
    { id: 6, type: 'express', isOpen: false, capacity: 4, serviceRate: 0.8, operatingCost: 1.0 },
    { id: 7, type: 'large_basket', isOpen: false, capacity: 8, serviceRate: 0.35, operatingCost: 0.9 },
    { id: 8, type: 'flexible', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.85 },
    { id: 9, type: 'flexible', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.85 }
  ]);
  const [laneSaving, setLaneSaving] = useState({});

  useEffect(() => {
    let mounted = true;
    api.getLanes().then((res) => {
      if (mounted && Array.isArray(res) && res.length === 10) {
        setLanes(res);
      }
    }).catch(() => {});
    return () => { mounted = false; };
  }, []);

  const handleLaneFieldChange = (laneId, field, value) => {
    setLanes((prev) =>
      prev.map((l) => (l.id === laneId ? { ...l, [field]: value } : l))
    );
  };

  const handleSaveSingleLane = async (laneId) => {
    const lane = lanes.find((l) => l.id === laneId);
    if (!lane) return;
    setLaneSaving((prev) => ({ ...prev, [laneId]: true }));
    try {
      await api.updateLane({
        laneId: lane.id,
        type: lane.type,
        isOpen: lane.isOpen,
        capacity: Number(lane.capacity),
        serviceRate: Number(lane.serviceRate),
        operatingCost: Number(lane.operatingCost)
      });
      setSuccessMessage(`Lane ${laneId} configuration updated and synced with backend!`);
    } catch (err) {
      setValidationError(`Failed to update Lane ${laneId}: ${err.message}`);
    } finally {
      setLaneSaving((prev) => ({ ...prev, [laneId]: false }));
    }
  };

  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.95);
  const [epsilon, setEpsilon] = useState(0.4);
  const [epsilonDecay, setEpsilonDecay] = useState(0.995);
  const [minEpsilon, setMinEpsilon] = useState(0.05);
  const [stepsPerEpisode, setStepsPerEpisode] = useState(25);

  // Independent episode configuration per model
  const [episodesMap, setEpisodesMap] = useState({
    q_learning: modelEpisodes.q_learning ?? DEFAULT_EPISODES.q_learning,
    sarsa: modelEpisodes.sarsa ?? DEFAULT_EPISODES.sarsa,
    bandit: modelEpisodes.bandit ?? DEFAULT_EPISODES.bandit,
    reinforce: modelEpisodes.reinforce ?? DEFAULT_EPISODES.reinforce,
    a2c: modelEpisodes.a2c ?? DEFAULT_EPISODES.a2c,
    ppo: modelEpisodes.ppo ?? DEFAULT_EPISODES.ppo,
    dqn: modelEpisodes.dqn ?? DEFAULT_EPISODES.dqn,
    double_dqn: modelEpisodes.double_dqn ?? DEFAULT_EPISODES.double_dqn,
    ddpg: modelEpisodes.ddpg ?? DEFAULT_EPISODES.ddpg
  });

  const [validationError, setValidationError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handleEpisodeChange = (mid, val) => {
    const num = parseInt(val, 10);
    setEpisodesMap((prev) => ({
      ...prev,
      [mid]: isNaN(num) ? '' : num
    }));
  };

  const handleResetDefaults = () => {
    setModelType('all');
    setAlpha(0.1);
    setGamma(0.95);
    setEpsilon(0.4);
    setEpsilonDecay(0.995);
    setMinEpsilon(0.05);
    setStepsPerEpisode(25);
    setEpisodesMap(DEFAULT_EPISODES);
    setLanes([
      { id: 0, type: 'express', isOpen: true, capacity: 4, serviceRate: 0.8, operatingCost: 1.0 },
      { id: 1, type: 'standard', isOpen: true, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
      { id: 2, type: 'standard', isOpen: true, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
      { id: 3, type: 'large_basket', isOpen: true, capacity: 8, serviceRate: 0.35, operatingCost: 0.9 },
      { id: 4, type: 'standard', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
      { id: 5, type: 'standard', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.8 },
      { id: 6, type: 'express', isOpen: false, capacity: 4, serviceRate: 0.8, operatingCost: 1.0 },
      { id: 7, type: 'large_basket', isOpen: false, capacity: 8, serviceRate: 0.35, operatingCost: 0.9 },
      { id: 8, type: 'flexible', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.85 },
      { id: 9, type: 'flexible', isOpen: false, capacity: 6, serviceRate: 0.5, operatingCost: 0.85 }
    ]);
    setValidationError('');
    setSuccessMessage('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSuccessMessage('');
    setValidationError('');

    // Validate that each episode count is a positive integer in [10, 2000]
    const cleanedEpisodes = {};
    for (const spec of MODEL_SPECS) {
      const val = episodesMap[spec.id];
      const parsed = parseInt(val, 10);
      if (isNaN(parsed) || parsed < 10 || parsed > 2000) {
        setValidationError(`Invalid episode count for ${spec.name}: must be an integer between 10 and 2000.`);
        return;
      }
      cleanedEpisodes[spec.id] = parsed;
    }

    const config = {
      modelType: modelType === 'all' ? null : modelType,
      modelEpisodes: cleanedEpisodes,
      episodes: modelType === 'all' ? 250 : cleanedEpisodes[modelType],
      serviceRates: lanes.map((l) => Number(l.serviceRate)),
      alpha: Number(alpha),
      gamma: Number(gamma),
      epsilon: Number(epsilon),
      epsilonDecay: Number(epsilonDecay),
      minEpsilon: Number(minEpsilon),
      stepsPerEpisode: Number(stepsPerEpisode)
    };

    await onUpdateConfig(config);
    setSuccessMessage('Algorithm hyperparameters and independent episode counts updated successfully!');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Banner */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Settings size={22} color="#10b981" />
          <h2 style={{ fontSize: '1.25rem' }}>Multi-Model Hyperparameters & Training Studio</h2>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
          Configure asymmetric checkout service rates, tune algorithm parameters, and specify independent training episode counts for each of the nine RL models.
        </p>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {/* Model Target Selection Card */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <Brain size={18} color="#6366f1" />
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>Training Target Model</h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '12px' }}>
            <label
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '10px',
                padding: '12px',
                borderRadius: '8px',
                border: modelType === 'all' ? '2px solid var(--accent-primary)' : '1px solid var(--border-color)',
                background: modelType === 'all' ? 'var(--accent-primary-light)' : '#ffffff',
                cursor: 'pointer'
              }}
            >
              <input
                type="radio"
                name="modelTarget"
                value="all"
                checked={modelType === 'all'}
                onChange={(e) => setModelType(e.target.value)}
                style={{ marginTop: '3px', accentColor: 'var(--accent-primary)' }}
              />
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>All Models in Arena</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Trains all 9 RL models concurrently</div>
              </div>
            </label>

            {MODEL_SPECS.map((opt) => (
              <label
                key={opt.id}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  padding: '12px',
                  borderRadius: '8px',
                  border: modelType === opt.id ? '2px solid var(--accent-primary)' : '1px solid var(--border-color)',
                  background: modelType === opt.id ? 'var(--accent-primary-light)' : '#ffffff',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="radio"
                  name="modelTarget"
                  value={opt.id}
                  checked={modelType === opt.id}
                  onChange={(e) => setModelType(e.target.value)}
                  style={{ marginTop: '3px', accentColor: 'var(--accent-primary)' }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{opt.name}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{opt.family}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Independent Training Episode Controls For All 9 Models */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <Sliders size={18} color="#0284c7" />
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>Independent Training Episodes (9 Models)</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '16px' }}>
            Assign tailored episode counts to each individual model architecture based on sample complexity and convergence characteristics (10 to 2000 episodes).
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            {MODEL_SPECS.map((spec) => {
              const currentEp = episodesMap[spec.id] ?? DEFAULT_EPISODES[spec.id];
              return (
                <div
                  key={spec.id}
                  style={{
                    padding: '14px',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-tertiary)'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <div>
                      <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>{spec.name}</strong>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{spec.family}</div>
                    </div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px', borderRadius: '4px', background: '#e0e7ff', color: '#4338ca' }}>
                      {currentEp} eps
                    </span>
                  </div>

                  <div style={{ marginTop: '8px' }}>
                    <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, marginBottom: '4px' }}>
                      Episodes:
                    </label>
                    <input
                      type="number"
                      min="10"
                      max="2000"
                      step="10"
                      value={currentEp}
                      onChange={(e) => handleEpisodeChange(spec.id, e.target.value)}
                      style={{
                        width: '100%',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        border: '1px solid var(--border-color)',
                        background: '#ffffff',
                        fontSize: '0.85rem'
                      }}
                    />
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    {spec.desc}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 10-Lane Configuration Studio */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Layers size={18} color="#10b981" />
              <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>10-Lane Checkout Configuration Studio</h3>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Configure Lane Types, Open Status, Capacities, Rates, and Operating Costs
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '8px 10px' }}>Lane</th>
                  <th style={{ padding: '8px 10px' }}>Type</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                  <th style={{ padding: '8px 10px' }}>Capacity (K)</th>
                  <th style={{ padding: '8px 10px' }}>Service Rate (μ)</th>
                  <th style={{ padding: '8px 10px' }}>Op Cost ($/s)</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right' }}>Sync Live</th>
                </tr>
              </thead>
              <tbody>
                {lanes.map((lane) => (
                  <tr key={lane.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700 }}>
                      Lane {lane.id}
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      <select
                        value={lane.type}
                        onChange={(e) => handleLaneFieldChange(lane.id, 'type', e.target.value)}
                        style={{ padding: '4px 8px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)', fontSize: '0.82rem' }}
                      >
                        <option value="express">Express (≤5 items)</option>
                        <option value="standard">Standard</option>
                        <option value="large_basket">Large Basket</option>
                        <option value="flexible">Flexible</option>
                      </select>
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      <button
                        type="button"
                        onClick={() => handleLaneFieldChange(lane.id, 'isOpen', !lane.isOpen)}
                        style={{
                          padding: '3px 8px',
                          borderRadius: '4px',
                          border: 'none',
                          cursor: 'pointer',
                          fontWeight: 700,
                          fontSize: '0.78rem',
                          background: lane.isOpen ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                          color: lane.isOpen ? '#10b981' : '#ef4444'
                        }}
                      >
                        {lane.isOpen ? 'Open' : 'Closed'}
                      </button>
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      <input
                        type="number"
                        min="2"
                        max="20"
                        value={lane.capacity}
                        onChange={(e) => handleLaneFieldChange(lane.id, 'capacity', e.target.value)}
                        style={{ width: '60px', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                      />
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      <input
                        type="number"
                        step="0.05"
                        min="0.1"
                        max="1.5"
                        value={lane.serviceRate}
                        onChange={(e) => handleLaneFieldChange(lane.id, 'serviceRate', e.target.value)}
                        style={{ width: '70px', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                      />
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      <input
                        type="number"
                        step="0.05"
                        min="0.2"
                        max="5.0"
                        value={lane.operatingCost}
                        onChange={(e) => handleLaneFieldChange(lane.id, 'operatingCost', e.target.value)}
                        style={{ width: '65px', padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                      />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                      <button
                        type="button"
                        onClick={() => handleSaveSingleLane(lane.id)}
                        disabled={laneSaving[lane.id]}
                        style={{
                          padding: '4px 10px',
                          borderRadius: '4px',
                          border: '1px solid var(--border-color)',
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-primary)',
                          cursor: 'pointer',
                          fontSize: '0.75rem',
                          fontWeight: 600
                        }}
                      >
                        {laneSaving[lane.id] ? 'Saving...' : 'Save'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Shared RL Training Hyperparameters */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <Brain size={18} color="#3b82f6" />
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>Shared Algorithm Hyperparameters</h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
                Learning Rate (α):
              </label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="1.0"
                value={alpha}
                onChange={(e) => setAlpha(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Step size (Default: 0.10)</span>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
                Discount Factor (γ):
              </label>
              <input
                type="number"
                step="0.01"
                min="0.5"
                max="0.99"
                value={gamma}
                onChange={(e) => setGamma(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Horizon discount weight (Default: 0.95)</span>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
                Initial Exploration (ε):
              </label>
              <input
                type="number"
                step="0.05"
                min="0.0"
                max="1.0"
                value={epsilon}
                onChange={(e) => setEpsilon(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Initial epsilon-greedy probability (Default: 0.40)</span>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
                Steps per Episode:
              </label>
              <input
                type="number"
                step="5"
                min="10"
                max="100"
                value={stepsPerEpisode}
                onChange={(e) => setStepsPerEpisode(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Simulation horizon per episode (Default: 25)</span>
            </div>
          </div>
        </div>

        {/* Validation Error Alert */}
        {validationError && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '14px 18px', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid #ef4444', borderRadius: '8px', color: '#dc2626' }}>
            <AlertCircle size={18} />
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{validationError}</span>
          </div>
        )}

        {/* Success Alert */}
        {successMessage && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '14px 18px', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid #10b981', borderRadius: '8px', color: '#10b981' }}>
            <CheckCircle2 size={18} />
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{successMessage}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
          <button type="button" className="btn btn-outline" onClick={handleResetDefaults} disabled={isTraining}>
            <RotateCcw size={16} />
            Reset Defaults
          </button>
          <button type="submit" className="btn btn-primary" disabled={isTraining}>
            {isTraining ? <RefreshCw size={16} className="spin-icon" /> : <Save size={16} />}
            {isTraining ? 'Training in Progress...' : 'Apply & Retrain Agents'}
          </button>
        </div>
      </form>
    </div>
  );
}

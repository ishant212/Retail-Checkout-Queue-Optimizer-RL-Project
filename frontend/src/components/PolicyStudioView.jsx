import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  Brain,
  Sliders,
  Zap,
  Repeat,
  CircleDot,
  TrendingUp,
  Cpu,
  Rocket,
  CheckCircle2,
  RefreshCw,
  Sparkles,
  Layers,
  GitFork,
  Target
} from 'lucide-react';
import { api } from '../api';

const MODELS = [
  { id: 'q_learning', name: 'Q-Learning', tag: 'Off-Policy TD', icon: Zap, color: '#10b981' },
  { id: 'sarsa', name: 'SARSA', tag: 'On-Policy TD', icon: Repeat, color: '#06b6d4' },
  { id: 'bandit', name: 'Multi-Armed Bandit', tag: 'Contextual UCB1', icon: CircleDot, color: '#f59e0b' },
  { id: 'reinforce', name: 'REINFORCE', tag: 'Monte Carlo PG', icon: TrendingUp, color: '#ec4899' },
  { id: 'a2c', name: 'A2C Actor-Critic', tag: 'Advantage TD', icon: Cpu, color: '#8b5cf6' },
  { id: 'ppo', name: 'PPO', tag: 'Clipped Surrogate', icon: Rocket, color: '#6366f1' },
  { id: 'dqn', name: 'DQN', tag: 'Deep Q-Net', icon: Layers, color: '#14b8a6' },
  { id: 'double_dqn', name: 'Double DQN', tag: 'Decoupled Target', icon: GitFork, color: '#0ea5e9' },
  { id: 'ddpg', name: 'DDPG', tag: 'Deterministic PG', icon: Target, color: '#a855f7' }
];

export default function PolicyStudioView({ trainingHistory, onRetrain, isTraining, maxQueueLen = 5 }) {
  const [selectedModel, setSelectedModel] = useState('q_learning');
  const [fixedLane2, setFixedLane2] = useState(0);
  const [policyData, setPolicyData] = useState(null);
  const [probeState, setProbeState] = useState({ q0: 1, q1: 2, q2: 0 });
  const [probeResult, setProbeResult] = useState(null);
  const [localHistory, setLocalHistory] = useState([]);

  // Fetch policy slice when fixedLane2 or selectedModel changes
  useEffect(() => {
    let isMounted = true;
    api.getPolicy(fixedLane2, selectedModel).then((data) => {
      if (isMounted) setPolicyData(data);
    }).catch(console.error);

    return () => { isMounted = false; };
  }, [fixedLane2, selectedModel]);

  // Fetch probe values for selected model
  useEffect(() => {
    let isMounted = true;
    api.getQValues(probeState.q0, probeState.q1, probeState.q2, selectedModel).then((data) => {
      if (isMounted) setProbeResult(data);
    }).catch(console.error);

    return () => { isMounted = false; };
  }, [probeState, selectedModel]);

  const probeChartData = probeResult?.qValues ? probeResult.qValues.map((v, i) => ({
    name: `Lane ${i}`,
    value: v,
    fill: i === probeResult.bestAction ? '#10b981' : (i < 2 ? '#3b82f6' : (i < 6 ? '#06b6d4' : '#f59e0b'))
  })) : [];

  const activeModelMeta = MODELS.find((m) => m.id === selectedModel) || MODELS[0];
  const ActiveIcon = activeModelMeta.icon;

  const handleRetrainCurrent = async () => {
    await onRetrain({ modelType: selectedModel, episodes: 250, stepsPerEpisode: 25 });
  };

  const handleRetrainAll = async () => {
    await onRetrain({ modelType: 'all', episodes: 250, stepsPerEpisode: 25 });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Policy Studio Header Controls */}
      <div className="glass-card" style={{ padding: '20px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <ActiveIcon size={22} color={activeModelMeta.color} />
            <h2 style={{ fontSize: '1.25rem', margin: 0 }}>RL Policy & Value Studio: {activeModelMeta.name}</h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Inspect optimal lane selection policies, value convergence, and arbitrary state transitions across {MODELS.length} RL architectures.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <button
            className="btn btn-secondary btn-sm"
            onClick={handleRetrainCurrent}
            disabled={isTraining}
            title={`Retrain only ${activeModelMeta.name}`}
            style={{ whiteSpace: 'nowrap', flexShrink: 0 }}
          >
            <RefreshCw size={14} className={isTraining ? 'spin-icon' : ''} style={{ flexShrink: 0 }} />
            <span>Retrain {activeModelMeta.name}</span>
          </button>

          <button
            className="btn btn-primary btn-sm"
            onClick={handleRetrainAll}
            disabled={isTraining}
            title={`Retrain all ${MODELS.length} models simultaneously`}
            style={{ whiteSpace: 'nowrap', flexShrink: 0 }}
          >
            <Sparkles size={14} className={isTraining ? 'spin-icon' : ''} style={{ flexShrink: 0 }} />
            <span>Retrain All {MODELS.length} Models</span>
          </button>
        </div>
      </div>

      {/* Model Selection Tabs Bar */}
      <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '6px', scrollbarWidth: 'thin' }}>
        {MODELS.map((m) => {
          const Icon = m.icon;
          const isSelected = m.id === selectedModel;
          return (
            <button
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className="btn"
              style={{
                flex: '0 0 auto',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '9px 16px',
                borderRadius: '8px',
                border: isSelected ? `2px solid ${m.color}` : '1px solid var(--border-color)',
                background: isSelected ? `${m.color}20` : 'var(--bg-secondary)',
                color: isSelected ? m.color : 'var(--text-primary)',
                fontWeight: isSelected ? 700 : 500,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={16} color={m.color} style={{ flexShrink: 0 }} />
              <span style={{ whiteSpace: 'nowrap' }}>{m.name}</span>
              <span style={{
                fontSize: '0.68rem',
                padding: '1px 6px',
                borderRadius: '4px',
                background: `${m.color}30`,
                color: m.color,
                whiteSpace: 'nowrap',
                flexShrink: 0
              }}>
                {m.tag}
              </span>
            </button>
          );
        })}
      </div>

      {/* Fixed Lane 2 Queue Slider */}
      <div className="glass-card" style={{ padding: '16px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Sliders size={18} color="var(--accent-color)" />
          <span style={{ fontWeight: 600 }}>2D Slice Plane (Fixed Counter 2 Queue):</span>
          <span className="badge badge-info" style={{ fontSize: '0.9rem', fontWeight: 700 }}>
            q₂ = {fixedLane2} customers
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: '240px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>0</span>
          <input
            type="range"
            min="0"
            max={maxQueueLen}
            value={fixedLane2}
            onChange={(e) => setFixedLane2(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{maxQueueLen}</span>
        </div>
      </div>

      {/* Policy Matrix & Heatmaps Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '24px' }}>
        {/* Learned Policy Decision Heatmap */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">{activeModelMeta.name} Decision Matrix (q₂ = {fixedLane2})</span>
            </div>
            <div style={{ display: 'flex', gap: '8px', fontSize: '0.75rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: '#10b981' }} /> L0 (Fast)
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: '#3b82f6' }} /> L1 (Mid)
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: '#f59e0b' }} /> L2 (Slow)
              </span>
            </div>
          </div>

          <div className="card-body">
            {policyData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '50px repeat(6, 1fr)', textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  <div>q₀ \ q₁</div>
                  {[0, 1, 2, 3, 4, 5].map((col) => <div key={col}>q₁={col}</div>)}
                </div>

                {policyData.learnedPolicy.map((row, rIdx) => (
                  <div key={rIdx} style={{ display: 'grid', gridTemplateColumns: '50px repeat(6, 1fr)', gap: '6px', alignItems: 'center' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textAlign: 'center' }}>
                      q₀={rIdx}
                    </div>
                    {row.map((action, cIdx) => {
                      const color = action === 0 ? '#10b981' : action === 1 ? '#3b82f6' : '#f59e0b';
                      return (
                        <div
                          key={cIdx}
                          style={{
                            background: `${color}25`,
                            border: `1px solid ${color}`,
                            color: color,
                            borderRadius: '6px',
                            height: '40px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontWeight: 700,
                            fontSize: '0.85rem',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease'
                          }}
                          onClick={() => setProbeState({ q0: rIdx, q1: cIdx, q2: fixedLane2 })}
                          title={`State (q0=${rIdx}, q1=${cIdx}, q2=${fixedLane2}) -> Pick Lane ${action}`}
                        >
                          L{action}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading decision matrix...</div>
            )}
          </div>
        </div>

        {/* Max Value Surface Heatmap */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">Max Action Value Surface (q₂ = {fixedLane2})</span>
            </div>
            <span className="badge badge-info">Expected Future Return</span>
          </div>

          <div className="card-body">
            {policyData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '50px repeat(6, 1fr)', textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  <div>q₀ \ q₁</div>
                  {[0, 1, 2, 3, 4, 5].map((col) => <div key={col}>q₁={col}</div>)}
                </div>

                {policyData.maxQValues.map((row, rIdx) => (
                  <div key={rIdx} style={{ display: 'grid', gridTemplateColumns: '50px repeat(6, 1fr)', gap: '6px', alignItems: 'center' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textAlign: 'center' }}>
                      q₀={rIdx}
                    </div>
                    {row.map((val, cIdx) => (
                      <div
                        key={cIdx}
                        style={{
                          background: 'var(--bg-tertiary)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '6px',
                          height: '40px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontFamily: 'monospace',
                          fontSize: '0.78rem',
                          fontWeight: 700,
                          color: val < -50 ? '#dc2626' : val < -25 ? '#d97706' : '#16a34a'
                        }}
                      >
                        {val}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading value surface...</div>
            )}
          </div>
        </div>
      </div>

      {/* State Value Probe Inspector */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Zap size={20} color="#10b981" />
              <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Arbitrary State Probe & Action Breakdown</h3>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginTop: '4px' }}>
              Select any queue configuration (q₀, q₁, q₂) to evaluate {activeModelMeta.name}'s action evaluations.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            {['q0', 'q1', 'q2'].map((laneKey, idx) => (
              <div key={laneKey} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>{laneKey}:</label>
                <select
                  value={probeState[laneKey]}
                  onChange={(e) => setProbeState({ ...probeState, [laneKey]: Number(e.target.value) })}
                  style={{
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-primary)',
                    borderRadius: '6px',
                    padding: '4px 8px'
                  }}
                >
                  {[0, 1, 2, 3, 4, 5].map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <div style={{ height: '220px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probeChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" opacity={0.6} />
                <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} />
                <YAxis stroke="var(--text-muted)" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: 'var(--border-color)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    boxShadow: 'var(--shadow-md)'
                  }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                  labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                />
                <Bar dataKey="value" name="Action Metric Value" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '12px' }}>
            <div style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Optimal Action Choice</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#10b981', marginTop: '2px' }}>
                Route to Lane {probeResult?.bestAction ?? 0}
              </div>
            </div>

            <div style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Action Distribution / Preferences</div>
              <div style={{ fontSize: '0.9rem', fontFamily: 'monospace', marginTop: '4px' }}>
                {probeResult ? probeResult.qValues.map((v, i) => `L${i}: ${v}`).join('  |  ') : '...'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

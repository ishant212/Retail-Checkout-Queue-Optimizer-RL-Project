import React, { useState } from 'react';
import {
  Play,
  Pause,
  StepForward,
  RotateCcw,
  Zap,
  Repeat,
  CircleDot,
  TrendingUp,
  Cpu,
  Rocket,
  Shuffle,
  Scale,
  UserCheck,
  ChevronDown,
  Layers,
  GitFork,
  Target,
  PlusCircle,
  MinusCircle,
  Clock,
  AlertCircle,
  CheckCircle2
} from 'lucide-react';

const MODEL_OPTIONS = [
  { id: 'double_dqn', label: 'Double DQN', category: 'Deep RL', tag: 'Proposed', icon: GitFork, color: '#0ea5e9' },
  { id: 'dqn', label: 'DQN (Deep Q-Net)', category: 'Deep RL', tag: 'Replay Buffer', icon: Layers, color: '#14b8a6' },
  { id: 'ppo', label: 'PPO Policy Opt', category: 'Actor-Critic', tag: 'Clipped Surrogate', icon: Rocket, color: '#6366f1' },
  { id: 'a2c', label: 'A2C Actor-Critic', category: 'Actor-Critic', tag: 'Advantage TD', icon: Cpu, color: '#8b5cf6' },
  { id: 'bandit', label: 'Multi-Armed Bandit', category: 'Adaptive', tag: 'UCB1 Arms', icon: CircleDot, color: '#f59e0b' },
  { id: 'q_learning', label: 'Q-Learning', category: 'Value-Based', tag: 'Off-Policy TD', icon: Zap, color: '#10b981' },
  { id: 'sarsa', label: 'SARSA', category: 'Value-Based', tag: 'On-Policy TD', icon: Repeat, color: '#06b6d4' },
  { id: 'reinforce', label: 'REINFORCE (w/ Base)', category: 'Policy Gradient', tag: 'Monte Carlo', icon: TrendingUp, color: '#ec4899' },
  { id: 'ddpg', label: 'DDPG Actor-Critic', category: 'Deep RL', tag: 'Deterministic PG', icon: Target, color: '#a855f7' },
  { id: 'shortest_queue', label: 'Shortest Queue', category: 'Heuristic', tag: 'Greedy Heuristic', icon: Scale, color: '#3b82f6' },
  { id: 'random', label: 'Random Valid Baseline', category: 'Baseline', tag: 'Uniform Policy', icon: Shuffle, color: '#94a3b8' },
  { id: 'manual', label: 'Manual Control', category: 'User', tag: 'Direct Actions', icon: UserCheck, color: '#f97316' }
];

export default function SimulationControls({
  isPlaying,
  onTogglePlay,
  onStep,
  onReset,
  mode,
  setMode,
  speed,
  setSpeed,
  stepCount,
  totalReward,
  lastActionValid = true,
  lastActionReason = '',
  onManualAction
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [actionCategory, setActionCategory] = useState('route'); // 'route', 'open', 'close', 'convert'
  const activeOpt = MODEL_OPTIONS.find((m) => m.id === mode) || MODEL_OPTIONS[0];
  const ActiveIcon = activeOpt.icon;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div className="glass-card controls-deck">
        {/* Play / Step / Reset Actions */}
        <div className="control-group">
          <button
            className={`btn ${isPlaying ? 'btn-secondary' : 'btn-primary'}`}
            onClick={onTogglePlay}
            title={isPlaying ? 'Pause Auto-dispatch' : 'Start Auto-dispatch'}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause Sim' : 'Auto Play'}
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => onStep()}
            disabled={isPlaying}
            title="Dispatch 1 customer arrival according to active policy"
          >
            <StepForward size={18} />
            Step Policy
          </button>

          <button
            className="btn btn-outline"
            onClick={onReset}
            title="Reset Queues to default state"
          >
            <RotateCcw size={18} />
            Reset State
          </button>
        </div>

        {/* Model & Policy Selector */}
        <div className="control-group" style={{ position: 'relative' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Active Policy:</span>

            <div style={{ position: 'relative' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                style={{
                  minWidth: '220px',
                  justifyContent: 'space-between',
                  background: 'var(--bg-tertiary)',
                  borderColor: 'var(--border-color)',
                  fontWeight: 600
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <ActiveIcon size={16} color={activeOpt.color} />
                  <span>{activeOpt.label}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      background: 'rgba(2, 132, 199, 0.12)',
                      color: 'var(--accent-primary)',
                      padding: '1px 5px',
                      borderRadius: '4px',
                      fontWeight: 700
                    }}
                  >
                    {activeOpt.tag}
                  </span>
                  <ChevronDown size={14} color="var(--text-muted)" />
                </div>
              </button>

              {dropdownOpen && (
                <div
                  className="dropdown-menu"
                  style={{
                    position: 'absolute',
                    top: 'calc(100% + 4px)',
                    left: 0,
                    width: '270px',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
                    zIndex: 100,
                    maxHeight: '380px',
                    overflowY: 'auto',
                    padding: '6px'
                  }}
                >
                  {MODEL_OPTIONS.map((opt) => {
                    const Icon = opt.icon;
                    const isSelected = opt.id === mode;
                    return (
                      <button
                        key={opt.id}
                        onClick={() => {
                          setMode(opt.id);
                          setDropdownOpen(false);
                        }}
                        style={{
                          width: '100%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '8px 10px',
                          border: 'none',
                          background: isSelected ? 'var(--accent-primary-light)' : 'transparent',
                          borderRadius: '6px',
                          cursor: 'pointer',
                          color: 'var(--text-primary)',
                          textAlign: 'left',
                          fontSize: '0.85rem'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Icon size={16} color={opt.color} />
                          <div>
                            <div style={{ fontWeight: isSelected ? 700 : 500 }}>{opt.label}</div>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{opt.category}</div>
                          </div>
                        </div>
                        <span
                          style={{
                            fontSize: '0.68rem',
                            padding: '2px 5px',
                            borderRadius: '4px',
                            background: isSelected ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                            color: isSelected ? '#ffffff' : 'var(--text-secondary)',
                            fontWeight: 600
                          }}
                        >
                          {opt.tag}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Speed Slider & Step Telemetry */}
        <div className="control-group">
          <div className="speed-slider-group">
            <span>Speed:</span>
            <input
              type="range"
              min="300"
              max="2500"
              step="100"
              value={2800 - speed}
              onChange={(e) => setSpeed(2800 - Number(e.target.value))}
              title="Adjust simulation delay per step"
            />
            <span style={{ minWidth: '40px', fontWeight: 600 }}>
              {speed <= 600 ? 'Fast' : speed <= 1200 ? 'Normal' : 'Slow'}
            </span>
          </div>

          <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: '16px', display: 'flex', gap: '14px' }}>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Step</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>{stepCount}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Reward</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: totalReward < 0 ? '#ef4444' : '#10b981' }}>
                {totalReward.toFixed(1)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Manual 41-Action Control Suite */}
      <div
        className="glass-card"
        style={{
          padding: '12px 18px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
              10-Lane Action Suite (41 Operations):
            </span>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                className={`btn btn-sm ${actionCategory === 'route' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                onClick={() => setActionCategory('route')}
              >
                Route (0..9)
              </button>
              <button
                className={`btn btn-sm ${actionCategory === 'open' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                onClick={() => setActionCategory('open')}
              >
                Open (10..19)
              </button>
              <button
                className={`btn btn-sm ${actionCategory === 'close' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                onClick={() => setActionCategory('close')}
              >
                Close (20..29)
              </button>
              <button
                className={`btn btn-sm ${actionCategory === 'convert' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                onClick={() => setActionCategory('convert')}
              >
                Convert (30..39)
              </button>
            </div>
          </div>

          {/* Validation Alert */}
          {!lastActionValid ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.78rem', color: '#dc2626', fontWeight: 700 }}>
              <AlertCircle size={14} /> Rejected: {lastActionReason} (Penalty Incurred)
            </span>
          ) : lastActionReason ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.78rem', color: '#10b981', fontWeight: 600 }}>
              <CheckCircle2 size={14} /> {lastActionReason}
            </span>
          ) : null}
        </div>

        {/* Action Button Grid */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          {actionCategory === 'route' && (
            <>
              {Array.from({ length: 10 }).map((_, i) => (
                <button
                  key={i}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '4px 8px', fontSize: '0.72rem' }}
                  onClick={() => onManualAction && onManualAction(i)}
                  disabled={isPlaying}
                  title={`Action ${i}: Route incoming customer to Lane ${i}`}
                >
                  A{i}: Route L{i}
                </button>
              ))}
            </>
          )}

          {actionCategory === 'open' && (
            <>
              {Array.from({ length: 10 }).map((_, i) => (
                <button
                  key={i}
                  className="btn btn-outline btn-sm"
                  style={{ padding: '4px 8px', fontSize: '0.72rem', color: '#10b981', borderColor: '#10b981' }}
                  onClick={() => onManualAction && onManualAction(10 + i)}
                  disabled={isPlaying}
                  title={`Action ${10 + i}: Open Lane ${i}`}
                >
                  <PlusCircle size={12} /> A{10 + i}: Open L{i}
                </button>
              ))}
            </>
          )}

          {actionCategory === 'close' && (
            <>
              {Array.from({ length: 10 }).map((_, i) => (
                <button
                  key={i}
                  className="btn btn-outline btn-sm"
                  style={{ padding: '4px 8px', fontSize: '0.72rem', color: '#ef4444', borderColor: '#ef4444' }}
                  onClick={() => onManualAction && onManualAction(20 + i)}
                  disabled={isPlaying}
                  title={`Action ${20 + i}: Close Lane ${i} safely (requires 0 queue)`}
                >
                  <MinusCircle size={12} /> A{20 + i}: Close L{i}
                </button>
              ))}
            </>
          )}

          {actionCategory === 'convert' && (
            <>
              {Array.from({ length: 10 }).map((_, i) => (
                <button
                  key={i}
                  className="btn btn-outline btn-sm"
                  style={{ padding: '4px 8px', fontSize: '0.72rem', color: '#f59e0b', borderColor: '#f59e0b' }}
                  onClick={() => onManualAction && onManualAction(30 + i)}
                  disabled={isPlaying}
                  title={`Action ${30 + i}: Convert Lane ${i} type (requires 0 queue)`}
                >
                  <Repeat size={12} /> A{30 + i}: Convert L{i}
                </button>
              ))}
            </>
          )}

          {/* Action 40: Hold Button */}
          <button
            className="btn btn-outline btn-sm"
            style={{ padding: '4px 10px', fontSize: '0.72rem', marginLeft: 'auto' }}
            onClick={() => onManualAction && onManualAction(40)}
            disabled={isPlaying}
            title="Action 40: Hold / No Action (Cashiers continuing service)"
          >
            <Clock size={12} /> A40: Hold / Standby
          </button>
        </div>
      </div>
    </div>
  );
}

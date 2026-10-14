import React from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import KPICards from './KPICards';
import CheckoutLanesView from './CheckoutLanesView';
import SimulationControls from './SimulationControls';
import { Sparkles, Trophy, Cpu, Zap, ArrowUpRight, CheckCircle2, Eye, Activity, DollarSign, Clock, ShieldAlert } from 'lucide-react';

export default function OverviewView({
  benchmarkData,
  currentQueues = Array(10).fill(0),
  serviceRates = [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5],
  openStatus = [true, true, true, true, false, false, false, false, false, false],
  laneTypes = ['express', 'standard', 'standard', 'large_basket', 'standard', 'standard', 'express', 'large_basket', 'flexible', 'flexible'],
  capacities = [4, 6, 6, 8, 6, 6, 4, 8, 6, 6],
  currentServing = Array(10).fill(null),
  incomingBasketSize = 5,
  isExpressEligible = true,
  counterUtilization = Array(10).fill(0),
  maxQueueLen = 6,
  lastAction = null,
  lastServed = Array(10).fill(false),
  qValues = Array(10).fill(0),
  allQValues = Array(41).fill(0),
  recommendedAction = 0,
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
  rewardBreakdown = {},
  stateComponents = {},
  congestionDuration = {},
  onManualDispatch,
  onManualAction
}) {
  const timeSeriesData = benchmarkData?.stepDetails ? benchmarkData.stepDetails.slice(0, 50) : [];
  const leaderboard = benchmarkData?.leaderboard || [];

  const laneBarData = currentQueues.map((q, idx) => {
    const rawType = (laneTypes && laneTypes[idx]) ? laneTypes[idx] : 'standard';
    const typeLabel = rawType.replace('_', ' ').toUpperCase();
    const cap = (capacities && capacities[idx]) ? capacities[idx] : maxQueueLen;
    return {
      name: `L${idx} (${typeLabel.slice(0, 4)})`,
      current: q || 0,
      capacity: cap,
      isOpen: openStatus[idx] ? 'Open' : 'Closed',
      rate: Math.round(((serviceRates && serviceRates[idx]) ?? 0.5) * 100)
    };
  });

  const bd = rewardBreakdown || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* KPI Cards Deck */}
      <KPICards
        benchmarkData={benchmarkData}
        currentQueues={currentQueues}
        serviceRates={serviceRates}
        activeModelId={mode}
      />

      {/* Simulation Controls Toolbar with 6-Action Suite */}
      <SimulationControls
        isPlaying={isPlaying}
        onTogglePlay={onTogglePlay}
        onStep={onStep}
        onReset={onReset}
        mode={mode}
        setMode={setMode}
        speed={speed}
        setSpeed={setSpeed}
        stepCount={stepCount}
        totalReward={totalReward}
        lastActionValid={lastActionValid}
        lastActionReason={lastActionReason}
        onManualAction={onManualAction}
      />

      {/* Visual Animated Checkout Counters */}
      <CheckoutLanesView
        queues={currentQueues}
        maxQueueLen={maxQueueLen}
        serviceRates={serviceRates}
        openStatus={openStatus}
        laneTypes={laneTypes}
        capacities={capacities}
        currentServing={currentServing}
        incomingBasketSize={incomingBasketSize}
        isExpressEligible={isExpressEligible}
        counterUtilization={counterUtilization}
        lastAction={lastAction}
        lastServed={lastServed}
        qValues={qValues}
        recommendedAction={recommendedAction}
        mode={mode}
        onManualDispatch={onManualDispatch}
        onManualAction={onManualAction}
      />

      {/* 10-State Observation Inspector & Centralized Reward Formula Card */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '24px' }}>
        {/* 10-State Space Observation Inspector */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <Eye size={18} color="#0284c7" />
              <span className="card-title">10-Lane Environment Observation Inspector</span>
            </div>
            <span className="badge badge-info">84-Dim Normalized Vector</span>
          </div>
          <div className="card-body" style={{ padding: '16px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '12px' }}>
              {/* 1. Queue Lengths */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  1. Queue Lengths [L0..L9]
                </div>
                <div style={{ fontSize: '0.9rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                  [{currentQueues.join(', ')}]
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Total: {currentQueues.reduce((a, b) => a + b, 0)} waiting customers</div>
              </div>

              {/* 2. Open / Closed Status */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  2. Counter Status [Open/Closed]
                </div>
                <div style={{ fontSize: '0.95rem', fontWeight: 800, marginTop: '2px', color: '#10b981' }}>
                  {openStatus.filter(Boolean).length} / 10 Lanes Open
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  [{openStatus.map((o) => (o ? 'O' : 'C')).join(' ')}]
                </div>
              </div>

              {/* 3. Lane Types & Capacities */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  3. Lane Configurations
                </div>
                <div style={{ fontSize: '0.88rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)' }}>
                  Exp: {laneTypes.filter((t) => t === 'express').length} | Std: {laneTypes.filter((t) => t === 'standard').length} | Lrg: {laneTypes.filter((t) => t === 'large_basket').length} | Flx: {laneTypes.filter((t) => t === 'flexible').length}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Capacities: [{capacities.join(', ')}]
                </div>
              </div>

              {/* 4. Waiting Time */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  4. Customer Waiting Time
                </div>
                <div style={{ fontSize: '1rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)' }}>
                  Avg: {stateComponents?.waiting_time?.avg ?? 0}s • Peak: {stateComponents?.waiting_time?.max ?? 0}s
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Step-accumulated: {stateComponents?.waiting_time?.step ?? 0}s</div>
              </div>

              {/* 5. Service Rates & Utilization */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  5. Service Rates & Utilization
                </div>
                <div style={{ fontSize: '0.88rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)' }}>
                  Avg μ: {(serviceRates.reduce((a, b) => a + b, 0) / (serviceRates.length || 1)).toFixed(2)} | Util: {Math.round((counterUtilization.reduce((a, b) => a + b, 0) / (counterUtilization.length || 1)) * 100)}%
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Serving now: {currentServing.filter(Boolean).length} of 10 lanes
                </div>
              </div>

              {/* 6. Arrival Rate */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  6. Customer Arrival Rate (λ)
                </div>
                <div style={{ fontSize: '1rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)' }}>
                  {stateComponents?.customer_arrival_rate ?? 1.0} cust / step
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Poisson arrival process</div>
              </div>

              {/* 7. Incoming Basket Size */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  7. Incoming Basket Size (b)
                </div>
                <div style={{ fontSize: '1rem', fontWeight: 800, marginTop: '2px', color: isExpressEligible ? '#10b981' : '#f59e0b' }}>
                  {incomingBasketSize} items
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {isExpressEligible ? 'Eligible for Express Lanes (L0, L6)' : 'Exceeds Express limit (>5 items)'}
                </div>
              </div>

              {/* 8. Simulation Time */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  8. Current Simulation Time (t)
                </div>
                <div style={{ fontSize: '1rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)' }}>
                  Step {stepCount}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Discrete simulation epoch</div>
              </div>

              {/* 9. Congestion Severity */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  9. Congestion Severity (C_t)
                </div>
                <div style={{ fontSize: '1rem', fontWeight: 800, marginTop: '2px', color: (stateComponents?.congestion_severity ?? 0) > 0 ? '#ef4444' : '#10b981' }}>
                  {stateComponents?.congestion_severity ?? 0} units
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Threshold = 10 total queue</div>
              </div>

              {/* 10. Operating Cost & Imbalance */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  10. Cost (K_t) & Imbalance (B_t)
                </div>
                <div style={{ fontSize: '0.92rem', fontWeight: 800, marginTop: '2px', color: 'var(--text-primary)' }}>
                  Cost: ${stateComponents?.operating_cost ?? 4.0} • Var: {stateComponents?.queue_imbalance ?? 0}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Single-step ΔT: {congestionDuration?.step ?? 0}s (Tot: {congestionDuration?.total ?? 0}s)</div>
              </div>
            </div>
          </div>
        </div>

        {/* Centralized Step Reward Function Breakdown */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <Activity size={18} color="#10b981" />
              <span className="card-title">Centralized Reward Formula Breakdown</span>
            </div>
            <span className="badge badge-success">Formula R_t Breakdown</span>
          </div>
          <div className="card-body" style={{ padding: '16px' }}>
            <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', background: 'var(--bg-tertiary)', padding: '10px', borderRadius: '6px', marginBottom: '12px', color: 'var(--text-secondary)' }}>
              R_t = α·N_served - β·W_t - γ·Q_t - δ·C_t - ρ·ΔT_cong - ζ·B_t - κ·K_t - ξ·I_t
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', fontSize: '0.82rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(16, 185, 129, 0.08)' }}>
                <span>+ Customer Served (α·N):</span>
                <strong style={{ color: '#10b981' }}>+{bd.term_served ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Waiting Time (β·W):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_wait ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Total Queue (γ·Q):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_queue ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Congestion Severity (δ·C):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_congestion ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Step Congestion Dur (ρ·ΔT):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_congestion_dur ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Queue Imbalance (ζ·B):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_imbalance ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Operating Cost (κ·K):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_operating ?? 0.0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.08)' }}>
                <span>- Invalid Action (ξ·I):</span>
                <strong style={{ color: '#ef4444' }}>{bd.term_invalid ?? 0.0}</strong>
              </div>
            </div>

            <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Net Step Reward: </span>
                <strong style={{ fontSize: '1.1rem', color: (bd.total_reward ?? 0) >= 0 ? '#10b981' : '#ef4444' }}>
                  {bd.total_reward ?? 0.0}
                </strong>
              </div>
              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Cumulative Episode: </span>
                <strong style={{ fontSize: '1.1rem', color: totalReward >= 0 ? '#10b981' : '#ef4444' }}>
                  {totalReward.toFixed(1)}
                </strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Multi-Model Benchmark Leaderboard Banner */}
      {leaderboard.length > 0 && (
        <div className="glass-card" style={{ padding: '20px 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Trophy size={20} color="#f59e0b" />
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>
                Multi-Model Arena Leaderboard (100-Step Discrete Benchmark)
              </h3>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Ranked dynamically by customer wait units (Click any model to activate)
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '12px' }}>
            {leaderboard.map((m, idx) => {
              const isSelected = m.id === mode;
              const isWinner = idx === 0;
              return (
                <div
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  style={{
                    padding: '12px',
                    borderRadius: '8px',
                    border: isSelected ? '2px solid var(--accent-primary)' : '1px solid var(--border-color)',
                    background: isSelected ? 'var(--accent-primary-light)' : '#ffffff',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    position: 'relative'
                  }}
                  title="Click to activate this model in simulation"
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span
                        style={{
                          fontSize: '0.72rem',
                          fontWeight: 800,
                          width: '20px',
                          height: '20px',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          background: isWinner ? '#d97706' : 'var(--bg-tertiary)',
                          color: isWinner ? '#ffffff' : 'var(--text-primary)'
                        }}
                      >
                        #{idx + 1}
                      </span>
                      <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>{m.name}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      {m.id === 'double_dqn' && (
                        <span style={{ fontSize: '0.65rem', background: 'var(--accent-primary)', color: '#fff', padding: '1px 5px', borderRadius: '4px', fontWeight: 600 }}>
                          Proposed
                        </span>
                      )}
                      {isSelected && (
                        <span style={{ fontSize: '0.65rem', background: 'var(--accent-primary)', color: '#fff', padding: '1px 5px', borderRadius: '4px', fontWeight: 700 }}>
                          Active
                        </span>
                      )}
                    </div>
                  </div>

                  <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Total Wait</div>
                      <div style={{ fontSize: '1rem', fontWeight: 800 }}>{m.totalWait} units</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Reduction</div>
                      <div
                        style={{
                          fontSize: '0.9rem',
                          fontWeight: 700,
                          color: m.improvement > 0 ? '#10b981' : m.improvement < 0 ? '#ef4444' : 'var(--text-muted)'
                        }}
                      >
                        {m.improvement > 0 ? `+${m.improvement}%` : `${m.improvement}%`}
                      </div>
                    </div>
                  </div>

                  {m.configuredEpisodes && (
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                      Configured: {m.configuredEpisodes} episodes
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recharts Data Analytics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '24px' }}>
        {/* Real-time Queue Load Over Time Chart */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">Store Queue Load Over Time</span>
            </div>
            <span className="badge badge-success">Benchmark (First 50 Steps)</span>
          </div>
          <div className="card-body" style={{ height: '320px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeSeriesData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorRandom" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorOpt" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" opacity={0.6} />
                <XAxis dataKey="step" stroke="var(--text-muted)" fontSize={12} label={{ value: 'Simulation Step', position: 'insideBottom', offset: -5 }} />
                <YAxis stroke="var(--text-muted)" fontSize={12} label={{ value: 'Total Queue', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--bg-secondary)',
                    borderColor: 'var(--border-color)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)'
                  }}
                />
                <Legend verticalAlign="top" height={36} />
                <Area
                  type="monotone"
                  dataKey="randomTotal"
                  name="Random Baseline"
                  stroke="#ef4444"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorRandom)"
                  strokeDasharray="4 4"
                />
                <Area
                  type="monotone"
                  dataKey="optTotal"
                  name="Active RL Agent"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#colorOpt)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Current Lane Occupancy Bar Chart */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">Live Lane Queue Occupancy & Counter Status</span>
            </div>
            <span className="badge badge-info">Capacity vs Active Load</span>
          </div>
          <div className="card-body" style={{ height: '320px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={laneBarData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" opacity={0.6} />
                <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} />
                <YAxis domain={[0, maxQueueLen]} stroke="var(--text-muted)" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--bg-secondary)',
                    borderColor: 'var(--border-color)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)'
                  }}
                />
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="current" name="Active Customers Waiting" fill="#f05136" radius={[6, 6, 0, 0]} />
                <Bar dataKey="capacity" name={`Maximum Lane Capacity (${maxQueueLen})`} fill="#d5ccc0" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Operational Intelligence Card */}
      <div className="glass-card">
        <div className="card-header">
          <div className="card-title-group">
            <Sparkles size={18} color="#10b981" />
            <span className="card-title">Reinforcement Learning Architecture Spectrum</span>
          </div>
          <span className="badge badge-success">
            {leaderboard.length ? `${leaderboard.length} Policies Evaluated` : '9 RL Architectures Spectrum'}
          </span>
        </div>
        <div className="card-body" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '20px' }}>
          <div style={{ display: 'flex', gap: '12px' }}>
            <CheckCircle2 size={20} color="#10b981" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Value-Based: Q-Learning & SARSA</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Q-Learning learns optimal off-policy values while SARSA accounts for exploration, learning safer routing policies near queue thresholds.
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <CheckCircle2 size={20} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Adaptive Baseline: Bandit (UCB1)</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Adaptive arm selection baseline utilizing Upper Confidence Bounds for immediate queue reward balancing without discount factor.
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <CheckCircle2 size={20} color="#6366f1" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Policy Gradients: REINFORCE, A2C, PPO</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Softmax preferences with learned baseline (REINFORCE), 1-step TD Actor-Critic (A2C), and clipped surrogate objectives (PPO) for high stability.
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <CheckCircle2 size={20} color="#0ea5e9" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Deep RL: DQN, Double DQN (Proposed), DDPG</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Neural function approximation with experience replay and decoupled greedy target evaluation (Double DQN) to eliminate maximization bias.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React from 'react';
import {
  User,
  CheckCircle2,
  AlertCircle,
  Zap,
  ArrowDown,
  UserPlus,
  Sparkles,
  ShoppingBag,
  Power,
  ShieldAlert,
  Repeat,
  Package
} from 'lucide-react';

const TYPE_CONFIG = {
  express: {
    label: 'Express Lane',
    desc: 'Basket ≤ 5 items only',
    color: '#10b981',
    colorBg: 'rgba(16, 185, 129, 0.12)',
    glow: 'rgba(16, 185, 129, 0.35)',
    maxBasket: 5
  },
  standard: {
    label: 'Standard Lane',
    desc: 'Normal carts (≤ 15 items)',
    color: '#3b82f6',
    colorBg: 'rgba(59, 130, 246, 0.12)',
    glow: 'rgba(59, 130, 246, 0.35)',
    maxBasket: 15
  },
  large_basket: {
    label: 'Large Basket',
    desc: 'High-volume carts & bulk',
    color: '#a855f7',
    colorBg: 'rgba(168, 85, 247, 0.12)',
    glow: 'rgba(168, 85, 247, 0.35)',
    maxBasket: 30
  },
  flexible: {
    label: 'Flexible Lane',
    desc: 'Dynamic demand convert',
    color: '#f59e0b',
    colorBg: 'rgba(245, 158, 11, 0.12)',
    glow: 'rgba(245, 158, 11, 0.35)',
    maxBasket: 15
  }
};

const METRIC_LABELS = {
  q_learning: 'Q*(s, a)',
  sarsa: 'Q_sarsa(s, a)',
  bandit: 'UCB1 Score',
  reinforce: 'Pref H(s, a)',
  a2c: 'Actor Adv',
  ppo: 'PPO Utility',
  dqn: 'Q_dqn(s, a)',
  double_dqn: 'Q_ddqn(s, a)',
  ddpg: 'Q_critic',
  shortest_queue: 'Heuristic',
  random: 'Uniform Prob',
  manual: 'Manual'
};

export default function CheckoutLanesView({
  queues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  maxQueueLen = 6,
  serviceRates = [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5],
  openStatus = [true, true, true, true, false, false, true, false, true, false],
  laneTypes = ['express', 'express', 'standard', 'standard', 'standard', 'standard', 'large_basket', 'large_basket', 'flexible', 'flexible'],
  capacities = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
  currentServing = [],
  incomingBasketSize = 5,
  isExpressEligible = true,
  counterUtilization = [],
  lastAction = null,
  lastServed = [],
  qValues = [],
  recommendedAction = 0,
  mode = 'double_dqn',
  onManualDispatch,
  onManualAction
}) {
  const metricLabel = METRIC_LABELS[mode] || 'Value(s, a)';

  // Build 10 lane items
  const lanes = Array.from({ length: 10 }).map((_, id) => {
    const lType = (laneTypes && laneTypes[id]) ? laneTypes[id] : 'standard';
    const conf = TYPE_CONFIG[lType] || TYPE_CONFIG.standard;
    const rate = (serviceRates && serviceRates[id] !== undefined) ? serviceRates[id] : 0.5;
    const cap = (capacities && capacities[id] !== undefined) ? capacities[id] : maxQueueLen;

    return {
      id,
      name: `Lane ${id}`,
      type: lType,
      typeLabel: conf.label,
      desc: conf.desc,
      color: conf.color,
      colorBg: conf.colorBg,
      glow: conf.glow,
      rate,
      capacity: cap,
      maxBasket: conf.maxBasket
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Incoming Customer Arrival Banner */}
      <div
        className="glass-card"
        style={{
          padding: '14px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          borderLeft: '4px solid var(--accent-primary)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '8px',
              background: 'var(--accent-primary-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-primary)'
            }}
          >
            <ShoppingBag size={22} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontWeight: 800, fontSize: '1rem' }}>Incoming Customer</span>
              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: '4px',
                  background: isExpressEligible ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                  color: isExpressEligible ? '#10b981' : '#d97706',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                {isExpressEligible ? 'Express Eligible (≤ 5 items)' : 'General Cart (> 5 items)'}
              </span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Basket Size: <strong>{incomingBasketSize} items</strong> • System Setup: 10 Dynamically Managed Lanes
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Agent Recommended Action:</span>
          <span className="badge badge-success" style={{ fontSize: '0.85rem', fontWeight: 800 }}>
            {recommendedAction < 10
              ? `Action ${recommendedAction} (Route Lane ${recommendedAction})`
              : recommendedAction < 20
              ? `Action ${recommendedAction} (Open Lane ${recommendedAction - 10})`
              : recommendedAction < 30
              ? `Action ${recommendedAction} (Close Lane ${recommendedAction - 20})`
              : recommendedAction < 40
              ? `Action ${recommendedAction} (Convert Lane ${recommendedAction - 30})`
              : `Action 40 (Hold)`}
          </span>
        </div>
      </div>

      {/* 10 Lanes Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '16px'
        }}
      >
        {lanes.map((lane) => {
          const count = queues[lane.id] ?? 0;
          const isOpen = openStatus[lane.id] !== false;
          const isTargetRoute = lastAction === lane.id;
          const isTargetOpen = lastAction === 10 + lane.id;
          const isTargetClose = lastAction === 20 + lane.id;
          const isTargetConvert = lastAction === 30 + lane.id;
          const isTarget = isTargetRoute || isTargetOpen || isTargetClose || isTargetConvert;

          const isServed = lastServed && lastServed[lane.id];
          const isRecommended = recommendedAction === lane.id;
          const qVal = qValues && qValues[lane.id] !== undefined ? qValues[lane.id] : 0;
          const isFull = count >= lane.capacity;
          const utilPct = Math.round(((counterUtilization && counterUtilization[lane.id]) ?? 0) * 100);
          const servingCust = currentServing && currentServing[lane.id];

          // Basket compatibility
          const isBasketCompatible =
            lane.type === 'express' ? incomingBasketSize <= 5 : true;

          return (
            <div
              key={lane.id}
              className={`glass-card ${isTarget ? 'active-target' : ''}`}
              style={{
                padding: '16px',
                borderRadius: '10px',
                border: isTarget
                  ? `2px solid ${lane.color}`
                  : isRecommended && isOpen
                  ? `2px solid ${lane.color}88`
                  : '1px solid var(--border-color)',
                boxShadow: isTarget && isOpen ? `0 0 20px ${lane.glow}` : undefined,
                opacity: isOpen ? 1 : 0.72,
                background: isOpen ? '#ffffff' : '#f8fafc',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px'
              }}
            >
              {/* Lane Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 800 }}>
                      {lane.name}
                    </h3>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        fontWeight: 700,
                        padding: '1px 6px',
                        borderRadius: '4px',
                        background: lane.colorBg,
                        color: lane.color
                      }}
                    >
                      {lane.type.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {lane.desc}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      fontWeight: 800,
                      padding: '2px 6px',
                      borderRadius: '10px',
                      background: isOpen ? 'rgba(16, 185, 129, 0.15)' : 'rgba(100, 116, 139, 0.15)',
                      color: isOpen ? '#10b981' : '#64748b',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                  >
                    <Power size={10} />
                    {isOpen ? 'OPEN' : 'CLOSED'}
                  </span>
                  <span
                    style={{
                      fontSize: '0.72rem',
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: 'var(--bg-tertiary)',
                      color: 'var(--text-secondary)'
                    }}
                  >
                    μ = {lane.rate}
                  </span>
                </div>
              </div>

              {/* Cashier Status / Currently Serving */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 10px', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
                <div
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    background: isOpen ? lane.colorBg : '#e2e8f0',
                    border: `1.5px solid ${isOpen ? lane.color : '#94a3b8'}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}
                >
                  <User size={18} color={isOpen ? lane.color : '#94a3b8'} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                      Util: <strong>{utilPct}%</strong>
                    </span>
                    {isServed ? (
                      <span style={{ color: '#10b981', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '2px' }}>
                        <CheckCircle2 size={12} /> Served
                      </span>
                    ) : servingCust ? (
                      <span style={{ color: '#3b82f6', fontWeight: 600 }}>
                        Scanning #{servingCust.id} ({servingCust.basket}b)
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>
                        {!isOpen ? 'Lane Closed' : count > 0 ? 'Busy' : 'Idle'}
                      </span>
                    )}
                  </div>
                  {/* Progress Bar */}
                  <div className="queue-progress-bar" style={{ marginTop: '4px', height: '6px' }}>
                    <div
                      className="queue-progress-fill"
                      style={{
                        width: `${(count / lane.capacity) * 100}%`,
                        backgroundColor: !isOpen ? '#94a3b8' : count >= 5 ? '#ef4444' : count >= 3 ? '#f59e0b' : lane.color
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Waiting Queue Tokens */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                  <span>Waiting: <strong style={{ color: 'var(--text-primary)' }}>{count}</strong> / {lane.capacity}</span>
                  {isFull && isOpen && <span style={{ color: '#ef4444', fontWeight: 700 }}>FULL</span>}
                </div>
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {Array.from({ length: lane.capacity }).map((_, slotIdx) => {
                    const isOcc = slotIdx < count;
                    return (
                      <div
                        key={slotIdx}
                        style={{
                          width: '18px',
                          height: '18px',
                          borderRadius: '4px',
                          backgroundColor: isOcc ? (isOpen ? lane.color : '#64748b') : 'transparent',
                          border: `1px solid ${isOcc ? (isOpen ? lane.color : '#64748b') : 'var(--border-color)'}`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                        title={isOcc ? `Customer ${slotIdx + 1}` : 'Empty'}
                      >
                        {isOcc && <User size={10} color="#ffffff" />}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Q-Value / Recommendation Telemetry */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '6px', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>{metricLabel}: <strong>{typeof qVal === 'number' ? qVal.toFixed(2) : qVal}</strong></span>
                {isRecommended && isOpen && (
                  <span className="badge badge-success" style={{ fontSize: '0.68rem', padding: '1px 5px' }}>
                    Recommended Pick
                  </span>
                )}
              </div>

              {/* Interactive Lane Controls */}
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '2px' }}>
                {/* Route Customer Button */}
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ flex: 1, padding: '4px 6px', fontSize: '0.72rem' }}
                  onClick={() => onManualAction ? onManualAction(lane.id) : (onManualDispatch && onManualDispatch(lane.id))}
                  disabled={!isOpen || isFull || !isBasketCompatible}
                  title={!isOpen ? 'Lane is closed' : isFull ? 'Lane is at capacity' : !isBasketCompatible ? 'Basket exceeds Express limit' : 'Route incoming customer to this lane'}
                >
                  <UserPlus size={12} />
                  Route
                </button>

                {/* Open / Close Toggle Button */}
                {!isOpen ? (
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '4px 8px', fontSize: '0.72rem', color: '#10b981' }}
                    onClick={() => onManualAction && onManualAction(10 + lane.id)}
                    title="Open this lane"
                  >
                    Open
                  </button>
                ) : (
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '4px 8px', fontSize: '0.72rem', color: '#ef4444' }}
                    onClick={() => onManualAction && onManualAction(20 + lane.id)}
                    disabled={count > 0 || servingCust !== null}
                    title={count > 0 || servingCust !== null ? 'Cannot close lane with customers' : 'Close lane safely'}
                  >
                    Close
                  </button>
                )}

                {/* Convert Type Button */}
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '4px 8px', fontSize: '0.72rem', color: '#f59e0b' }}
                  onClick={() => onManualAction && onManualAction(30 + lane.id)}
                  disabled={count > 0 || servingCust !== null}
                  title={count > 0 || servingCust !== null ? 'Cannot convert busy lane' : 'Cycle lane type (Express/Standard/Large/Flexible)'}
                >
                  <Repeat size={11} />
                  Type
                </button>
              </div>

              {/* Action Applied Feedback */}
              {isTarget && (
                <div
                  style={{
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    padding: '3px 6px',
                    borderRadius: '4px',
                    background: lane.colorBg,
                    color: lane.color,
                    textAlign: 'center',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px'
                  }}
                >
                  <ArrowDown size={12} />
                  {isTargetRoute ? 'Customer Assigned Here' : isTargetOpen ? 'Lane Opened' : isTargetClose ? 'Lane Closed' : 'Lane Type Converted'}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

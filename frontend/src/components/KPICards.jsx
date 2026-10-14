import React from 'react';
import {
  TrendingDown,
  Users,
  Timer,
  Scale,
  Zap,
  Award
} from 'lucide-react';

export default function KPICards({
  benchmarkData,
  currentQueues,
  serviceRates,
  activeModelId = 'q_learning'
}) {
  const randWait = benchmarkData?.randomTotalWait ?? 381;
  const randAvgWait = benchmarkData?.randomAvgWaitPerStep ?? (randWait / 100).toFixed(2);
  const randVariance = benchmarkData?.randomAvgVariance ?? 1.25;

  // Resolve active model from leaderboard or models dictionary
  const leaderboardItem = benchmarkData?.leaderboard?.find((m) => m.id === activeModelId);
  const modelData = benchmarkData?.models?.[activeModelId];

  // Resolve model friendly name & tag
  let modelName = leaderboardItem?.name || modelData?.name;
  let modelTag = leaderboardItem?.tag || modelData?.family;

  if (!modelName) {
    if (activeModelId === 'shortest_queue') {
      modelName = 'Shortest Queue';
      modelTag = 'Greedy Heuristic';
    } else if (activeModelId === 'random') {
      modelName = 'Random Baseline';
      modelTag = 'Uniform';
    } else if (activeModelId === 'manual') {
      modelName = 'Manual Dispatch';
      modelTag = 'User Control';
    } else {
      modelName = 'RL Optimized';
      modelTag = '1-Step TD';
    }
  }

  let optWait;
  let reductionPct;
  let optAvgWait;
  let optVariance;

  if (activeModelId === 'random') {
    optWait = randWait;
    reductionPct = 0;
    optAvgWait = randAvgWait;
    optVariance = randVariance;
  } else if (activeModelId === 'shortest_queue') {
    optWait = leaderboardItem?.totalWait ?? benchmarkData?.heuristicTotalWait ?? 305;
    reductionPct = leaderboardItem?.improvement ?? (randWait > 0 ? Number(((randWait - optWait) / randWait * 100).toFixed(2)) : 19.95);
    optAvgWait = leaderboardItem?.avgWait ?? benchmarkData?.heuristicAvgWaitPerStep ?? (optWait / 100).toFixed(2);
    optVariance = leaderboardItem?.variance ?? 0.55;
  } else if (leaderboardItem || modelData) {
    optWait = leaderboardItem?.totalWait ?? modelData?.totalWait ?? benchmarkData?.optimizedTotalWait ?? 268;
    reductionPct = leaderboardItem?.improvement ?? modelData?.congestionReductionPct ?? benchmarkData?.congestionReductionPct ?? 29.66;
    optAvgWait = leaderboardItem?.avgWait ?? modelData?.avgWaitPerStep ?? benchmarkData?.optAvgWaitPerStep ?? (optWait / 100).toFixed(2);
    optVariance = leaderboardItem?.variance ?? modelData?.avgVariance ?? benchmarkData?.optAvgVariance ?? 0.38;
  } else {
    optWait = benchmarkData?.optimizedTotalWait ?? 268;
    reductionPct = benchmarkData?.congestionReductionPct ?? 29.66;
    optAvgWait = benchmarkData?.optAvgWaitPerStep ?? (optWait / 100).toFixed(2);
    optVariance = benchmarkData?.optAvgVariance ?? 0.38;
  }

  const numericReduction = Number(reductionPct);
  const waitDiff = randWait - optWait;

  const cards = [
    {
      label: 'Congestion Reduction',
      value: `${numericReduction.toFixed(2)}%`,
      unit: 'Improvement',
      subtext: waitDiff >= 0 ? `+${waitDiff} wait-units prevented vs random` : `${waitDiff} wait-units vs random`,
      icon: TrendingDown,
      accent: numericReduction >= 0 ? '#10b981' : '#ef4444',
      badge: modelName,
      badgeClass: numericReduction > 0 ? 'badge-success' : (numericReduction === 0 ? 'badge-info' : 'badge-danger')
    },
    {
      label: 'Total Customer Wait-Units',
      value: optWait,
      unit: 'Wait-Units',
      subtext: `Baseline Random: ${randWait} units (100 steps)`,
      icon: Timer,
      accent: '#3b82f6',
      badge: `${numericReduction >= 0 ? `-${numericReduction.toFixed(1)}%` : `+${Math.abs(numericReduction).toFixed(1)}%`} Time`,
      badgeClass: 'badge-info'
    },
    {
      label: 'Average Queue / Step',
      value: typeof optAvgWait === 'number' ? optAvgWait.toFixed(2) : optAvgWait,
      unit: 'Cust / Step',
      subtext: `Random load: ${randAvgWait} cust/step`,
      icon: Users,
      accent: '#8b5cf6',
      badge: Number(optAvgWait) <= Number(randAvgWait) ? 'High Flow' : 'Heavy Queue',
      badgeClass: 'badge-success'
    },
    {
      label: 'Cashier Workload Variance',
      value: typeof optVariance === 'number' ? optVariance.toFixed(3) : optVariance,
      unit: 'Var σ²',
      subtext: 'Lower variance = balanced cashier queues',
      icon: Scale,
      accent: '#f59e0b',
      badge: Number(optVariance) <= 0.65 ? 'Balanced' : 'Skewed',
      badgeClass: 'badge-warning'
    }
  ];

  return (
    <div className="kpi-grid">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className="glass-card kpi-card"
            style={{ '--card-accent': card.accent }}
          >
            <div className="kpi-top">
              <span className="kpi-label">{card.label}</span>
              <div className="kpi-icon-box">
                <Icon size={20} />
              </div>
            </div>

            <div className="kpi-value-row">
              <span className="kpi-value">{card.value}</span>
              <span className="kpi-unit">{card.unit}</span>
            </div>

            <div className="kpi-subtext">
              <span className={`badge ${card.badgeClass}`}>{card.badge}</span>
              <span>{card.subtext}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

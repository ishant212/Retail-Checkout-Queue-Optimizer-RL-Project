import React, { useState } from 'react';
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
import { Download, Search, FileSpreadsheet, Trophy, ArrowDownRight, TrendingDown, Clock, Users, DollarSign, Activity } from 'lucide-react';

export default function BenchmarkLogsView({ benchmarkData }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedActionFilter, setSelectedActionFilter] = useState('all');
  const [activeChartMetric, setActiveChartMetric] = useState('totalWait');

  const steps = benchmarkData?.stepDetails || [];
  const randTotalWait = benchmarkData?.randomTotalWait ?? 381;
  const optTotalWait = benchmarkData?.optimizedTotalWait ?? 268;
  const reductionPct = benchmarkData?.congestionReductionPct ?? 29.66;
  const leaderboard = benchmarkData?.leaderboard || [];

  const formatAction = (act) => {
    if (act === null || act === undefined) return 'Hold';
    if (typeof act === 'string') return act;
    const a = Number(act);
    if (a >= 0 && a < 10) return `Route L${a}`;
    if (a >= 10 && a < 20) return `Open L${a - 10}`;
    if (a >= 20 && a < 30) return `Close L${a - 20}`;
    if (a >= 30 && a < 40) return `Convert L${a - 30}`;
    if (a === 40) return 'Hold';
    return `Action ${a}`;
  };

  // Filter logs
  const filteredSteps = steps.filter((s) => {
    const matchesSearch = s.step.toString().includes(searchTerm);
    const matchesAction = selectedActionFilter === 'all' || s.optAction.toString() === selectedActionFilter;
    return matchesSearch && matchesAction;
  });

  // Export CSV
  const exportCSV = () => {
    if (!steps.length) return;
    const headers = 'Step,Random_Lanes,Random_Total,Random_Action,Opt_Lanes,Opt_Total,Opt_Action,Savings\n';
    const rows = steps.map((s) =>
      `"${s.step}","[${s.randomLanes.join(';')}]",${s.randomTotal},"${formatAction(s.randomAction)}","[${s.optLanes.join(';')}]",${s.optTotal},"${formatAction(s.optAction)}",${s.waitDifference}`
    ).join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optiqueue_10lane_benchmark_100steps.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const chartData = leaderboard.map((m) => ({
    name: m.name,
    totalWait: m.totalWait ?? 0,
    congestionDuration: m.congestionDuration ?? 0,
    customersServed: m.customersServed ?? 0,
    operatingCost: m.operatingCost ?? 0,
    cumulativeReward: m.cumulativeReward ?? 0,
    improvement: m.improvement ?? 0
  }));

  const METRIC_CONFIG = {
    totalWait: { label: 'Total Customer Wait', color: '#f05136', unit: 'wait-units' },
    congestionDuration: { label: 'Congestion Duration', color: '#ef4444', unit: 'steps' },
    customersServed: { label: 'Customers Served', color: '#10b981', unit: 'served' },
    operatingCost: { label: 'Operating Cost', color: '#d97706', unit: '$' },
    cumulativeReward: { label: 'Cumulative Reward', color: '#6366f1', unit: 'reward' }
  };

  const activeConf = METRIC_CONFIG[activeChartMetric] || METRIC_CONFIG.totalWait;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Benchmark Summary Card */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <FileSpreadsheet size={22} color="#10b981" />
              <h2 style={{ fontSize: '1.25rem' }}>100-Timestep Multi-Model Benchmark Arena</h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
              Side-by-side empirical evaluation across all 9 Reinforcement Learning algorithms and heuristic baselines under identical seeds.
            </p>
          </div>

          <button className="btn btn-secondary" onClick={exportCSV} disabled={!steps.length}>
            <Download size={16} />
            Export Benchmark CSV
          </button>
        </div>

        {/* Metric Cards Row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginTop: '20px' }}>
          <div style={{ padding: '16px', borderRadius: '10px', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Random Baseline Wait</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#ef4444', marginTop: '4px' }}>
              {randTotalWait} <span style={{ fontSize: '0.8rem', fontWeight: 400 }}>units</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Avg {benchmarkData?.randomAvgWaitPerStep ?? 3.81} / step
            </div>
          </div>

          <div style={{ padding: '16px', borderRadius: '10px', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Top RL Optimizer Wait</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#10b981', marginTop: '4px' }}>
              {leaderboard[0]?.totalWait ?? optTotalWait} <span style={{ fontSize: '0.8rem', fontWeight: 400 }}>units</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Leader: {leaderboard[0]?.name ?? 'Double DQN'}
            </div>
          </div>

          <div style={{ padding: '16px', borderRadius: '10px', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Max Congestion Reduction</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#10b981', marginTop: '4px' }}>
              {leaderboard[0]?.improvement ? `+${leaderboard[0].improvement}%` : `+${reductionPct}%`}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              vs Baseline Random Dispatch
            </div>
          </div>

          <div style={{ padding: '16px', borderRadius: '10px', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Models Evaluated</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-color)', marginTop: '4px' }}>
              {leaderboard.length || 11} <span style={{ fontSize: '0.8rem', fontWeight: 400 }}>policies</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              {leaderboard.length ? `${leaderboard.length - 2} RL + 2 Baselines` : '9 RL + 2 Baselines'}
            </div>
          </div>
        </div>
      </div>

      {/* Leaderboard Table & Chart Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '24px' }}>
        {/* Leaderboard Table */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <Trophy size={18} color="#f59e0b" />
              <span className="card-title">Multi-Model Performance Ranking (All 11 Policies)</span>
            </div>
            <span className="badge badge-success">Ranked by Total Wait</span>
          </div>

          <div className="card-body" style={{ padding: '0' }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-tertiary)', borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '10px 12px' }}>Rank</th>
                    <th style={{ padding: '10px 12px' }}>Algorithm</th>
                    <th style={{ padding: '10px 12px' }}>Type</th>
                    <th style={{ padding: '10px 12px' }}>Total Wait</th>
                    <th style={{ padding: '10px 12px' }}>Congestion Dur</th>
                    <th style={{ padding: '10px 12px' }}>Served</th>
                    <th style={{ padding: '10px 12px' }}>Cost</th>
                    <th style={{ padding: '10px 12px' }}>Reward</th>
                    <th style={{ padding: '10px 12px' }}>Reduction</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((m, idx) => {
                    const isWinner = idx === 0;
                    return (
                      <tr key={m.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 800 }}>
                          <span style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 22,
                            height: 22,
                            borderRadius: '50%',
                            background: isWinner ? '#f59e0b' : 'var(--bg-tertiary)',
                            color: isWinner ? '#000' : 'var(--text-primary)',
                            fontSize: '0.72rem'
                          }}>
                            #{idx + 1}
                          </span>
                        </td>
                        <td style={{ padding: '10px 12px', fontWeight: 700 }}>{m.name}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--text-muted)', fontSize: '0.74rem' }}>{m.tag}</td>
                        <td style={{ padding: '10px 12px', fontWeight: 700, fontFamily: 'monospace' }}>{m.totalWait}</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: (m.congestionDuration ?? 0) > 0 ? '#ef4444' : '#10b981' }}>
                          {m.congestionDuration ?? 0}s
                        </td>
                        <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#10b981' }}>{m.customersServed ?? '-'}</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>${m.operatingCost ?? '-'}</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: (m.cumulativeReward ?? 0) >= 0 ? '#10b981' : '#ef4444' }}>
                          {m.cumulativeReward ?? '-'}
                        </td>
                        <td style={{
                          padding: '10px 12px',
                          fontWeight: 700,
                          color: m.improvement > 0 ? '#10b981' : m.improvement < 0 ? '#ef4444' : 'var(--text-muted)'
                        }}>
                          {m.improvement > 0 ? `+${m.improvement}%` : `${m.improvement}%`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Multi-Metric Comparison Bar Chart */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">{activeConf.label} by Algorithm</span>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {[
                { id: 'totalWait', label: 'Wait' },
                { id: 'congestionDuration', label: 'Congestion' },
                { id: 'customersServed', label: 'Served' },
                { id: 'operatingCost', label: 'Cost' },
                { id: 'cumulativeReward', label: 'Reward' }
              ].map((btn) => (
                <button
                  key={btn.id}
                  onClick={() => setActiveChartMetric(btn.id)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: '4px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    border: '1px solid var(--border-color)',
                    background: activeChartMetric === btn.id ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                    color: activeChartMetric === btn.id ? '#ffffff' : 'var(--text-primary)',
                    cursor: 'pointer'
                  }}
                >
                  {btn.label}
                </button>
              ))}
            </div>
          </div>

          <div className="card-body" style={{ height: '340px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 20, left: 40, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" opacity={0.6} />
                <XAxis type="number" stroke="var(--text-muted)" fontSize={12} />
                <YAxis dataKey="name" type="category" stroke="var(--text-muted)" fontSize={11} width={120} />
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
                <Bar dataKey={activeChartMetric} name={activeConf.label} fill={activeConf.color} radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Step-by-Step Transition Log Table */}
      <div className="glass-card">
        <div className="card-header">
          <div className="card-title-group">
            <span className="card-title">Detailed Step-by-Step Transition Logs ({filteredSteps.length} Steps)</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ position: 'relative' }}>
              <Search size={15} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search step #..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  padding: '6px 12px 6px 32px',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem'
                }}
              />
            </div>
          </div>
        </div>

        <div className="card-body" style={{ padding: 0 }}>
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
              <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-tertiary)', zIndex: 10 }}>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-primary)', fontWeight: 700 }}>
                  <th style={{ padding: '10px 14px' }}>Step</th>
                  <th style={{ padding: '10px 14px' }}>Random Queue [L0..L9]</th>
                  <th style={{ padding: '10px 14px' }}>Random Action</th>
                  <th style={{ padding: '10px 14px' }}>RL Queue [L0..L9]</th>
                  <th style={{ padding: '10px 14px' }}>RL Action</th>
                  <th style={{ padding: '10px 14px' }}>Wait Difference</th>
                </tr>
              </thead>
              <tbody>
                {filteredSteps.slice(0, 100).map((s) => (
                  <tr key={s.step} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '10px 14px', fontWeight: 600 }}>#{s.step}</td>
                    <td style={{ padding: '10px 14px', fontFamily: 'monospace' }}>
                      [{s.randomLanes.join(', ')}] = {s.randomTotal}
                    </td>
                    <td style={{ padding: '10px 14px', color: '#ef4444', fontWeight: 600 }}>
                      {formatAction(s.randomAction)}
                    </td>
                    <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#10b981', fontWeight: 600 }}>
                      [{s.optLanes.join(', ')}] = {s.optTotal}
                    </td>
                    <td style={{ padding: '10px 14px', color: '#10b981', fontWeight: 600 }}>
                      {formatAction(s.optAction)}
                    </td>
                    <td style={{ padding: '10px 14px', fontWeight: 700, color: s.waitDifference >= 0 ? '#10b981' : '#ef4444' }}>
                      {s.waitDifference >= 0 ? `+${s.waitDifference}` : s.waitDifference}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

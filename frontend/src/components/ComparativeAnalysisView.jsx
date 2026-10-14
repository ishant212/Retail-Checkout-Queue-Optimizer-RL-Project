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
  Download,
  RefreshCw,
  Zap,
  TrendingDown,
  Users,
  DollarSign,
  Layers,
  Scale,
  Sparkles,
  AlertTriangle,
  Flame,
  Clock
} from 'lucide-react';
import { api } from '../api';

const MODEL_PALETTE = {
  double_dqn: '#f05136',       // FACE Prep Terracotta Coral (Proposed)
  dqn: '#0284c7',              // Sky Blue
  ppo: '#7c3aed',              // Purple
  a2c: '#4f46e5',              // Indigo
  bandit: '#d97706',           // Amber
  q_learning: '#16a34a',       // Green
  sarsa: '#0d9488',            // Teal
  reinforce: '#ea580c',        // Orange
  ddpg: '#db2777',             // Rose
  min_estimated_wait: '#2563eb',// Blue
  shortest_queue: '#0891b2',   // Cyan
  rule_based: '#dc2626'        // Red (Baseline)
};

export default function ComparativeAnalysisView() {
  const [data, setData] = useState(null);
  const [selectedScenario, setSelectedScenario] = useState('normal');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Active models selected for Queue Length Over Time line chart
  const [selectedTimeModels, setSelectedTimeModels] = useState([
    'double_dqn',
    'dqn',
    'ppo',
    'shortest_queue',
    'rule_based'
  ]);

  const loadData = async (scenario = selectedScenario) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.getComparativeAnalysis(scenario);
      setData(res);
    } catch (err) {
      console.error('Comparative analysis fetch error:', err);
      setError('Could not load comparative analysis data from Python backend.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData(selectedScenario);
  }, [selectedScenario]);

  const handleScenarioChange = (scenId) => {
    setSelectedScenario(scenId);
  };

  const toggleTimeModel = (modelId) => {
    setSelectedTimeModels((prev) =>
      prev.includes(modelId)
        ? (prev.length > 1 ? prev.filter((m) => m !== modelId) : prev)
        : [...prev, modelId]
    );
  };

  const exportTableCSV = () => {
    if (!data?.tableIV) return;
    const headers = 'Strategy / Model,Avg Wait Time (min),Active Counters,Cumulative Reward,Reported Wait Improvement,Throughput (cust/min),Hourly Staff Cost ($)\n';
    const rows = data.tableIV.map((r) =>
      `"${r.strategy}","${r.avgWaitTimeDisplay}","${r.activeCountersDisplay}","${r.cumulativeRewardDisplay}","${r.reportedWaitImprovement}",${r.customerThroughput},${r.hourlyStaffCost}`
    ).join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TABLE_IV_Comparative_Analysis_${selectedScenario}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const tableIV = data?.tableIV || [];
  const scenarioInfo = data?.scenario || {};
  const allScenarios = data?.allScenarios || [];
  const queueOverTime = data?.queueOverTime || [];
  const staffCostData = data?.staffCostData || [];
  const throughputData = data?.throughputData || [];
  const waitTimeData = data?.waitTimeData || [];

  const proposedModel = tableIV.find((r) => r.isProposed) || tableIV[0];
  const baselineModel = tableIV.find((r) => r.id === 'rule_based') || tableIV[tableIV.length - 1];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Page Title & Controls */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{
                background: 'var(--accent-primary-light)',
                color: 'var(--accent-primary)',
                padding: '8px',
                borderRadius: '8px'
              }}>
                <Scale size={22} />
              </div>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0 }}>
                Comparative Model & Strategy Analysis
              </h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', marginTop: '6px', fontSize: '0.88rem' }}>
              Comprehensive empirical evaluation comparing {tableIV.length || 12} allocation methods (Baselines, Heuristics, Tabular RL, Actor-Critic, & Deep RL).
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              className="btn btn-outline"
              onClick={exportTableCSV}
              disabled={isLoading || !tableIV.length}
              title="Download IEEE/ACM Formatted Table CSV"
            >
              <Download size={16} />
              <span>Export Table IV (CSV)</span>
            </button>
            <button
              className="btn btn-primary"
              onClick={() => loadData(selectedScenario)}
              disabled={isLoading}
            >
              <RefreshCw size={16} className={isLoading ? 'spin-icon' : ''} />
              <span>{isLoading ? 'Simulating...' : 'Re-run Evaluation'}</span>
            </button>
          </div>
        </div>

        {/* Dimension 5: Performance Under Stress Scenarios Selector */}
        <div style={{ marginTop: '24px', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', marginBottom: '12px' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Stress Scenario Environment:
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', fontWeight: 600 }}>
              λ = {scenarioInfo.arrivalRate} cust/sec | μ = [{scenarioInfo.serviceRates?.join(', ')}] | {scenarioInfo.duration}s window
            </span>
          </div>

          <div className="scenario-selector-bar">
            {allScenarios.map((scen) => {
              const isActive = selectedScenario === scen.id;
              return (
                <button
                  key={scen.id}
                  className={`scenario-pill ${isActive ? 'active' : ''}`}
                  onClick={() => handleScenarioChange(scen.id)}
                  disabled={isLoading}
                >
                  {scen.id === 'rush_hour' && <Flame size={15} color={isActive ? '#fff' : '#f97316'} />}
                  {scen.id === 'cashier_fatigue' && <AlertTriangle size={15} color={isActive ? '#fff' : '#f59e0b'} />}
                  {scen.id === 'flash_surge' && <Zap size={15} color={isActive ? '#fff' : '#06b6d4'} />}
                  {scen.id === 'normal' && <Clock size={15} color={isActive ? '#fff' : '#10b981'} />}
                  <span>{scen.label}</span>
                </button>
              );
            })}
          </div>

          {scenarioInfo.description && (
            <div style={{
              background: 'var(--bg-tertiary)',
              borderRadius: '8px',
              padding: '10px 16px',
              fontSize: '0.84rem',
              color: 'var(--text-secondary)',
              marginTop: '10px'
            }}>
              <strong style={{ color: 'var(--text-primary)' }}>Scenario Profile:</strong> {scenarioInfo.description}
            </div>
          )}
        </div>
      </div>

      {/* Highlights KPI Cards */}
      {proposedModel && (
        <div className="kpi-grid">
          <div className="glass-card kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">
                {proposedModel?.strategy ? `${proposedModel.strategy.replace(' Manager', '')} Wait` : 'Proposed Model Wait'}
              </span>
              <div className="kpi-icon-wrapper" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }}>
                <Clock size={20} />
              </div>
            </div>
            <div className="kpi-value" style={{ color: '#10b981' }}>
              {proposedModel.avgWaitTimeDisplay} min
            </div>
            <div className="kpi-footer">
              <span style={{ color: '#10b981', fontWeight: 600 }}>
                {proposedModel.reportedWaitImprovement} vs Baseline
              </span>
            </div>
          </div>

          <div className="glass-card kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">Active Counters (Proposed)</span>
              <div className="kpi-icon-wrapper" style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6' }}>
                <Layers size={20} />
              </div>
            </div>
            <div className="kpi-value">
              {proposedModel.activeCountersDisplay}
            </div>
            <div className="kpi-footer">
              <span style={{ color: 'var(--text-secondary)' }}>
                Baseline Active: {baselineModel?.activeCountersDisplay || '3.12 ± 0.15'}
              </span>
            </div>
          </div>

          <div className="glass-card kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">Throughput Capacity</span>
              <div className="kpi-icon-wrapper" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6' }}>
                <Users size={20} />
              </div>
            </div>
            <div className="kpi-value" style={{ color: '#8b5cf6' }}>
              {proposedModel.customerThroughput}
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginLeft: '6px' }}>cust/min</span>
            </div>
            <div className="kpi-footer">
              <span style={{ color: 'var(--text-secondary)' }}>
                {proposedModel.totalCustomersServed} checkouts served
              </span>
            </div>
          </div>

          <div className="glass-card kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">Labor Cost Efficiency</span>
              <div className="kpi-icon-wrapper" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }}>
                <DollarSign size={20} />
              </div>
            </div>
            <div className="kpi-value">
              ${proposedModel.hourlyStaffCost}
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginLeft: '6px' }}>/hr</span>
            </div>
            <div className="kpi-footer">
              <span style={{ color: '#10b981', fontWeight: 600 }}>
                ${proposedModel.staffCostPerCustomer} / checkout
              </span>
            </div>
          </div>
        </div>
      )}

      {/* TABLE IV - COMPARATIVE ANALYSIS ACROSS STRATEGIES */}
      {/* Academic Paper Table exact replica of uploaded image */}
      <div className="academic-paper-card">
        <div className="paper-caption">
          TABLE IV - COMPARATIVE ANALYSIS ACROSS STRATEGIES
        </div>

        <div className="table-iv-wrapper">
          <table className="table-iv">
            <thead>
              <tr>
                <th style={{ width: '32%' }}>Strategy / Model</th>
                <th style={{ width: '20%', textAlign: 'center' }}>Avg Wait Time (min)</th>
                <th style={{ width: '18%', textAlign: 'center' }}>Active Counters</th>
                <th style={{ width: '18%', textAlign: 'center' }}>Cumulative Reward</th>
                <th style={{ width: '12%', textAlign: 'center' }}>Reported Wait Improvement</th>
              </tr>
            </thead>
            <tbody>
              {tableIV.map((row) => {
                const isProposed = row.isProposed;
                const isBaseline = row.id === 'rule_based';
                const improvementNum = row.reportedWaitImprovementPct;
                const isPositive = improvementNum > 0;

                return (
                  <tr
                    key={row.id}
                    className={isProposed ? 'proposed-row' : ''}
                  >
                    <td style={{ fontWeight: isProposed ? 700 : 500 }}>
                      <span>{row.strategy}</span>
                      {isProposed && (
                        <span className="table-badge-proposed">Proposed</span>
                      )}
                    </td>
                    <td className="num-cell">
                      {row.avgWaitTimeDisplay}
                    </td>
                    <td className="num-cell">
                      {row.activeCountersDisplay}
                    </td>
                    <td className="num-cell">
                      {row.cumulativeRewardDisplay}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {isBaseline ? (
                        <span className="table-badge-baseline">Baseline</span>
                      ) : isPositive ? (
                        <span className="table-badge-positive">{row.reportedWaitImprovement}</span>
                      ) : (
                        <span className="table-badge-negative">{row.reportedWaitImprovement}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          marginTop: '12px',
          fontSize: '0.78rem',
          color: 'var(--text-muted)'
        }}>
          <span>Values represent mean ± standard error evaluated across 3 randomized trial seeds (SEEDS=[42, 101, 777]).</span>
          <span>Baseline: Rule-Based Random Manager</span>
        </div>
      </div>

      {/* Dimension 1: Queue Length over Time */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingDown size={20} color="var(--accent-primary)" />
              Queue Length Over Time
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.84rem', marginTop: '4px' }}>
              Chronological queue dynamics comparing queue buildup and dissipation across simulation duration ($t \in [0, {scenarioInfo.duration}s]$).
            </p>
          </div>

          {/* Model Toggle Chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {tableIV.map((m) => {
              const isSelected = selectedTimeModels.includes(m.id);
              const color = MODEL_PALETTE[m.id] || '#9ca3af';
              return (
                <button
                  key={m.id}
                  onClick={() => toggleTimeModel(m.id)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '16px',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    border: `1px solid ${isSelected ? color : 'var(--border-color)'}`,
                    background: isSelected ? `${color}20` : 'transparent',
                    color: isSelected ? color : 'var(--text-muted)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px'
                  }}
                >
                  <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: color
                  }} />
                  {m.strategy.replace(' Manager', '').replace(' (Proposed)', '')}
                </button>
              );
            })}
          </div>
        </div>

        <div style={{ width: '100%', height: 350 }}>
          <ResponsiveContainer>
            <LineChart data={queueOverTime} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis
                dataKey="time"
                stroke="var(--text-secondary)"
                tick={{ fontSize: 11 }}
                label={{ value: 'Simulation Time (seconds)', position: 'insideBottom', offset: -10, fill: 'var(--text-muted)', fontSize: 12 }}
              />
              <YAxis
                stroke="var(--text-secondary)"
                tick={{ fontSize: 11 }}
                label={{ value: 'Total Queue Length', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  boxShadow: 'var(--shadow-md)',
                  fontSize: '0.82rem',
                  color: 'var(--text-primary)'
                }}
                itemStyle={{ color: 'var(--text-primary)' }}
                labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
              />
              <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '0.82rem' }} />

              {selectedTimeModels.map((modelId) => {
                const strat = tableIV.find((t) => t.id === modelId);
                const color = MODEL_PALETTE[modelId] || '#9ca3af';
                const isProposed = strat?.isProposed ?? (modelId === 'double_dqn');
                const isBaseline = modelId === 'rule_based';

                return (
                  <Line
                    key={modelId}
                    type="monotone"
                    dataKey={modelId}
                    name={strat?.strategy?.replace(' (Proposed)', '') || modelId}
                    stroke={color}
                    strokeWidth={isProposed ? 3 : isBaseline ? 2 : 1.8}
                    strokeDasharray={isBaseline ? '5 5' : undefined}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid of Section 2: Staff Cost & Section 3: Customer Throughput */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '24px' }}>
        {/* Dimension 2: Staff Cost Comparison */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <div style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b', padding: '6px', borderRadius: '6px' }}>
              <DollarSign size={18} />
            </div>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>
                Staff Cost Comparison
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', margin: 0 }}>
                Hourly labor expense based on counter utilization ($18.50 base rate, 3 counters).
              </p>
            </div>
          </div>

          <div style={{ width: '100%', height: 300, marginTop: '16px' }}>
            <ResponsiveContainer>
              <BarChart data={staffCostData} margin={{ top: 10, right: 10, left: -10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis
                  dataKey="name"
                  stroke="var(--text-secondary)"
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11 }}
                  label={{ value: 'Hourly Labor Cost ($)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }}
                />
                <Tooltip
                  formatter={(val, name) => [
                    name === 'hourlyStaffCost' ? `$${val}/hr` : `${val}¢`,
                    name === 'hourlyStaffCost' ? 'Hourly Labor Cost' : 'Cost per Customer'
                  ]}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    boxShadow: 'var(--shadow-md)',
                    fontSize: '0.82rem',
                    color: 'var(--text-primary)'
                  }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                  labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                />
                <Bar dataKey="hourlyStaffCost" fill="#d97706" radius={[4, 4, 0, 0]} name="Hourly Labor Cost" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Dimension 3: Customer Throughput Comparison */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <div style={{ background: '#ede9fe', color: '#7c3aed', padding: '6px', borderRadius: '6px' }}>
              <Users size={18} />
            </div>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>
                Customer Throughput Comparison
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', margin: 0 }}>
                Rate of completed checkouts (cust/min) served under identical arrival profiles.
              </p>
            </div>
          </div>

          <div style={{ width: '100%', height: 300, marginTop: '16px' }}>
            <ResponsiveContainer>
              <BarChart data={throughputData} margin={{ top: 10, right: 10, left: -10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis
                  dataKey="name"
                  stroke="var(--text-secondary)"
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11 }}
                  label={{ value: 'Throughput (cust/min)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }}
                />
                <Tooltip
                  formatter={(val) => [`${val} cust/min`, 'Customer Throughput']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    boxShadow: 'var(--shadow-md)',
                    fontSize: '0.82rem',
                    color: 'var(--text-primary)'
                  }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                  labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                />
                <Bar dataKey="throughput" fill="#7c3aed" radius={[4, 4, 0, 0]} name="Throughput (cust/min)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Dimension 4: Average Waiting Time by Strategy with Improvements */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Clock size={20} color="var(--accent-primary)" />
              Average Waiting Time by Strategy
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.84rem', marginTop: '4px' }}>
              Comparing average customer wait time (seconds) across all 12 strategies, ordered by congestion clearance performance.
            </p>
          </div>
          <span style={{ fontSize: '0.82rem', color: 'var(--accent-success)', fontWeight: 600 }}>
            Lower is better (Faster queue clearance)
          </span>
        </div>

        <div style={{ width: '100%', height: 320 }}>
          <ResponsiveContainer>
            <BarChart data={waitTimeData} margin={{ top: 10, right: 20, left: 10, bottom: 45 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis
                dataKey="name"
                stroke="var(--text-secondary)"
                angle={-35}
                textAnchor="end"
                interval={0}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                stroke="var(--text-secondary)"
                tick={{ fontSize: 11 }}
                label={{ value: 'Avg Wait Time (seconds)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
              />
              <Tooltip
                formatter={(val, name, item) => [
                  `${val}s (${item.payload.avgWaitMin} min) | Improvement: ${item.payload.improvement > 0 ? '+' : ''}${item.payload.improvement}%`,
                  'Avg Wait Time'
                ]}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  boxShadow: 'var(--shadow-md)',
                  fontSize: '0.82rem',
                  color: 'var(--text-primary)'
                }}
                itemStyle={{ color: 'var(--text-primary)' }}
                labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
              />
              <Bar
                dataKey="avgWaitSec"
                fill="#f05136"
                radius={[4, 4, 0, 0]}
                name="Avg Wait Time (s)"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Family Architecture Summary */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles size={18} color="var(--accent-primary)" />
          Taxonomy of Evaluated Reinforcement Learning & Heuristic Models
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div style={{ background: 'var(--bg-tertiary)', padding: '16px', borderRadius: '10px', borderLeft: '4px solid #10b981' }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '0.92rem', color: '#10b981' }}>
              Deep RL ({proposedModel?.strategy ? proposedModel.strategy.replace(' Manager', '') : 'Double DQN (Proposed)'})
            </h4>
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Decoupled action selection and Q-value evaluation eliminates maximization bias, yielding superior queue stability under non-stationary retail traffic.
            </p>
          </div>

          <div style={{ background: 'var(--bg-tertiary)', padding: '16px', borderRadius: '10px', borderLeft: '4px solid #8b5cf6' }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '0.92rem', color: '#8b5cf6' }}>Actor-Critic (PPO & A2C)</h4>
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Clipped surrogate objective in PPO prevents catastrophic policy updates. Synchronous A2C balances value estimation and policy variance.
            </p>
          </div>

          <div style={{ background: 'var(--bg-tertiary)', padding: '16px', borderRadius: '10px', borderLeft: '4px solid #3b82f6' }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '0.92rem', color: '#3b82f6' }}>Heuristic Benchmarks</h4>
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Shortest-Queue and Minimum Estimated Wait (q/μ) represent industry baselines, but lack predictive buffer anticipation for bursty surges.
            </p>
          </div>

          <div style={{ background: 'var(--bg-tertiary)', padding: '16px', borderRadius: '10px', borderLeft: '4px solid #f59e0b' }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '0.92rem', color: '#f59e0b' }}>Value-Based TD & Bandits</h4>
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Tabular Q-Learning, On-Policy SARSA, and UCB1 Bandits learn state-action values over discrete state tuples (q0, q1, q2).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

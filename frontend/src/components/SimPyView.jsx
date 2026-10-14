import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { Clock, Play, CheckCircle2, TrendingDown, Users, Percent, RefreshCw, Trophy } from 'lucide-react';
import { api } from '../api';

export default function SimPyView({ initialData }) {
  const [simpyResults, setSimpyResults] = useState(initialData);
  const [arrivalRate, setArrivalRate] = useState(1.0);
  const [simTime, setSimTime] = useState(120);
  const [isRunning, setIsRunning] = useState(false);

  const runSimPySimulation = async () => {
    setIsRunning(true);
    try {
      const data = await api.runSimPy({
        arrivalRate: Number(arrivalRate),
        simulationTime: Number(simTime)
      });
      setSimpyResults(data);
    } catch (err) {
      console.error('Failed to run SimPy simulation:', err);
    } finally {
      setIsRunning(false);
    }
  };

  const summary = simpyResults?.summary || [];
  const bestPolicy = simpyResults?.bestPolicy || (summary[0]?.label ?? 'Q-Learning');

  // Chart data for wait times
  const waitTimeChartData = summary.map((s) => ({
    policy: s.label.replace(' (Off-Policy TD)', '').replace(' (On-Policy TD)', '').replace(' (w/ Baseline)', ''),
    avgWaitTime: s.avgWaitTime,
    maxWaitTime: s.maxWaitTime,
    served: s.customersServed
  }));

  // Chart data for utilization across 10 lanes grouped by type
  const utilizationChartData = summary.map((s) => {
    const u = s.utilization || [];
    const avgUtil = u.length ? (u.reduce((a, b) => a + b, 0) / u.length) : 0;
    const expUtil = ((u[0] || 0) + (u[6] || 0)) / 2;
    const stdUtil = ((u[1] || 0) + (u[2] || 0) + (u[4] || 0) + (u[5] || 0)) / 4;
    const lrgUtil = ((u[3] || 0) + (u[7] || 0)) / 2;
    const flxUtil = ((u[8] || 0) + (u[9] || 0)) / 2;

    return {
      policy: s.label.replace(' (Off-Policy TD)', '').replace(' (On-Policy TD)', '').replace(' (w/ Baseline)', ''),
      'Overall Store Avg': Number(avgUtil.toFixed(1)),
      'Express Lanes (L0, L6)': Number(expUtil.toFixed(1)),
      'Standard Lanes (L1-2, L4-5)': Number(stdUtil.toFixed(1)),
      'Large Basket (L3, L7)': Number(lrgUtil.toFixed(1)),
      'Flexible Lanes (L8-9)': Number(flxUtil.toFixed(1))
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* SimPy Header & Controls */}
      <div className="glass-card" style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Clock size={22} color="#10b981" />
            <h2 style={{ fontSize: '1.25rem' }}>SimPy Multi-Policy Discrete-Event Simulation Arena</h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Simulates continuous stochastic customer arrivals (`simpy.Environment`) and cashier server resources across all 9 RL algorithms and baseline heuristics.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem' }}>
            <span>Arrival Rate (λ):</span>
            <input
              type="number"
              step="0.1"
              min="0.5"
              max="2.5"
              value={arrivalRate}
              onChange={(e) => setArrivalRate(e.target.value)}
              style={{
                width: '70px',
                background: 'var(--bg-tertiary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                padding: '6px 8px',
                fontWeight: 600
              }}
            />
            <span style={{ color: 'var(--text-muted)' }}>cust/sec</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem' }}>
            <span>Duration:</span>
            <select
              value={simTime}
              onChange={(e) => setSimTime(e.target.value)}
              style={{
                background: 'var(--bg-tertiary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                padding: '6px 10px',
                fontWeight: 600
              }}
            >
              <option value="60">60 Seconds</option>
              <option value="120">120 Seconds</option>
              <option value="300">300 Seconds</option>
            </select>
          </div>

          <button
            className="btn btn-primary"
            onClick={runSimPySimulation}
            disabled={isRunning}
          >
            {isRunning ? <RefreshCw size={16} className="spin-icon" /> : <Play size={16} />}
            {isRunning ? 'Running SimPy...' : 'Execute Arena Run'}
          </button>
        </div>
      </div>

      {/* Summary KPI Cards for SimPy */}
      <div className="kpi-grid">
        <div className="glass-card kpi-card" style={{ '--card-accent': '#10b981' }}>
          <div className="kpi-top">
            <span className="kpi-label">Top Performing Strategy</span>
            <div className="kpi-icon-box">
              <Trophy size={20} color="#f59e0b" />
            </div>
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value" style={{ fontSize: '1.25rem' }}>{bestPolicy}</span>
          </div>
          <div className="kpi-subtext">
            <span className="badge badge-success">Discrete-Event Winner</span>
            <span>Lowest Average Wait Time</span>
          </div>
        </div>

        {summary.slice(0, 3).map((pol, idx) => (
          <div
            key={pol.policy}
            className="glass-card kpi-card"
            style={{ '--card-accent': pol.color || '#3b82f6' }}
          >
            <div className="kpi-top">
              <span className="kpi-label">#{idx + 1}: {pol.label}</span>
              <div className="kpi-icon-box">
                <Users size={20} color={pol.color} />
              </div>
            </div>
            <div className="kpi-value-row">
              <span className="kpi-value">{pol.avgWaitTime}</span>
              <span className="kpi-unit">sec wait</span>
            </div>
            <div className="kpi-subtext">
              <span>Served: {pol.customersServed} | Max: {pol.maxWaitTime}s</span>
            </div>
          </div>
        ))}
      </div>

      {/* Visual Charts: Average Wait Times & Server Utilization */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '24px' }}>
        {/* Wait Time Bar Chart */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">Average & Max Customer Wait Time by Model</span>
            </div>
            <span className="badge badge-success">Seconds</span>
          </div>
          <div className="card-body" style={{ height: '340px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={waitTimeChartData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" opacity={0.6} />
                <XAxis dataKey="policy" stroke="var(--text-muted)" fontSize={11} interval={0} angle={-15} textAnchor="end" />
                <YAxis stroke="var(--text-muted)" fontSize={12} label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
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
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="avgWaitTime" name="Average Wait Time (s)" fill="#f05136" radius={[6, 6, 0, 0]} />
                <Bar dataKey="maxWaitTime" name="Maximum Peak Wait (s)" fill="#d97706" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cashier Utilization Bar Chart */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-title">Cashier Server Utilization (%) Across Policies</span>
            </div>
            <span className="badge badge-info">Busy Time / Total Time</span>
          </div>
          <div className="card-body" style={{ height: '340px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={utilizationChartData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" opacity={0.6} />
                <XAxis dataKey="policy" stroke="var(--text-muted)" fontSize={11} interval={0} angle={-15} textAnchor="end" />
                <YAxis domain={[0, 100]} stroke="var(--text-muted)" fontSize={12} label={{ value: '% Utilization', angle: -90, position: 'insideLeft' }} />
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
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="Overall Store Avg" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Express Lanes (L0, L6)" fill="#16a34a" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Standard Lanes (L1-2, L4-5)" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Large Basket (L3, L7)" fill="#d97706" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Flexible Lanes (L8-9)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* SimPy Comparison Table */}
      <div className="glass-card">
        <div className="card-header">
          <div className="card-title-group">
            <span className="card-title">SimPy Discrete-Event 10-Lane Arena Matrix</span>
          </div>
          <span className="badge badge-success">Duration: {simTime}s | Arrival Rate λ: {arrivalRate}</span>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          <div className="table-responsive">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Dispatch Strategy</th>
                  <th>Customers Served</th>
                  <th>Average Wait Time</th>
                  <th>Maximum Peak Wait</th>
                  <th>Avg Total Store Queue</th>
                  <th>Store Avg Util</th>
                  <th>Express Util (L0,6)</th>
                  <th>Standard Util</th>
                  <th>Large Basket Util</th>
                  <th>Flexible Util</th>
                </tr>
              </thead>
              <tbody>
                {summary.map((row) => {
                  const u = row.utilization || [];
                  const avgUtil = u.length ? (u.reduce((a, b) => a + b, 0) / u.length) : 0;
                  const expUtil = ((u[0] || 0) + (u[6] || 0)) / 2;
                  const stdUtil = ((u[1] || 0) + (u[2] || 0) + (u[4] || 0) + (u[5] || 0)) / 4;
                  const lrgUtil = ((u[3] || 0) + (u[7] || 0)) / 2;
                  const flxUtil = ((u[8] || 0) + (u[9] || 0)) / 2;

                  return (
                    <tr key={row.policy}>
                      <td style={{ fontWeight: 700, color: row.color || 'inherit' }}>{row.label}</td>
                      <td>{row.customersServed} customers</td>
                      <td style={{ fontWeight: 700, color: row.avgWaitTime < 2.5 ? '#10b981' : row.avgWaitTime > 4.0 ? '#ef4444' : 'inherit' }}>
                        {row.avgWaitTime}s
                      </td>
                      <td>{row.maxWaitTime}s</td>
                      <td>{row.avgTotalQueue}</td>
                      <td style={{ fontWeight: 600 }}>{avgUtil.toFixed(1)}%</td>
                      <td>{expUtil.toFixed(1)}%</td>
                      <td>{stdUtil.toFixed(1)}%</td>
                      <td>{lrgUtil.toFixed(1)}%</td>
                      <td>{flxUtil.toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

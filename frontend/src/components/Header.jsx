import React from 'react';
import { Play, RotateCcw, BarChart3, Menu, RefreshCw } from 'lucide-react';

export default function Header({
  activeTab,
  onRunBenchmark,
  onResetQueues,
  isBenchmarking,
  totalWaiting,
  onToggleMobileSidebar,
  activeModelName = 'Reinforcement Learning'
}) {
  const titles = {
    overview: {
      title: 'Retail Operations & Queue Intelligence',
      desc: 'Real-time telemetry and reinforcement learning queue dispatch analysis'
    },
    comparison: {
      title: 'Comparative Benchmark & Academic Paper Table IV',
      desc: 'Rigorous empirical evaluation across 9 RL algorithms and 2 heuristic baselines'
    },
    simulation: {
      title: 'Autonomous Checkout Lanes Simulator',
      desc: `Live interactive 3-lane dispatching with ${activeModelName} control`
    },
    policy: {
      title: 'RL Policy & Action-Value Studio',
      desc: 'Q-matrix heatmaps, convergence curves, and optimal action distributions'
    },
    simpy: {
      title: 'SimPy Discrete-Event Simulation Engine',
      desc: 'Continuous-time queue dynamics comparing Random, Heuristic, and RL'
    },
    benchmark: {
      title: '100-Timestep Benchmark & Transition Logs',
      desc: 'Comprehensive performance breakdown, store load curves, and event logs'
    },
    config: {
      title: 'Simulation Hyperparameters & Service Rates',
      desc: 'Customize cashier departure probabilities, learning rates, and retrain the agent'
    }
  };

  const current = titles[activeTab] || titles.overview;

  return (
    <header className="header">
      <div className="header-left">
        <button
          className="btn btn-outline btn-icon"
          style={{ display: 'none' }}
          id="mobile-menu-btn"
          onClick={onToggleMobileSidebar}
          aria-label="Toggle Navigation"
        >
          <Menu size={18} />
        </button>

        <div className="header-title-container">
          <h1 className="header-page-title">{current.title}</h1>
          <p className="header-page-desc">{current.desc}</p>
        </div>
      </div>

      <div className="header-right">
        <div className="badge badge-success" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
          Active Waiting: {totalWaiting} Customers
        </div>

        <button
          className="btn btn-secondary btn-sm"
          onClick={onResetQueues}
          title="Reset Interactive Queues to random state"
        >
          <RotateCcw size={14} />
          Reset
        </button>

        <button
          className="btn btn-primary btn-sm"
          onClick={onRunBenchmark}
          disabled={isBenchmarking}
          title="Run 100-timestep benchmark comparing Baseline vs Q-Learning"
        >
          {isBenchmarking ? <RefreshCw size={14} className="animate-spin" /> : <BarChart3 size={14} />}
          {isBenchmarking ? 'Running Benchmark...' : 'Run Benchmark'}
        </button>
      </div>
    </header>
  );
}

import React from 'react';
import {
  LayoutDashboard,
  ShoppingCart,
  BrainCircuit,
  Clock,
  FileSpreadsheet,
  Settings,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Scale
} from 'lucide-react';

export default function Sidebar({
  activeTab,
  setActiveTab,
  isCollapsed,
  setIsCollapsed,
  backendConnected,
  trainedEpisodes
}) {
  const navItems = [
    { id: 'overview', label: 'Store Overview', icon: LayoutDashboard },
    { id: 'comparison', label: 'Comparative Analysis', icon: Scale },
    { id: 'simulation', label: 'Live Checkout Sim', icon: ShoppingCart },
    { id: 'policy', label: 'RL Policy Studio', icon: BrainCircuit },
    { id: 'simpy', label: 'SimPy Event Engine', icon: Clock },
    { id: 'benchmark', label: 'Benchmark & Logs', icon: FileSpreadsheet },
    { id: 'config', label: 'Hyperparameters', icon: Settings }
  ];

  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <a href="#overview" className="brand-wrapper" onClick={(e) => { e.preventDefault(); setActiveTab('overview'); }}>
          <div className="brand-icon">
            <Sparkles size={22} />
          </div>
          {!isCollapsed && (
            <div className="brand-text-col">
              <span className="brand-name">OptiQueue AI</span>
              <span className="brand-tagline">Retail Queue Optimizer</span>
            </div>
          )}
        </a>
        <button
          className="btn btn-outline btn-sm btn-icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          aria-label="Toggle Sidebar"
        >
          {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="sidebar-nav">
        {!isCollapsed && <div className="nav-section-title">Analytics & Control</div>}
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
              title={isCollapsed ? item.label : ''}
            >
              <Icon size={20} className="nav-icon" />
              {!isCollapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        {!isCollapsed ? (
          <div className="status-pill">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div className={`status-dot ${backendConnected ? '' : 'offline'}`} />
              <span style={{ fontWeight: 600 }}>
                {backendConnected ? 'Python API Online' : 'Connecting API...'}
              </span>
            </div>
            <span style={{ color: 'var(--text-muted)' }}>
              {trainedEpisodes} Ep
            </span>
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center' }} title={backendConnected ? 'API Online' : 'API Offline'}>
            <div className={`status-dot ${backendConnected ? '' : 'offline'}`} />
          </div>
        )}
      </div>
    </aside>
  );
}

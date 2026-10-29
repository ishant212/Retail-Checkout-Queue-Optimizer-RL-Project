import React from 'react';
import {
  ShoppingCart,
  Play,
  Pause,
  Clock,
  BarChart2,
  ChevronDown,
  Wifi,
  WifiOff
} from 'lucide-react';

/**
 * SimulationHeader - Compact top navigation bar containing:
 * 1. OptiQueue AI branding
 * 2. View switcher pills (Simulation, Analytics, RL Training, Settings)
 * 3. Play/Pause toggle button
 * 4. Speed selector dropdown (0.5x, 1x, 2x, 5x)
 * 5. Simulation time clock
 * 6. Episode and active RL algorithm indicator
 * 7. Live / Demo mode indicator
 */
export default function SimulationHeader({
  activeTab = 'simulation',
  onTabChange,
  scenario = 'peak',
  onScenarioChange,
  isPlaying = false,
  onTogglePlay,
  speed = 1,
  onSpeedChange,
  simTime = '00:12:34',
  episode = 2450,
  activeAlgorithm = 'Double DQN',
  isBackendConnected = true,
  onStep
}) {
  const tabs = [
    { id: 'simulation', label: 'Simulation' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'training', label: 'RL Training' },
    { id: 'settings', label: 'Settings' }
  ];

  return (
    <header className="w-full flex items-center justify-between px-4 py-2.5 bg-[#0a0e17] border-b border-slate-800/90 select-none z-30">
      {/* 1. Left Branding */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-cyan-700 flex items-center justify-center text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.4)]">
          <ShoppingCart size={18} strokeWidth={2.5} />
        </div>
        <div className="flex flex-col leading-tight">
          <div className="flex items-center gap-2">
            <span className="text-base font-extrabold tracking-tight text-white">
              OptiQueue AI
            </span>
            {/* Backend connectivity indicator */}
            <span
              className={`text-[9px] px-1.5 py-0.5 rounded-full font-mono flex items-center gap-1 border ${
                isBackendConnected
                  ? 'bg-emerald-950/60 border-emerald-600/40 text-emerald-400'
                  : 'bg-amber-950/60 border-amber-600/40 text-amber-400'
              }`}
              title={isBackendConnected ? 'Connected to Flask simulation backend' : 'Running in deterministic Demo Mode'}
            >
              {isBackendConnected ? <Wifi size={9} /> : <WifiOff size={9} />}
              <span>{isBackendConnected ? 'LIVE' : 'DEMO'}</span>
            </span>
          </div>
          <span className="text-[11px] font-medium text-slate-400">
            Retail Checkout Simulation
          </span>
        </div>
      </div>

      {/* 2. Center View Switcher Navigation & Demand Scenario Selector */}
      <div className="flex items-center gap-3">
        <nav className="flex items-center p-1 rounded-xl bg-[#0e1627] border border-slate-800/80 shadow-inner">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange && onTabChange(tab.id)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-cyan-500/10 border border-cyan-500/60 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.25)]'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Demand Scenario Pills */}
        <div className="flex items-center p-1 rounded-xl bg-[#0e1627] border border-slate-800/80 shadow-inner">
          {[
            { id: 'off_peak', label: 'Off-Peak', icon: '🌙' },
            { id: 'normal', label: 'Normal Traffic', icon: '🏬' },
            { id: 'peak', label: 'Peak Hours', icon: '⚡' }
          ].map((scen) => {
            const isSelected = scenario === scen.id;
            return (
              <button
                key={scen.id}
                onClick={() => onScenarioChange && onScenarioChange(scen.id)}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all duration-150 cursor-pointer flex items-center gap-1.5 ${
                  isSelected
                    ? scen.id === 'peak'
                      ? 'bg-rose-500/20 border border-rose-500/60 text-rose-300 shadow-[0_0_10px_rgba(244,63,94,0.3)]'
                      : scen.id === 'normal'
                      ? 'bg-cyan-500/20 border border-cyan-500/60 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                      : 'bg-emerald-500/20 border border-emerald-500/60 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
                title={`Switch store demand profile to ${scen.label}`}
              >
                <span className="text-xs">{scen.icon}</span>
                <span>{scen.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 3. Right Simulation Controls & Indicators */}
      <div className="flex items-center gap-4">
        {/* Play / Pause Toggle Button */}
        <button
          onClick={onTogglePlay}
          className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-200 cursor-pointer shadow-md ${
            isPlaying
              ? 'bg-cyan-500 text-slate-950 shadow-[0_0_12px_#22d3ee] hover:bg-cyan-400'
              : 'bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:border-cyan-500'
          }`}
          title={isPlaying ? 'Pause Simulation' : 'Start / Resume Simulation'}
        >
          {isPlaying ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" className="ml-0.5" />}
        </button>

        {/* Speed Selector Dropdown */}
        <div className="relative">
          <select
            value={speed}
            onChange={(e) => onSpeedChange && onSpeedChange(Number(e.target.value))}
            className="appearance-none bg-[#0e1627] border border-slate-800 rounded-lg px-2.5 py-1 pr-6 text-xs font-mono font-semibold text-slate-200 cursor-pointer hover:border-slate-700 focus:outline-none focus:border-cyan-500"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={5}>5x</option>
          </select>
          <ChevronDown
            size={12}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
          />
        </div>

        {/* Elapsed Simulation Time */}
        <div className="flex items-center gap-2 border-l border-slate-800/80 pl-3">
          <div className="flex flex-col text-right leading-none">
            <span className="font-mono text-xs font-bold text-slate-200">
              {simTime}
            </span>
            <span className="text-[9px] text-slate-400 mt-0.5">
              Simulation Time
            </span>
          </div>
        </div>

        {/* Episode and Active Algorithm Badge */}
        <div className="flex items-center gap-2 border-l border-slate-800/80 pl-3">
          <div className="w-7 h-7 rounded-lg bg-slate-800/70 border border-slate-700/60 flex items-center justify-center text-cyan-400">
            <BarChart2 size={13} />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-[11px] font-mono font-bold text-slate-200">
              Episode: {episode}
            </span>
            <span className="text-[9px] text-slate-400 mt-0.5">
              Algorithm: {activeAlgorithm}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

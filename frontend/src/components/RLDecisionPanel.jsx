import React, { useState } from 'react';
import { Cpu, ChevronDown, ChevronUp } from 'lucide-react';
import { INITIAL_Q_VALUES } from '../simulationTypes';

/**
 * RLDecisionPanel - Right side bottom card showing:
 * 1. Selected Action pill
 * 2. Active Policy (e.g. ε-greedy)
 * 3. Top 5 Action Q-Values with horizontal cyan bars
 * 4. Collapsible Advanced Diagnostics (TD Error, Target, Discount)
 */
export default function RLDecisionPanel({
  selectedAction = 2,
  actionSemantic = 'Route to Lane #2',
  policyName = 'ε-greedy (ε = 0.1)',
  qValues = INITIAL_Q_VALUES,
  policyUpdate = null
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Normalize top Q-Values for horizontal bar percentage
  const maxQ = Math.max(...qValues.map((v) => v.q), 1.0);

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-3 shadow-lg select-none">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-800/80">
        <div className="w-5 h-5 rounded-full bg-cyan-950 flex items-center justify-center text-cyan-400">
          <Cpu size={12} />
        </div>
        <span className="text-xs font-semibold text-slate-200">
          RL Action Selection
        </span>
      </div>

      {/* Selected Action Pill & Policy */}
      <div className="space-y-2 mb-3 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-slate-400">Selected Action</span>
          <span className="px-2.5 py-1 rounded-md bg-cyan-950 border border-cyan-500/80 text-cyan-300 font-semibold font-mono text-[11px] shadow-[0_0_8px_rgba(6,182,212,0.3)]">
            {typeof actionSemantic === 'object' ? actionSemantic.label : actionSemantic || `Route to Lane #${selectedAction}`}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-400">Policy</span>
          <span className="font-mono text-slate-300 text-[11px]">{policyName}</span>
        </div>
      </div>

      {/* Top 5 Q-Values with horizontal bars */}
      <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
        <span className="text-[10px] text-slate-400 font-medium block mb-1">
          Q-Value (top 5)
        </span>

        {qValues.slice(0, 5).map((item, idx) => {
          const barWidth = Math.min(100, Math.max(10, (item.q / maxQ) * 100));
          return (
            <div key={idx} className="flex items-center gap-2 text-xs">
              <span className="w-14 text-slate-300 text-[11px] font-mono shrink-0">
                {item.label}
              </span>
              <span className="w-12 text-slate-400 text-[10px] font-mono shrink-0 text-right">
                {typeof item.q === 'number' ? item.q.toFixed(3) : item.q}
              </span>
              <div className="flex-1 h-2 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full shadow-[0_0_6px_#22d3ee]"
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Collapsible Advanced Technical Diagnostics */}
      <div className="mt-3 pt-2 border-t border-slate-800/80">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between text-[10px] text-slate-400 hover:text-cyan-400 transition-colors py-1 cursor-pointer"
        >
          <span>Advanced Diagnostics</span>
          {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>

        {showAdvanced && (
          <div className="mt-2 space-y-1 bg-slate-950/60 p-2 rounded text-[10px] font-mono text-slate-400">
            <div className="flex justify-between">
              <span>TD Target:</span>
              <span className="text-slate-200">{policyUpdate?.tdTarget ?? '2.840'}</span>
            </div>
            <div className="flex justify-between">
              <span>TD Error (δ):</span>
              <span className="text-slate-200">{policyUpdate?.tdError ?? '0.124'}</span>
            </div>
            <div className="flex justify-between">
              <span>Gamma (γ):</span>
              <span className="text-slate-200">{policyUpdate?.gamma ?? '0.950'}</span>
            </div>
            <div className="flex justify-between">
              <span>Alpha (α):</span>
              <span className="text-slate-200">{policyUpdate?.alpha ?? '0.100'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

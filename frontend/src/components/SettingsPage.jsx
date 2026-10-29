import React, { useState } from 'react';
import {
  Settings,
  Sliders,
  DollarSign,
  Layers,
  Save,
  CheckCircle,
  RotateCcw
} from 'lucide-react';
import { LANE_CONFIG } from '../simulationTypes';
import { api } from '../api';

export default function SettingsPage({
  openStatus = [],
  capacities = [],
  serviceRates = [],
  laneTypes = [],
  onUpdateLaneConfig
}) {
  const [arrivalRate, setArrivalRate] = useState(1.0);
  const [expressLimit, setExpressLimit] = useState(5);
  const [standardLimit, setStandardLimit] = useState(15);

  // Multi-term Reward Configuration
  const [rewardWeights, setRewardWeights] = useState({
    alpha: 3.0, // Departure reward
    beta: 0.15, // Wait penalty
    gamma: 0.25, // Queue penalty
    delta: 0.8, // Congestion penalty
    rho: 1.0, // Congestion duration
    zeta: 0.3, // Imbalance
    kappa: 0.25, // Operating cost
    xi: 5.0 // Invalid action
  });

  const [savedMessage, setSavedMessage] = useState(false);

  const handleSave = () => {
    setSavedMessage(true);
    setTimeout(() => setSavedMessage(false), 2500);
  };

  const toggleLane = async (laneId) => {
    const current = openStatus[laneId] !== undefined ? openStatus[laneId] : true;
    try {
      await api.updateLane({ laneId, isOpen: !current });
      if (onUpdateLaneConfig) {
        onUpdateLaneConfig(laneId, { isOpen: !current });
      }
    } catch (err) {
      console.warn('Update lane failed:', err);
      if (onUpdateLaneConfig) {
        onUpdateLaneConfig(laneId, { isOpen: !current });
      }
    }
  };

  return (
    <div className="w-full flex-1 flex flex-col gap-4 p-4 overflow-y-auto select-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white tracking-tight">
            Environment & Simulation Parameters
          </h2>
          <p className="text-xs text-slate-400">
            Configure arrival processes, multi-term reward formulation weights, and checkout lane constraints
          </p>
        </div>

        <div className="flex items-center gap-2">
          {savedMessage && (
            <span className="flex items-center gap-1 text-xs text-emerald-400 font-medium">
              <CheckCircle size={14} /> Saved Successfully
            </span>
          )}
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs shadow-[0_0_12px_rgba(6,182,212,0.4)] cursor-pointer"
          >
            <Save size={13} />
            <span>Apply Settings</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Left Column: Arrival & Reward formulation */}
        <div className="col-span-12 lg:col-span-6 flex flex-col gap-4">
          {/* Arrival & Basket Limits */}
          <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-4 shadow-lg space-y-3">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
              <Sliders size={14} className="text-cyan-400" />
              <span className="text-xs font-semibold text-slate-200">
                Customer Arrival & Basket Limits
              </span>
            </div>

            <div className="grid grid-cols-3 gap-3 text-xs">
              <div>
                <label className="text-[10px] text-slate-400 block mb-1">Arrival Rate (λ)</label>
                <input
                  type="number"
                  step="0.1"
                  value={arrivalRate}
                  onChange={(e) => setArrivalRate(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono text-xs focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-1">Express Max Items</label>
                <input
                  type="number"
                  value={expressLimit}
                  onChange={(e) => setExpressLimit(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono text-xs focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-1">Standard Max Items</label>
                <input
                  type="number"
                  value={standardLimit}
                  onChange={(e) => setStandardLimit(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono text-xs focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Reward Formulation Weights */}
          <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-4 shadow-lg space-y-3">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
              <DollarSign size={14} className="text-cyan-400" />
              <span className="text-xs font-semibold text-slate-200">
                Multi-Term Reward Formulation Weights (R_t)
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Departure Reward (α)</label>
                <input
                  type="number"
                  step="0.1"
                  value={rewardWeights.alpha}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, alpha: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Wait Penalty (β)</label>
                <input
                  type="number"
                  step="0.05"
                  value={rewardWeights.beta}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, beta: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Queue Penalty (γ)</label>
                <input
                  type="number"
                  step="0.05"
                  value={rewardWeights.gamma}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, gamma: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Congestion Severity (δ)</label>
                <input
                  type="number"
                  step="0.1"
                  value={rewardWeights.delta}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, delta: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Congestion Duration (ρ)</label>
                <input
                  type="number"
                  step="0.1"
                  value={rewardWeights.rho}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, rho: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Imbalance Penalty (ζ)</label>
                <input
                  type="number"
                  step="0.05"
                  value={rewardWeights.zeta}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, zeta: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Operating Cost (κ)</label>
                <input
                  type="number"
                  step="0.05"
                  value={rewardWeights.kappa}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, kappa: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[10px] text-slate-400 block mb-0.5">Invalid Action Penalty (ξ)</label>
                <input
                  type="number"
                  step="0.5"
                  value={rewardWeights.xi}
                  onChange={(e) => setRewardWeights({ ...rewardWeights, xi: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Checkout Lane Status Controls (6 columns) */}
        <div className="col-span-12 lg:col-span-6 flex flex-col gap-4">
          <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-4 shadow-lg space-y-3">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
              <Layers size={14} className="text-cyan-400" />
              <span className="text-xs font-semibold text-slate-200">
                Checkout Lanes Configuration & Operational Status
              </span>
            </div>

            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
              {LANE_CONFIG.map((lane) => {
                const isOpen = openStatus[lane.id] !== undefined ? openStatus[lane.id] : true;
                const cap = capacities[lane.id] || lane.capacity;
                const type = laneTypes[lane.id] || lane.type;

                return (
                  <div
                    key={lane.id}
                    className="p-2.5 rounded-lg border border-slate-800/80 bg-slate-900/50 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center font-mono font-bold text-slate-200 text-xs">
                        {lane.label}
                      </div>
                      <div>
                        <div className="font-semibold text-slate-200">
                          {lane.name} Lane
                        </div>
                        <div className="text-[10px] text-slate-400">
                          Capacity: {cap} • Basket: {lane.limit}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => toggleLane(lane.id)}
                        className={`px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                          isOpen
                            ? 'bg-emerald-950/80 border border-emerald-500/80 text-emerald-300'
                            : 'bg-rose-950/80 border border-rose-500/80 text-rose-300'
                        }`}
                      >
                        {isOpen ? 'Open' : 'Closed'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React from 'react';
import { User } from 'lucide-react';
import CustomerBadge from './CustomerBadge';
import { LANE_CONFIG } from '../simulationTypes';

/**
 * LaneInspector - Right side top card showing detailed inspector for selected lane:
 * Status, Queue Length, Service Rate, Lane Type, and visual mini queue representation.
 */
export default function LaneInspector({
  laneId = 2,
  openStatus = [],
  currentQueues = [],
  capacities = [],
  serviceRates = [],
  laneTypes = [],
  laneCustomers = []
}) {
  const cfg = LANE_CONFIG[laneId] || LANE_CONFIG[2];
  const isOpen = openStatus[laneId] !== undefined ? openStatus[laneId] : true;
  const queueLen = currentQueues[laneId] !== undefined ? currentQueues[laneId] : 4;
  const maxCap = capacities[laneId] || cfg.capacity;
  const sRate = serviceRates[laneId] !== undefined ? serviceRates[laneId] : 0.5;
  const type = laneTypes[laneId] || cfg.type;

  const laneData = laneCustomers[laneId] || {};
  const queueList = laneData.queue || [];
  const servingCustomer = laneData.serving || null;

  let typeDisplay = 'Standard (≤ 15)';
  if (type === 'express') typeDisplay = 'Express (≤ 5)';
  else if (type === 'large_basket') typeDisplay = 'Bulk Cart (Any)';
  else if (type === 'flexible') typeDisplay = 'Self-Serve (≤ 10)';

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-3 shadow-lg select-none">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-cyan-950 flex items-center justify-center text-cyan-400">
            <User size={12} />
          </div>
          <span className="text-xs font-semibold text-slate-200">
            Lane Information
          </span>
        </div>
        <span className="text-[11px] font-mono text-cyan-400">
          #{laneId} - {cfg.name} Lane
        </span>
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        {/* Left column: Key Stats */}
        <div className="space-y-2.5">
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400">Status</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span
                className={`w-2 h-2 rounded-full ${
                  isOpen ? 'bg-emerald-400 shadow-[0_0_6px_#34d399]' : 'bg-rose-500'
                }`}
              />
              <span className="font-medium text-slate-200">
                {isOpen ? 'Open' : 'Closed'}
              </span>
            </div>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400">Queue Length</span>
            <span className="font-mono font-semibold text-slate-200 mt-0.5">
              {queueLen} / {maxCap}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400">Service Rate (μ)</span>
            <span className="font-mono font-medium text-slate-200 mt-0.5">
              {(sRate * 0.24).toFixed(2)} items/s
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400">Type</span>
            <span className="font-medium text-slate-200 mt-0.5">{typeDisplay}</span>
          </div>
        </div>

        {/* Right column: Vertical mini queue visualization */}
        <div className="flex flex-col items-center justify-between bg-slate-950/60 p-2 rounded-lg border border-slate-800/80 min-h-[160px]">
          {/* Queue elements with labels */}
          <div className="w-full flex flex-col items-center gap-1.5 flex-1 justify-end">
            {/* Show empty slot */}
            {maxCap > queueLen && (
              <div className="flex items-center gap-2 w-full justify-start text-[10px] text-slate-500">
                <CustomerBadge isEmpty size="sm" />
                <span>Empty slot</span>
              </div>
            )}

            {/* Customers waiting */}
            {queueList.length > 0
              ? queueList.map((cust, idx) => (
                  <div
                    key={cust.id || idx}
                    className="flex items-center gap-2 w-full justify-start text-[10px] text-slate-300"
                  >
                    <CustomerBadge basket={cust.basket} size="sm" />
                    <span>{idx === 0 ? 'Next to serve' : 'Waiting'}</span>
                  </div>
                ))
              : Array.from({ length: Math.min(4, queueLen) }).map((_, idx) => {
                  const b = [12, 8, 6, 4][idx] || 5;
                  return (
                    <div
                      key={idx}
                      className="flex items-center gap-2 w-full justify-start text-[10px] text-slate-300"
                    >
                      <CustomerBadge basket={b} size="sm" />
                      <span>{idx === 0 ? 'Next to serve' : 'Waiting'}</span>
                    </div>
                  );
                })}
          </div>

          {/* Cashier silhouette seated at bottom */}
          <div className="mt-2 pt-1 border-t border-slate-800/80 w-full flex items-center justify-center gap-1.5 text-slate-400">
            <div className="w-4 h-4 rounded-full bg-slate-700" />
            <span className="text-[10px] font-medium">Cashier Active</span>
          </div>
        </div>
      </div>
    </div>
  );
}

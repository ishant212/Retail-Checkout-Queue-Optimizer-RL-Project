import React from 'react';
import {
  User,
  Users,
  CheckCircle,
  Clock,
  BarChart2,
  Coins,
  ArrowUp,
  ArrowDown
} from 'lucide-react';

/**
 * MetricsBar - 8 KPI metric cards across the top of the dashboard
 * directly matching the reference design.
 */
export default function MetricsBar({
  totalCustomers = 1054,
  currentlyInStore = 42,
  servedCustomers = 1012,
  avgWaitTime = 2.8,
  throughput = 43.2,
  activeCashiers = 8,
  totalCashiers = 10,
  totalLaborCost = 148.0,
  avgQueueLength = 4.2
}) {
  const cashierPercent = Math.min(100, (activeCashiers / totalCashiers) * 100);

  return (
    <div className="w-full grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-2.5 select-none">
      {/* 1. Total customers */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <User size={13} className="text-slate-400" />
          <span className="text-[11px] font-medium truncate">Total customers</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {totalCustomers.toLocaleString()}
          </div>
          <div className="flex items-center gap-0.5 text-[10px] text-emerald-400 font-medium">
            <ArrowUp size={10} />
            <span>+12%</span>
          </div>
        </div>
      </div>

      {/* 2. Currently in store */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Users size={13} className="text-cyan-400" />
          <span className="text-[11px] font-medium truncate">Currently in store</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {currentlyInStore}
          </div>
          <div className="text-[9px] text-slate-400 truncate">
            (in queues + service)
          </div>
        </div>
      </div>

      {/* 3. Served customers */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <CheckCircle size={13} className="text-emerald-400" />
          <span className="text-[11px] font-medium truncate">Served customers</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {servedCustomers.toLocaleString()}
          </div>
          <div className="flex items-center gap-0.5 text-[10px] text-emerald-400 font-medium">
            <ArrowUp size={10} />
            <span>+15%</span>
          </div>
        </div>
      </div>

      {/* 4. Avg. wait time */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Clock size={13} className="text-purple-400" />
          <span className="text-[11px] font-medium truncate">Avg. wait time</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {typeof avgWaitTime === 'number' ? avgWaitTime.toFixed(1) : avgWaitTime} min
          </div>
          <div className="flex items-center gap-0.5 text-[10px] text-emerald-400 font-medium">
            <ArrowDown size={10} />
            <span>-18%</span>
          </div>
        </div>
      </div>

      {/* 5. Throughput */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <BarChart2 size={13} className="text-emerald-400" />
          <span className="text-[11px] font-medium truncate">Throughput</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {typeof throughput === 'number' ? throughput.toFixed(1) : throughput} <span className="text-[10px] font-normal text-slate-400">cust/hr</span>
          </div>
          <div className="flex items-center gap-0.5 text-[10px] text-emerald-400 font-medium">
            <ArrowUp size={10} />
            <span>+20%</span>
          </div>
        </div>
      </div>

      {/* 6. Active cashiers */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <User size={13} className="text-cyan-400" />
          <span className="text-[11px] font-medium truncate">Active cashiers</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {activeCashiers} / {totalCashiers}
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1">
            <div
              className="h-full bg-cyan-400 rounded-full shadow-[0_0_6px_#22d3ee]"
              style={{ width: `${cashierPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* 7. Total labor cost */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Coins size={13} className="text-amber-400" />
          <span className="text-[11px] font-medium truncate">Total labor cost</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            ${typeof totalLaborCost === 'number' ? totalLaborCost.toFixed(1) : totalLaborCost}
          </div>
          <div className="text-[9px] text-slate-400">
            / hour
          </div>
        </div>
      </div>

      {/* 8. Avg. queue length */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Users size={13} className="text-purple-400" />
          <span className="text-[11px] font-medium truncate">Avg. queue length</span>
        </div>
        <div className="mt-1">
          <div className="text-base font-bold text-slate-100 font-mono">
            {typeof avgQueueLength === 'number' ? avgQueueLength.toFixed(1) : avgQueueLength}
          </div>
          <div className="flex items-center gap-0.5 text-[10px] text-emerald-400 font-medium">
            <ArrowDown size={10} />
            <span>-25%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

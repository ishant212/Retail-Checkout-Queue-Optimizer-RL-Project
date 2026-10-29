import React from 'react';
import { User } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';

/**
 * IncomingCustomerPanel - Left side panel showing:
 * 1. Current Incoming Customer details (ID, Basket Size, Express Eligibility, Arrival Time)
 * 2. Next Arrival countdown & progress bar
 * 3. Basket Size Distribution histogram
 */
export default function IncomingCustomerPanel({
  incomingCustomer = {
    id: 'C#1055',
    basket: 7,
    isExpress: false,
    arrivalTime: '00:12:34'
  },
  nextArrivalSeconds = 12,
  nextArrivalMax = 15,
  basketDistribution = [
    { items: 1, count: 5 },
    { items: 2, count: 12 },
    { items: 3, count: 18 },
    { items: 4, count: 24 },
    { items: 5, count: 15 },
    { items: 6, count: 10 },
    { items: 7, count: 8 },
    { items: 8, count: 10 },
    { items: 9, count: 7 },
    { items: 10, count: 5 },
    { items: 12, count: 3 },
    { items: 15, count: 2 },
    { items: 18, count: 1 },
    { items: 20, count: 1 }
  ],
  avgBasketSize = 8.7
}) {
  const arrivalPercent = Math.min(100, Math.max(0, (nextArrivalSeconds / nextArrivalMax) * 100));

  return (
    <div className="w-full flex flex-col gap-3 select-none">
      {/* 1. Incoming Customer Card */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-3 shadow-lg">
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-800/80">
          <div className="w-5 h-5 rounded-full bg-cyan-950 flex items-center justify-center text-cyan-400">
            <User size={12} />
          </div>
          <span className="text-xs font-semibold text-slate-200">Incoming Customer</span>
        </div>

        <div className="space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">ID:</span>
            <span className="font-mono font-medium text-slate-200">{incomingCustomer.id}</span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Basket Size:</span>
            <span className="font-semibold text-slate-100">{incomingCustomer.basket} items</span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Express Eligible:</span>
            <span
              className={`font-semibold ${
                incomingCustomer.isExpress ? 'text-emerald-400' : 'text-slate-400'
              }`}
            >
              {incomingCustomer.isExpress ? 'Yes' : 'No'}
            </span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Arrival Time:</span>
            <span className="font-mono text-slate-300">{incomingCustomer.arrivalTime}</span>
          </div>
        </div>
      </div>

      {/* 2. Next Arrival Countdown Card */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-3 shadow-lg">
        <span className="text-[11px] font-medium text-slate-400 block mb-1">Next Arrival</span>
        <div className="text-xl font-bold font-mono text-slate-100 mb-2">{nextArrivalSeconds}s</div>
        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-cyan-400 rounded-full transition-all duration-300 shadow-[0_0_8px_#22d3ee]"
            style={{ width: `${arrivalPercent}%` }}
          />
        </div>
      </div>

      {/* 3. Basket Size Distribution Histogram */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-3 shadow-lg flex flex-col justify-between h-[180px]">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] font-medium text-slate-300">Basket Size Distribution</span>
        </div>

        <div className="w-full flex-1 min-h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={basketDistribution} margin={{ top: 5, right: 0, left: -25, bottom: 0 }}>
              <XAxis
                dataKey="items"
                stroke="#64748b"
                tick={{ fontSize: 9 }}
                tickLine={false}
                interval={3}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 8 }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '6px',
                  fontSize: '10px'
                }}
              />
              <Bar dataKey="count" fill="#38bdf8" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/80 flex justify-between items-center">
          <span>Avg: {avgBasketSize} items</span>
          <span className="text-[9px] text-slate-500">Normal Dist</span>
        </div>
      </div>
    </div>
  );
}

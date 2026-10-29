import React from 'react';
import {
  Clock,
  TrendingUp,
  BarChart2,
  Activity
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from 'recharts';

/**
 * AnalyticsRow - 4 compact bottom panels exactly matching the reference design:
 * 1. Queue Lengths (Real-time) - Bar chart across 10 lanes
 * 2. Average Wait Time - Cyan area chart with 2.8 min callout badge
 * 3. Throughput - Yellow area chart with 43.2 /hr callout badge
 * 4. Reward & Training - Multi-line chart for R_t and Avg Reward (100 ep)
 */
export default function AnalyticsRow({
  currentQueues = [3, 2, 4, 3, 4, 3, 3, 4, 4, 2],
  currentWaitTime = 2.8,
  currentThroughput = 43.2,
  waitTimeHistory = [],
  throughputHistory = [],
  rewardHistory = []
}) {
  // Format bar data for 10 lanes
  const queueBarData = currentQueues.map((q, idx) => ({
    lane: `#${idx}`,
    queue: q
  }));

  // Bar colors matching reference: cyan for standard, red/coral for high queue, amber for moderate
  const getBarColor = (laneIdx, queue) => {
    if (queue >= 4) return '#f87171'; // red
    if (queue === 3) return '#fbbf24'; // amber
    if (laneIdx === 2) return '#22d3ee'; // cyan
    return '#38bdf8'; // sky
  };

  // Generate realistic smooth default curves if history is short
  const defaultWaitHistory =
    waitTimeHistory.length > 5
      ? waitTimeHistory
      : [
          { time: '0', wait: 2.1 },
          { time: '5', wait: 4.2 },
          { time: '10', wait: 4.8 },
          { time: '15', wait: 3.5 },
          { time: '20', wait: 4.3 },
          { time: '25', wait: 2.7 },
          { time: '30', wait: 2.8 }
        ];

  const defaultThroughputHistory =
    throughputHistory.length > 5
      ? throughputHistory
      : [
          { time: '0', tp: 22 },
          { time: '5', tp: 34 },
          { time: '10', tp: 41 },
          { time: '15', tp: 48 },
          { time: '20', tp: 62 },
          { time: '25', tp: 51 },
          { time: '30', tp: 43.2 }
        ];

  const defaultRewardHistory =
    rewardHistory.length > 10
      ? rewardHistory
      : Array.from({ length: 40 }).map((_, i) => {
          const ep = i * 65;
          const noise = (Math.sin(i * 0.8) * 20) + (Math.random() * 15 - 7);
          const base = -30 + Math.min(85, i * 2.8);
          return {
            ep,
            reward: Math.round(base + noise),
            avgReward: Math.round(base)
          };
        });

  return (
    <div className="w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2.5 select-none">
      {/* 1. Queue Lengths (Real-time) */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between h-[150px]">
        <div className="flex items-center gap-1.5 mb-1">
          <Activity size={12} className="text-cyan-400" />
          <span className="text-[11px] font-semibold text-slate-200">
            Queue Lengths <span className="text-slate-400 font-normal">(Real-time)</span>
          </span>
        </div>

        <div className="w-full flex-1 min-h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={queueBarData} margin={{ top: 5, right: 0, left: -28, bottom: -5 }}>
              <XAxis
                dataKey="lane"
                stroke="#64748b"
                tick={{ fontSize: 9 }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 8 }}
                tickLine={false}
                axisLine={false}
                domain={[0, 8]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '6px',
                  fontSize: '10px'
                }}
              />
              <Bar dataKey="queue" radius={[2, 2, 0, 0]}>
                {queueBarData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={getBarColor(index, entry.queue)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. Average Wait Time */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between h-[150px] relative">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="text-cyan-400" />
            <span className="text-[11px] font-semibold text-slate-200">Average Wait Time</span>
          </div>
          <span className="px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/50 text-cyan-300 font-mono text-[10px] font-bold">
            {typeof currentWaitTime === 'number' ? currentWaitTime.toFixed(1) : currentWaitTime} min
          </span>
        </div>

        <div className="w-full flex-1 min-h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={defaultWaitHistory} margin={{ top: 5, right: 5, left: -28, bottom: -5 }}>
              <defs>
                <linearGradient id="waitGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="time"
                stroke="#64748b"
                tick={{ fontSize: 9 }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 8 }}
                tickLine={false}
                axisLine={false}
                domain={[0, 8]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '6px',
                  fontSize: '10px'
                }}
              />
              <Area
                type="monotone"
                dataKey="wait"
                stroke="#22d3ee"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#waitGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Throughput */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between h-[150px] relative">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5">
            <BarChart2 size={12} className="text-amber-400" />
            <span className="text-[11px] font-semibold text-slate-200">Throughput</span>
          </div>
          <span className="px-2 py-0.5 rounded bg-amber-950/80 border border-amber-500/50 text-amber-300 font-mono text-[10px] font-bold">
            {typeof currentThroughput === 'number' ? currentThroughput.toFixed(1) : currentThroughput} /hr
          </span>
        </div>

        <div className="w-full flex-1 min-h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={defaultThroughputHistory} margin={{ top: 5, right: 5, left: -28, bottom: -5 }}>
              <defs>
                <linearGradient id="tpGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="time"
                stroke="#64748b"
                tick={{ fontSize: 9 }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 8 }}
                tickLine={false}
                axisLine={false}
                domain={[0, 80]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '6px',
                  fontSize: '10px'
                }}
              />
              <Area
                type="monotone"
                dataKey="tp"
                stroke="#f59e0b"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#tpGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 4. Reward & Training */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1626] p-2.5 shadow-md flex flex-col justify-between h-[150px]">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5">
            <TrendingUp size={12} className="text-cyan-400" />
            <span className="text-[11px] font-semibold text-slate-200">Reward & Training</span>
          </div>
          <div className="flex items-center gap-2 text-[9px]">
            <span className="flex items-center gap-1 text-cyan-400">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              Reward (R_t)
            </span>
            <span className="flex items-center gap-1 text-purple-400">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
              Avg. (100 ep)
            </span>
          </div>
        </div>

        <div className="w-full flex-1 min-h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={defaultRewardHistory} margin={{ top: 5, right: 5, left: -28, bottom: -5 }}>
              <XAxis
                dataKey="ep"
                stroke="#64748b"
                tick={{ fontSize: 9 }}
                tickLine={false}
                interval={10}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 8 }}
                tickLine={false}
                axisLine={false}
                domain={[-50, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '6px',
                  fontSize: '10px'
                }}
              />
              <Line
                type="monotone"
                dataKey="reward"
                stroke="#22d3ee"
                strokeWidth={1}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="avgReward"
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

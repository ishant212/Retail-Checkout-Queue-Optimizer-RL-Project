import React from 'react';

/**
 * CashierStation - Modern 2.5D supermarket checkout counter with conveyor belt,
 * POS touch register, minimal cashier silhouette, and lane dividers.
 */
export default function CashierStation({
  isOpen = true,
  isServing = false,
  servingCustomer = null,
  isTarget = false,
  laneType = 'standard'
}) {
  return (
    <div className="relative w-full flex flex-col items-center">
      {/* Chrome Divider Stanchions (left and right guide rails) */}
      <div className="absolute -left-1 top-2 bottom-1 w-0.5 bg-gradient-to-b from-slate-400 via-slate-600 to-slate-800 rounded-full opacity-60 z-10" />
      <div className="absolute -right-1 top-2 bottom-1 w-0.5 bg-gradient-to-b from-slate-400 via-slate-600 to-slate-800 rounded-full opacity-60 z-10" />

      {/* Main Checkout Station Assembly */}
      <div className="relative w-full max-w-[84px] h-[78px] flex flex-col items-center justify-end">
        {/* Cashier Silhouette seated behind register */}
        <div className="absolute top-1 z-0 flex flex-col items-center">
          {/* Head */}
          <div
            className={`w-4 h-4 rounded-full transition-colors duration-300 ${
              isOpen ? 'bg-slate-700' : 'bg-slate-800'
            }`}
          />
          {/* Shoulders / Torso */}
          <div
            className={`w-7 h-5 rounded-t-lg transition-colors duration-300 -mt-0.5 ${
              isOpen ? 'bg-slate-800' : 'bg-slate-900'
            }`}
          />
        </div>

        {/* Register Counter Unit */}
        <div
          className={`relative z-10 w-full h-[52px] rounded-md border flex flex-col justify-between p-1 shadow-lg transition-all duration-300 ${
            isTarget
              ? 'bg-gradient-to-b from-slate-800 to-slate-900 border-cyan-400 shadow-cyan-950/60'
              : 'bg-gradient-to-b from-slate-800/90 to-slate-950 border-slate-700/80 shadow-black/80'
          }`}
        >
          {/* Upper surface: Conveyor belt & Scanner */}
          <div className="w-full flex items-center justify-between gap-1 h-5 px-1 bg-slate-950/70 rounded border border-slate-800/80">
            {/* Conveyor track */}
            <div className={`flex-1 h-2 rounded-sm ${isOpen ? 'conveyor-belt' : 'bg-slate-900'}`} />

            {/* POS Monitor / Register screen */}
            <div className="relative w-3.5 h-3 bg-slate-900 rounded-sm border border-slate-600 flex items-center justify-center shrink-0">
              <div
                className={`w-2 h-1.5 rounded-[1px] ${
                  isOpen
                    ? isServing
                      ? 'bg-cyan-400 animate-pulse'
                      : 'bg-emerald-400'
                    : 'bg-rose-500'
                }`}
              />
            </div>
          </div>

          {/* Lower front panel of counter with subtle stainless steel accent */}
          <div className="w-full h-4 bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 rounded flex items-center justify-between px-1.5 border-t border-slate-700/50">
            {/* Status LED */}
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                isOpen ? 'bg-emerald-400 shadow-[0_0_4px_#34d399]' : 'bg-rose-500 shadow-[0_0_4px_#f43f5e]'
              }`}
            />
            {/* Scanner beam indicator */}
            {isServing && (
              <div className="w-6 h-0.5 bg-cyan-400/80 rounded-full shadow-[0_0_4px_#22d3ee] animate-pulse" />
            )}
            <div className="text-[8px] font-mono font-medium text-slate-400">POS</div>
          </div>
        </div>
      </div>
    </div>
  );
}

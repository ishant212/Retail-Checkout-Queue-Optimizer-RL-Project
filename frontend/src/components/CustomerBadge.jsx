import React from 'react';

/**
 * CustomerBadge - Solid circular badge displaying customer basket item count.
 * Formatted with spherical lighting, bold centered number, and basket-size color tiers.
 */
export default function CustomerBadge({
  basket = 0,
  size = 'md', // 'sm' | 'md' | 'lg'
  isEmpty = false,
  className = ''
}) {
  const sizeMap = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm',
    lg: 'w-9 h-9 text-base'
  };

  if (isEmpty) {
    return (
      <div
        className={`${sizeMap[size] || sizeMap.md} rounded-full border border-dashed border-slate-600/50 flex items-center justify-center bg-slate-900/30 shrink-0 ${className}`}
      />
    );
  }

  // Color gradient based on basket size
  let gradient = 'radial-gradient(circle at 35% 35%, #38bdf8 0%, #0284c7 60%, #0369a1 100%)'; // 1-4 items: Cyan
  let borderColor = 'rgba(56, 189, 248, 0.6)';
  let shadowColor = 'rgba(2, 132, 199, 0.4)';

  if (basket >= 12) {
    // 12+: Coral Red
    gradient = 'radial-gradient(circle at 35% 35%, #f87171 0%, #ef4444 60%, #b91c1c 100%)';
    borderColor = 'rgba(248, 113, 113, 0.7)';
    shadowColor = 'rgba(239, 68, 68, 0.4)';
  } else if (basket >= 8) {
    // 8-11: Amber/Orange
    gradient = 'radial-gradient(circle at 35% 35%, #fbbf24 0%, #f59e0b 60%, #b45309 100%)';
    borderColor = 'rgba(251, 191, 36, 0.7)';
    shadowColor = 'rgba(245, 158, 11, 0.4)';
  } else if (basket >= 5) {
    // 5-7: Emerald Green
    gradient = 'radial-gradient(circle at 35% 35%, #4ade80 0%, #22c55e 60%, #15803d 100%)';
    borderColor = 'rgba(74, 222, 128, 0.7)';
    shadowColor = 'rgba(34, 197, 94, 0.4)';
  }

  return (
    <div
      className={`${sizeMap[size] || sizeMap.md} rounded-full flex items-center justify-center font-bold text-white shrink-0 transition-transform duration-200 select-none ${className}`}
      style={{
        background: gradient,
        boxShadow: `0 3px 6px rgba(0, 0, 0, 0.6), 0 0 8px ${shadowColor}`,
        border: `1.5px solid ${borderColor}`
      }}
      title={`Customer basket: ${basket} items`}
    >
      <span className="leading-none drop-shadow-sm">{basket}</span>
    </div>
  );
}

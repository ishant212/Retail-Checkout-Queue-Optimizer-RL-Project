/**
 * OptiQueue AI - Shared Simulation Types and Constants
 */

export const LANE_TYPES = {
  EXPRESS: 'express',
  STANDARD: 'standard',
  LARGE_BASKET: 'large_basket',
  FLEXIBLE: 'flexible'
};

export const LANE_CONFIG = [
  { id: 0, label: '#0', type: 'express', name: 'Express', capacity: 6, limit: '≤ 5' },
  { id: 1, label: '#1', type: 'express', name: 'Express', capacity: 6, limit: '≤ 5' },
  { id: 2, label: '#2', type: 'standard', name: 'Standard', capacity: 8, limit: '≤ 15' },
  { id: 3, label: '#3', type: 'standard', name: 'Standard', capacity: 8, limit: '≤ 15' },
  { id: 4, label: '#4', type: 'standard', name: 'Standard', capacity: 8, limit: '≤ 15' },
  { id: 5, label: '#5', type: 'standard', name: 'Standard', capacity: 8, limit: '≤ 15' },
  { id: 6, label: '#6', type: 'standard', name: 'Standard', capacity: 8, limit: '≤ 15' },
  { id: 7, label: '#7', type: 'standard', name: 'Standard', capacity: 8, limit: '≤ 15' },
  { id: 8, label: '#8', type: 'large_basket', name: 'Bulk Cart', capacity: 8, limit: 'Any' },
  { id: 9, label: '#9', type: 'flexible', name: 'Self-Serve', capacity: 6, limit: '≤ 10' }
];

export const INITIAL_QUEUES = [3, 2, 4, 3, 4, 3, 3, 4, 4, 2];

export const INITIAL_LANE_CUSTOMERS = [
  {
    serving: { id: 101, basket: 2, wait: 1.2 },
    queue: [
      { id: 102, basket: 2, wait: 2.1 },
      { id: 103, basket: 1, wait: 1.5 },
      { id: 104, basket: 3, wait: 0.8 }
    ]
  },
  {
    serving: { id: 105, basket: 4, wait: 2.4 },
    queue: [
      { id: 106, basket: 4, wait: 1.9 },
      { id: 107, basket: 2, wait: 0.6 }
    ]
  },
  {
    serving: { id: 108, basket: 8, wait: 3.8 },
    queue: [
      { id: 109, basket: 12, wait: 3.2 },
      { id: 110, basket: 8, wait: 2.5 },
      { id: 111, basket: 6, wait: 1.8 },
      { id: 112, basket: 4, wait: 0.9 }
    ]
  },
  {
    serving: { id: 113, basket: 5, wait: 1.7 },
    queue: [
      { id: 114, basket: 7, wait: 2.3 },
      { id: 115, basket: 5, wait: 1.4 },
      { id: 116, basket: 3, wait: 0.7 }
    ]
  },
  {
    serving: { id: 117, basket: 9, wait: 4.1 },
    queue: [
      { id: 118, basket: 15, wait: 3.5 },
      { id: 119, basket: 11, wait: 2.8 },
      { id: 120, basket: 6, wait: 2.0 },
      { id: 121, basket: 5, wait: 1.1 }
    ]
  },
  {
    serving: { id: 122, basket: 6, wait: 2.9 },
    queue: [
      { id: 123, basket: 9, wait: 2.2 },
      { id: 124, basket: 4, wait: 1.5 },
      { id: 125, basket: 2, wait: 0.5 }
    ]
  },
  {
    serving: { id: 126, basket: 7, wait: 3.0 },
    queue: [
      { id: 127, basket: 6, wait: 2.4 },
      { id: 128, basket: 3, wait: 1.6 },
      { id: 129, basket: 2, wait: 0.8 }
    ]
  },
  {
    serving: { id: 130, basket: 10, wait: 4.5 },
    queue: [
      { id: 131, basket: 17, wait: 3.9 },
      { id: 132, basket: 9, wait: 3.0 },
      { id: 133, basket: 5, wait: 2.1 },
      { id: 134, basket: 4, wait: 1.2 }
    ]
  },
  {
    serving: { id: 135, basket: 14, wait: 5.2 },
    queue: [
      { id: 136, basket: 18, wait: 4.6 },
      { id: 137, basket: 16, wait: 3.8 },
      { id: 138, basket: 14, wait: 2.9 },
      { id: 139, basket: 11, wait: 1.7 }
    ]
  },
  {
    serving: { id: 140, basket: 3, wait: 1.5 },
    queue: [
      { id: 141, basket: 4, wait: 1.8 },
      { id: 142, basket: 2, wait: 0.9 }
    ]
  }
];

export const INITIAL_Q_VALUES = [
  { action: 2, label: 'Lane 2', q: 1.254 },
  { action: 5, label: 'Lane 5', q: 1.102 },
  { action: 3, label: 'Lane 3', q: 0.987 },
  { action: 6, label: 'Lane 6', q: 0.865 },
  { action: 0, label: 'Lane 0', q: 0.743 }
];

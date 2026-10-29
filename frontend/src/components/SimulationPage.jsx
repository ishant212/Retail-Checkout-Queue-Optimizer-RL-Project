import React, { useState } from 'react';
import MetricsBar from './MetricsBar';
import CheckoutFloor from './CheckoutFloor';
import IncomingCustomerPanel from './IncomingCustomerPanel';
import LaneInspector from './LaneInspector';
import RLDecisionPanel from './RLDecisionPanel';
import AnalyticsRow from './AnalyticsRow';

/**
 * SimulationPage - The primary operational view recreating the exact
 * layout and visual composition from the reference image.
 */
export default function SimulationPage({
  currentQueues = [3, 2, 4, 3, 4, 3, 3, 4, 4, 2],
  openStatus = [true, true, true, true, true, true, true, true, true, true],
  capacities = [6, 6, 8, 8, 8, 8, 8, 8, 8, 6],
  serviceRates = [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5],
  laneTypes = ['express', 'express', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 'large_basket', 'flexible'],
  laneCustomers = [],
  recommendedAction = 2,
  actionSemantic = 'Route to Lane #2',
  qValues = [],
  policyUpdate = null,
  stepCount = 0,
  activeAlgorithm = 'Double DQN',
  totalCustomers = 1054,
  currentlyInStore = 42,
  servedCustomers = 1012,
  avgWaitTime = 2.8,
  throughput = 43.2,
  activeCashiers = 8,
  totalLaborCost = 148.0,
  avgQueueLength = 4.2,
  incomingCustomer = {
    id: 'C#1055',
    basket: 7,
    isExpress: false,
    arrivalTime: '00:12:34'
  },
  nextArrivalSeconds = 12,
  waitTimeHistory = [],
  throughputHistory = [],
  rewardHistory = [],
  onSelectAction
}) {
  const [selectedLaneIdx, setSelectedLaneIdx] = useState(2);

  // Default to RL target lane if user hasn't explicitly selected another lane
  const activeInspectionLane = selectedLaneIdx !== null ? selectedLaneIdx : recommendedAction;

  return (
    <div className="w-full flex-1 flex flex-col gap-3 p-3.5 overflow-y-auto">
      {/* 1. Top KPI Summary Metric Cards (8 items) */}
      <MetricsBar
        totalCustomers={totalCustomers}
        currentlyInStore={currentlyInStore}
        servedCustomers={servedCustomers}
        avgWaitTime={avgWaitTime}
        throughput={throughput}
        activeCashiers={activeCashiers}
        totalCashiers={10}
        totalLaborCost={totalLaborCost}
        avgQueueLength={avgQueueLength}
      />

      {/* 2. Central Simulation Arena (Left: Incoming, Center: 10-Lane Floor, Right: Inspector & RL Decision) */}
      <div className="w-full grid grid-cols-12 gap-3 items-stretch">
        {/* Left Panel: Incoming Customer (2 columns on xl) */}
        <div className="col-span-12 lg:col-span-3 xl:col-span-2 flex flex-col gap-3">
          <IncomingCustomerPanel
            incomingCustomer={incomingCustomer}
            nextArrivalSeconds={nextArrivalSeconds}
            nextArrivalMax={15}
            avgBasketSize={8.7}
          />
        </div>

        {/* Center Panel: Supermarket Checkout Area (7 columns on xl) */}
        <div className="col-span-12 lg:col-span-6 xl:col-span-7 flex flex-col">
          <CheckoutFloor
            currentQueues={currentQueues}
            openStatus={openStatus}
            capacities={capacities}
            laneTypes={laneTypes}
            laneCustomers={laneCustomers}
            recommendedAction={recommendedAction}
            selectedLaneIdx={activeInspectionLane}
            onSelectLane={(id) => setSelectedLaneIdx(id)}
          />
        </div>

        {/* Right Panel: Lane Information & RL Action Selection (3 columns on xl) */}
        <div className="col-span-12 lg:col-span-3 xl:col-span-3 flex flex-col gap-3">
          <LaneInspector
            laneId={activeInspectionLane}
            openStatus={openStatus}
            currentQueues={currentQueues}
            capacities={capacities}
            serviceRates={serviceRates}
            laneTypes={laneTypes}
            laneCustomers={laneCustomers}
          />

          <RLDecisionPanel
            selectedAction={recommendedAction}
            actionSemantic={actionSemantic}
            policyName={`ε-greedy (ε = ${policyUpdate?.epsilon ?? '0.10'})`}
            qValues={qValues}
            policyUpdate={policyUpdate}
          />
        </div>
      </div>

      {/* 3. Bottom Analytics Panels (4 items) */}
      <AnalyticsRow
        currentQueues={currentQueues}
        currentWaitTime={avgWaitTime}
        currentThroughput={throughput}
        waitTimeHistory={waitTimeHistory}
        throughputHistory={throughputHistory}
        rewardHistory={rewardHistory}
      />
    </div>
  );
}

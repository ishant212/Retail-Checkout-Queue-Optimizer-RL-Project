import React, { useState, useEffect, useRef, useCallback } from 'react';
import SimulationHeader from './components/SimulationHeader';
import SimulationPage from './components/SimulationPage';
import AnalyticsPage from './components/AnalyticsPage';
import RLTrainingPage from './components/RLTrainingPage';
import SettingsPage from './components/SettingsPage';
import { api } from './api';
import {
  INITIAL_QUEUES,
  INITIAL_LANE_CUSTOMERS,
  INITIAL_Q_VALUES,
  LANE_CONFIG
} from './simulationTypes';

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('simulation'); // 'simulation' | 'analytics' | 'training' | 'settings'
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  // Active Model
  const [activeModelId, setActiveModelId] = useState('double_dqn');
  const [activeModelName, setActiveModelName] = useState('Double DQN');
  const [availableModels, setAvailableModels] = useState([]);
  const [episode, setEpisode] = useState(2450);

  // Simulation Controls
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [simSeconds, setSimSeconds] = useState(754); // 00:12:34
  const [stepCount, setStepCount] = useState(0);
  const [scenario, setScenario] = useState('peak'); // 'off_peak' | 'normal' | 'peak'

  // Simulation Arena State (10 Lanes)
  const [currentQueues, setCurrentQueues] = useState(INITIAL_QUEUES);
  const [openStatus, setOpenStatus] = useState([true, true, true, true, true, true, true, true, true, true]);
  const [laneTypes, setLaneTypes] = useState([
    'express',
    'express',
    'standard',
    'standard',
    'standard',
    'standard',
    'standard',
    'standard',
    'large_basket',
    'flexible'
  ]);
  const [capacities, setCapacities] = useState([6, 6, 8, 8, 8, 8, 8, 8, 8, 6]);
  const [serviceRates, setServiceRates] = useState([0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5]);
  const [laneCustomers, setLaneCustomers] = useState(INITIAL_LANE_CUSTOMERS);

  // RL Decision State
  const [recommendedAction, setRecommendedAction] = useState(2);
  const [actionSemantic, setActionSemantic] = useState('Route to Lane #2');
  const [qValues, setQValues] = useState(INITIAL_Q_VALUES);
  const [policyUpdate, setPolicyUpdate] = useState({
    modelId: 'double_dqn',
    modelName: 'Double DQN',
    epsilon: 0.1,
    gamma: 0.95,
    alpha: 0.1,
    tdTarget: 2.84,
    tdError: 0.124
  });

  // Incoming Customer & Countdown
  const [incomingCustomer, setIncomingCustomer] = useState({
    id: 'C#1055',
    basket: 7,
    isExpress: false,
    arrivalTime: '00:12:34'
  });
  const [nextArrivalSeconds, setNextArrivalSeconds] = useState(12);

  // KPIs
  const [totalCustomers, setTotalCustomers] = useState(1054);
  const [servedCustomers, setServedCustomers] = useState(1012);
  const [currentlyInStore, setCurrentlyInStore] = useState(42);
  const [avgWaitTime, setAvgWaitTime] = useState(2.8);
  const [throughput, setThroughput] = useState(43.2);
  const [totalLaborCost, setTotalLaborCost] = useState(148.0);
  const [avgQueueLength, setAvgQueueLength] = useState(4.2);

  // Charts Historical Data
  const [waitTimeHistory, setWaitTimeHistory] = useState([
    { time: '0', wait: 2.1 },
    { time: '5', wait: 4.2 },
    { time: '10', wait: 4.8 },
    { time: '15', wait: 3.5 },
    { time: '20', wait: 4.3 },
    { time: '25', wait: 2.7 },
    { time: '30', wait: 2.8 }
  ]);
  const [throughputHistory, setThroughputHistory] = useState([
    { time: '0', tp: 22 },
    { time: '5', tp: 34 },
    { time: '10', tp: 41 },
    { time: '15', tp: 48 },
    { time: '20', tp: 62 },
    { time: '25', tp: 51 },
    { time: '30', tp: 43.2 }
  ]);
  const [rewardHistory, setRewardHistory] = useState([]);

  // Auto-play interval ref
  const playTimerRef = useRef(null);
  const clockTimerRef = useRef(null);

  // Format Elapsed Time HH:MM:SS
  const formatSimTime = (totalSec) => {
    const hrs = String(Math.floor(totalSec / 3600)).padStart(2, '0');
    const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
    const secs = String(totalSec % 60).padStart(2, '0');
    return `${hrs}:${mins}:${secs}`;
  };

  // Helper to apply authoritative state from simulation backend
  const applySimulationPayload = useCallback((data) => {
    if (!data) return;
    if (data.state) setCurrentQueues(data.state);
    if (data.openStatus) setOpenStatus(data.openStatus);
    if (data.laneTypes) setLaneTypes(data.laneTypes);
    if (data.capacities) setCapacities(data.capacities);
    if (data.serviceRates) setServiceRates(data.serviceRates);
    if (data.action !== undefined && data.action < 10) {
      setRecommendedAction(data.action);
    } else if (data.recommendedAction !== undefined) {
      setRecommendedAction(data.recommendedAction);
    }
    if (data.actionSemantic) {
      setActionSemantic(data.actionSemantic);
    }
    if (data.laneCustomers) {
      setLaneCustomers(data.laneCustomers);
    }
    if (data.policyUpdate) {
      setPolicyUpdate(data.policyUpdate);
      if (data.policyUpdate.topQValues) {
        setQValues(data.policyUpdate.topQValues);
      }
    }
    if (data.scenario) {
      setScenario(data.scenario);
    }

    // Authoritative Metrics from backend
    if (data.totalCustomers !== undefined) setTotalCustomers(data.totalCustomers);
    if (data.servedCustomers !== undefined) setServedCustomers(data.servedCustomers);
    if (data.currentlyInStore !== undefined) setCurrentlyInStore(data.currentlyInStore);
    if (data.avgWaitTime !== undefined) setAvgWaitTime(data.avgWaitTime);
    if (data.throughput !== undefined) setThroughput(data.throughput);
    if (data.totalLaborCost !== undefined) setTotalLaborCost(data.totalLaborCost);
    if (data.avgQueueLength !== undefined) setAvgQueueLength(data.avgQueueLength);
  }, []);

  // Initialize Backend Connection & State
  useEffect(() => {
    let isMounted = true;

    async function init() {
      try {
        const status = await api.getStatus();
        if (isMounted) {
          setIsBackendConnected(status.status === 'online');
          if (status.models) setAvailableModels(status.models);
          if (status.activeModelId) setActiveModelId(status.activeModelId);
          if (status.activeModelName) setActiveModelName(status.activeModelName);
        }

        const initial = await api.resetInteractive('double_dqn', 'peak');
        if (isMounted && initial) {
          applySimulationPayload(initial);
        }
      } catch (err) {
        console.warn('Backend connection warning:', err.message);
        setIsBackendConnected(false);
      }
    }

    init();
    return () => {
      isMounted = false;
    };
  }, [applySimulationPayload]);

  // Scenario Change Handler
  const handleScenarioChange = useCallback(async (newScenario) => {
    setScenario(newScenario);
    try {
      if (isBackendConnected) {
        await api.setScenario(newScenario);
        const resetData = await api.resetInteractive(activeModelId, newScenario);
        if (resetData) {
          applySimulationPayload(resetData);
        }
      }
    } catch (err) {
      console.warn('Error switching scenario:', err.message);
    }
  }, [isBackendConnected, activeModelId, applySimulationPayload]);

  // Execution Step function (calls real backend or fallback generator)
  const handleStep = useCallback(async () => {
    setStepCount((prev) => prev + 1);
    setSimSeconds((prev) => prev + 2);

    try {
      if (isBackendConnected) {
        const stepData = await api.stepInteractive(null, activeModelId, scenario);
        if (stepData) {
          applySimulationPayload(stepData);

          // Update incoming customer
          const nextBasket = stepData.incomingBasketSize || Math.floor(Math.random() * 14) + 1;
          setIncomingCustomer({
            id: `C#${1055 + stepCount + 1}`,
            basket: nextBasket,
            isExpress: nextBasket <= 5,
            arrivalTime: formatSimTime(simSeconds + 2)
          });
          setNextArrivalSeconds(Math.floor(Math.random() * 10) + 8);
          return;
        }
      }
    } catch (err) {
      console.warn('Step interactive error:', err.message);
    }

    // Deterministic simulation step fallback for Demo Mode
    setCurrentQueues((prev) => {
      const next = [...prev];
      const target = (recommendedAction + 1) % 10;
      setRecommendedAction(target);
      setActionSemantic(`Route to Lane #${target}`);

      // Customer enters target lane
      if (next[target] < 6) next[target] += 1;
      // Some other lane finishes checkout
      const finishedLane = (target + 3) % 10;
      if (next[finishedLane] > 1) next[finishedLane] -= 1;

      return next;
    });

    setServedCustomers((prev) => prev + 1);
    setTotalCustomers((prev) => prev + 1);
    const nextBasket = Math.floor(Math.random() * 14) + 1;
    setIncomingCustomer({
      id: `C#${1055 + stepCount + 1}`,
      basket: nextBasket,
      isExpress: nextBasket <= 5,
      arrivalTime: formatSimTime(simSeconds + 2)
    });
    setNextArrivalSeconds(Math.floor(Math.random() * 10) + 8);
  }, [
    isBackendConnected,
    activeModelId,
    scenario,
    stepCount,
    simSeconds,
    recommendedAction,
    applySimulationPayload
  ]);

  // Next arrival countdown timer
  useEffect(() => {
    const timer = setInterval(() => {
      setNextArrivalSeconds((prev) => {
        if (prev <= 1) {
          if (isPlaying) {
            handleStep();
          }
          return 14;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isPlaying, handleStep]);

  // Simulation execution runner when isPlaying is true
  useEffect(() => {
    if (isPlaying) {
      const intervalMs = Math.max(200, 1000 / speedMultiplier);
      playTimerRef.current = setInterval(() => {
        handleStep();
      }, intervalMs);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, speedMultiplier, handleStep]);

  // Change model
  const handleSelectModel = async (modelId) => {
    try {
      setActiveModelId(modelId);
      const res = await api.selectModel(modelId);
      if (res.activeModelName) setActiveModelName(res.activeModelName);
    } catch (err) {
      console.warn('Select model error:', err.message);
    }
  };

  // Update Lane Configuration
  const handleUpdateLaneConfig = (laneId, updates) => {
    if (updates.isOpen !== undefined) {
      setOpenStatus((prev) => {
        const copy = [...prev];
        copy[laneId] = updates.isOpen;
        return copy;
      });
    }
  };

  const activeCashiersCount = openStatus.filter(Boolean).length;

  return (
    <div className="w-screen h-screen flex flex-col bg-[#0a0e17] text-slate-100 overflow-hidden font-sans">
      {/* 1. Compact Top Navigation Bar */}
      <SimulationHeader
        activeTab={activeTab}
        onTabChange={(tab) => setActiveTab(tab)}
        scenario={scenario}
        onScenarioChange={handleScenarioChange}
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying(!isPlaying)}
        speed={speedMultiplier}
        onSpeedChange={(val) => setSpeedMultiplier(val)}
        simTime={formatSimTime(simSeconds)}
        episode={episode}
        activeAlgorithm={activeModelName}
        isBackendConnected={isBackendConnected}
        onStep={handleStep}
      />

      {/* 2. Main View Area based on Active Tab */}
      <main className="flex-1 overflow-hidden flex flex-col">
        {activeTab === 'simulation' && (
          <SimulationPage
            currentQueues={currentQueues}
            openStatus={openStatus}
            capacities={capacities}
            serviceRates={serviceRates}
            laneTypes={laneTypes}
            laneCustomers={laneCustomers}
            recommendedAction={recommendedAction}
            actionSemantic={actionSemantic}
            qValues={qValues}
            policyUpdate={policyUpdate}
            stepCount={stepCount}
            activeAlgorithm={activeModelName}
            totalCustomers={totalCustomers}
            currentlyInStore={currentlyInStore}
            servedCustomers={servedCustomers}
            avgWaitTime={avgWaitTime}
            throughput={throughput}
            activeCashiers={activeCashiersCount}
            totalLaborCost={totalLaborCost}
            avgQueueLength={avgQueueLength}
            incomingCustomer={incomingCustomer}
            nextArrivalSeconds={nextArrivalSeconds}
            waitTimeHistory={waitTimeHistory}
            throughputHistory={throughputHistory}
            rewardHistory={rewardHistory}
            onSelectAction={(act) => setRecommendedAction(act)}
          />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsPage isBackendConnected={isBackendConnected} />
        )}

        {activeTab === 'training' && (
          <RLTrainingPage
            activeModelId={activeModelId}
            availableModels={availableModels}
            onSelectModel={handleSelectModel}
            isBackendConnected={isBackendConnected}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsPage
            openStatus={openStatus}
            capacities={capacities}
            serviceRates={serviceRates}
            laneTypes={laneTypes}
            onUpdateLaneConfig={handleUpdateLaneConfig}
          />
        )}
      </main>
    </div>
  );
}

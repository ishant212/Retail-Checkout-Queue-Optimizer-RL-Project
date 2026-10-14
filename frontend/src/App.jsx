import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import OverviewView from './components/OverviewView';
import PolicyStudioView from './components/PolicyStudioView';
import SimPyView from './components/SimPyView';
import BenchmarkLogsView from './components/BenchmarkLogsView';
import ConfigModal from './components/ConfigModal';
import CheckoutLanesView from './components/CheckoutLanesView';
import SimulationControls from './components/SimulationControls';
import ComparativeAnalysisView from './components/ComparativeAnalysisView';
import { api } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Status & Telemetry
  const [backendConnected, setBackendConnected] = useState(false);
  const [trainedEpisodes, setTrainedEpisodes] = useState(250);
  const [modelEpisodes, setModelEpisodes] = useState({
    q_learning: 250,
    sarsa: 250,
    bandit: 250,
    reinforce: 200,
    a2c: 200,
    ppo: 200,
    dqn: 200,
    double_dqn: 250,
    ddpg: 150
  });
  const [serviceRates, setServiceRates] = useState([0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.5, 0.5]);
  const [maxQueueLen, setMaxQueueLen] = useState(6);
  const [availableModels, setAvailableModels] = useState([]);

  // Interactive Simulation State (10 Lanes & 10 States)
  const [currentQueues, setCurrentQueues] = useState(Array(10).fill(0));
  const [openStatus, setOpenStatus] = useState([true, true, true, true, false, false, false, false, false, false]);
  const [laneTypes, setLaneTypes] = useState(['express', 'standard', 'standard', 'large_basket', 'standard', 'standard', 'express', 'large_basket', 'flexible', 'flexible']);
  const [capacities, setCapacities] = useState([4, 6, 6, 8, 6, 6, 4, 8, 6, 6]);
  const [currentServing, setCurrentServing] = useState(Array(10).fill(null));
  const [incomingBasketSize, setIncomingBasketSize] = useState(5);
  const [isExpressEligible, setIsExpressEligible] = useState(true);
  const [counterUtilization, setCounterUtilization] = useState(Array(10).fill(0));
  const [lastAction, setLastAction] = useState(null);
  const [lastActionValid, setLastActionValid] = useState(true);
  const [lastActionReason, setLastActionReason] = useState('');
  const [lastServed, setLastServed] = useState(Array(10).fill(false));
  const [qValues, setQValues] = useState(Array(10).fill(0));
  const [allQValues, setAllQValues] = useState(Array(41).fill(0));
  const [recommendedAction, setRecommendedAction] = useState(0);
  const [rewardBreakdown, setRewardBreakdown] = useState({});
  const [stateComponents, setStateComponents] = useState({});
  const [congestionDuration, setCongestionDuration] = useState({ step: 0, continuous: 0, total: 0 });

  // Active Model & Policy Controls (Defaults to Proposed Double DQN)
  const [isPlaying, setIsPlaying] = useState(false);
  const [mode, setMode] = useState('double_dqn');
  const [speed, setSpeed] = useState(1000);
  const [stepCount, setStepCount] = useState(0);
  const [totalReward, setTotalReward] = useState(0);

  // Benchmark & Historical Data
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [simpyData, setSimpyData] = useState(null);

  const autoPlayTimerRef = useRef(null);

  // Initial Data Fetch
  useEffect(() => {
    let isMounted = true;

    async function initData() {
      try {
        const status = await api.getStatus();
        if (isMounted) {
          setBackendConnected(status.status === 'online');
          setTrainedEpisodes(status.trainedEpisodes);
          if (status.modelEpisodes) setModelEpisodes(status.modelEpisodes);
          setServiceRates(status.serviceRates);
          setMaxQueueLen(status.maxQueueLen);
          if (status.models) setAvailableModels(status.models);
        }

        // Run default multi-model benchmark
        const bench = await api.runBenchmark(100, status.serviceRates);
        if (isMounted) {
          setBenchmarkData(bench);
        }

        // Reset interactive state
        const initialInteractive = await api.resetInteractive('double_dqn');
        if (isMounted) {
          if (initialInteractive.state) setCurrentQueues(initialInteractive.state);
          if (initialInteractive.openStatus) setOpenStatus(initialInteractive.openStatus);
          if (initialInteractive.laneTypes) setLaneTypes(initialInteractive.laneTypes);
          if (initialInteractive.capacities) setCapacities(initialInteractive.capacities);
          if (initialInteractive.currentServing) setCurrentServing(initialInteractive.currentServing);
          if (initialInteractive.counterUtilization) setCounterUtilization(initialInteractive.counterUtilization);
          if (initialInteractive.incomingBasketSize !== undefined) setIncomingBasketSize(initialInteractive.incomingBasketSize);
          if (initialInteractive.isExpressEligible !== undefined) setIsExpressEligible(initialInteractive.isExpressEligible);
          setQValues(initialInteractive.qValues || Array(10).fill(0));
          if (initialInteractive.allQValues) setAllQValues(initialInteractive.allQValues);
          setRecommendedAction(initialInteractive.recommendedAction ?? 0);
          setStateComponents(initialInteractive.stateComponents || {});
          setCongestionDuration(initialInteractive.congestionDuration || {});
        }

        // Fetch initial SimPy arena run
        const simpyRes = await api.runSimPy({ arrivalRate: 1.0, simulationTime: 120 });
        if (isMounted) {
          setSimpyData(simpyRes);
        }
      } catch (err) {
        console.warn('Initial backend fetch encountered error, fallback active:', err);
      }
    }

    initData();

    // Health check ping interval
    const pingInterval = setInterval(async () => {
      try {
        const s = await api.getStatus();
        if (isMounted) {
          setBackendConnected(s.status === 'online');
          if (s.models) setAvailableModels(s.models);
          if (s.modelEpisodes) setModelEpisodes(s.modelEpisodes);
        }
      } catch {
        if (isMounted) setBackendConnected(false);
      }
    }, 8000);

    return () => {
      isMounted = false;
      clearInterval(pingInterval);
    };
  }, []);

  // Sync Q-values and action recommendation when mode switches
  useEffect(() => {
    if (['q_learning', 'sarsa', 'bandit', 'reinforce', 'a2c', 'ppo', 'dqn', 'double_dqn', 'ddpg'].includes(mode)) {
      api.selectModel(mode).catch(() => {});
      api.getQValues(currentQueues[0] || 0, currentQueues[1] || 0, currentQueues[2] || 0, mode)
        .then((res) => {
          setQValues(res.qValues || Array(10).fill(0));
          if (res.allQValues) setAllQValues(res.allQValues);
          setRecommendedAction(res.bestAction ?? 0);
        })
        .catch(console.error);
    }
  }, [mode]);

  // Step Handler
  const handleStep = async (actionOverride = null) => {
    try {
      const res = await api.stepInteractive(actionOverride, mode);
      setCurrentQueues(res.nextState);
      setLastAction(res.action);
      setLastServed(res.served || Array(10).fill(false));
      setQValues(res.qValues || Array(10).fill(0));
      if (res.allQValues) setAllQValues(res.allQValues);
      setRecommendedAction(res.recommendedAction);
      setStepCount(res.step);
      setTotalReward((prev) => res.cumulativeReward ?? (prev + res.reward));

      if (res.openStatus) setOpenStatus(res.openStatus);
      if (res.laneTypes) setLaneTypes(res.laneTypes);
      if (res.capacities) setCapacities(res.capacities);
      if (res.currentServing) setCurrentServing(res.currentServing);
      if (res.incomingBasketSize !== undefined) setIncomingBasketSize(res.incomingBasketSize);
      if (res.isExpressEligible !== undefined) setIsExpressEligible(res.isExpressEligible);
      if (res.counterUtilization) setCounterUtilization(res.counterUtilization);
      setLastActionValid(res.actionValid ?? true);
      setLastActionReason(res.actionReason ?? '');
      setRewardBreakdown(res.rewardBreakdown ?? {});
      setStateComponents(res.stateComponents ?? {});
      setCongestionDuration(res.congestionDuration ?? {});
    } catch (err) {
      console.error('Interactive step failed:', err);
    }
  };

  // Reset Handler
  const handleReset = async () => {
    setIsPlaying(false);
    try {
      const res = await api.resetInteractive(mode);
      setCurrentQueues(res.state);
      setLastAction(null);
      setLastServed(Array(10).fill(false));
      setQValues(res.qValues || Array(10).fill(0));
      if (res.allQValues) setAllQValues(res.allQValues);
      setRecommendedAction(res.recommendedAction);
      setStepCount(0);
      setTotalReward(0);

      if (res.openStatus) setOpenStatus(res.openStatus);
      if (res.laneTypes) setLaneTypes(res.laneTypes);
      if (res.capacities) setCapacities(res.capacities);
      if (res.currentServing) setCurrentServing(res.currentServing);
      if (res.incomingBasketSize !== undefined) setIncomingBasketSize(res.incomingBasketSize);
      if (res.isExpressEligible !== undefined) setIsExpressEligible(res.isExpressEligible);
      setLastActionValid(true);
      setLastActionReason('Session Reset');
      setRewardBreakdown({});
      setStateComponents(res.stateComponents ?? {});
      setCongestionDuration(res.congestionDuration ?? {});
    } catch (err) {
      console.error('Reset failed:', err);
    }
  };

  // Auto-play loop
  useEffect(() => {
    if (isPlaying) {
      autoPlayTimerRef.current = setInterval(() => {
        handleStep();
      }, speed);
    } else {
      if (autoPlayTimerRef.current) {
        clearInterval(autoPlayTimerRef.current);
        autoPlayTimerRef.current = null;
      }
    }

    return () => {
      if (autoPlayTimerRef.current) {
        clearInterval(autoPlayTimerRef.current);
      }
    };
  }, [isPlaying, speed, mode]);

  // Run Benchmark Handler
  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await api.runBenchmark(100, serviceRates, mode);
      setBenchmarkData(res);
    } catch (err) {
      console.error('Benchmark run failed:', err);
    } finally {
      setIsBenchmarking(false);
    }
  };

  // Retrain Handler
  const handleRetrain = async (config) => {
    setIsTraining(true);
    try {
      const res = await api.train(config);
      setTrainedEpisodes(res.episodes);
      if (res.modelEpisodes) setModelEpisodes(res.modelEpisodes);
      setTrainingHistory(res.history);
      if (config.serviceRates) {
        setServiceRates(config.serviceRates);
      }
      // Re-run multi-model benchmark with newly trained parameters
      const bench = await api.runBenchmark(100, config.serviceRates);
      setBenchmarkData(bench);
      // Re-run SimPy arena
      const simpyRes = await api.runSimPy({ arrivalRate: 1.0, simulationTime: 120 });
      setSimpyData(simpyRes);
    } catch (err) {
      console.error('Training failed:', err);
    } finally {
      setIsTraining(false);
    }
  };

  const totalWaiting = currentQueues.reduce((a, b) => a + b, 0);

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isCollapsed={isSidebarCollapsed}
        setIsCollapsed={setIsSidebarCollapsed}
        backendConnected={backendConnected}
        trainedEpisodes={trainedEpisodes}
      />

      {/* Main Content Area */}
      <div className="main-content">
        <Header
          activeTab={activeTab}
          onRunBenchmark={handleRunBenchmark}
          onResetQueues={handleReset}
          isBenchmarking={isBenchmarking}
          totalWaiting={totalWaiting}
          onToggleMobileSidebar={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
          activeModelName={availableModels.find((m) => m.id === mode)?.name || mode.replace('_', ' ').toUpperCase()}
        />

        <main className="page-body">
          {activeTab === 'overview' && (
            <OverviewView
              benchmarkData={benchmarkData}
              currentQueues={currentQueues}
              serviceRates={serviceRates}
              openStatus={openStatus}
              laneTypes={laneTypes}
              capacities={capacities}
              currentServing={currentServing}
              incomingBasketSize={incomingBasketSize}
              isExpressEligible={isExpressEligible}
              counterUtilization={counterUtilization}
              maxQueueLen={maxQueueLen}
              lastAction={lastAction}
              lastServed={lastServed}
              qValues={qValues}
              allQValues={allQValues}
              recommendedAction={recommendedAction}
              isPlaying={isPlaying}
              onTogglePlay={() => setIsPlaying(!isPlaying)}
              onStep={() => handleStep()}
              onReset={handleReset}
              mode={mode}
              setMode={setMode}
              speed={speed}
              setSpeed={setSpeed}
              stepCount={stepCount}
              totalReward={totalReward}
              lastActionValid={lastActionValid}
              lastActionReason={lastActionReason}
              rewardBreakdown={rewardBreakdown}
              stateComponents={stateComponents}
              congestionDuration={congestionDuration}
              onManualDispatch={(laneId) => handleStep(laneId)}
              onManualAction={(act) => handleStep(act)}
            />
          )}

          {activeTab === 'comparison' && (
            <ComparativeAnalysisView />
          )}

          {activeTab === 'simulation' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <SimulationControls
                isPlaying={isPlaying}
                onTogglePlay={() => setIsPlaying(!isPlaying)}
                onStep={() => handleStep()}
                onReset={handleReset}
                mode={mode}
                setMode={setMode}
                speed={speed}
                setSpeed={setSpeed}
                stepCount={stepCount}
                totalReward={totalReward}
                lastActionValid={lastActionValid}
                lastActionReason={lastActionReason}
                onManualAction={(act) => handleStep(act)}
              />

              <CheckoutLanesView
                queues={currentQueues}
                maxQueueLen={maxQueueLen}
                serviceRates={serviceRates}
                openStatus={openStatus}
                laneTypes={laneTypes}
                capacities={capacities}
                currentServing={currentServing}
                incomingBasketSize={incomingBasketSize}
                isExpressEligible={isExpressEligible}
                counterUtilization={counterUtilization}
                lastAction={lastAction}
                lastServed={lastServed}
                qValues={qValues}
                recommendedAction={recommendedAction}
                mode={mode}
                onManualDispatch={(laneId) => handleStep(laneId)}
                onManualAction={(act) => handleStep(act)}
              />
            </div>
          )}

          {activeTab === 'policy' && (
            <PolicyStudioView
              trainingHistory={trainingHistory}
              onRetrain={handleRetrain}
              isTraining={isTraining}
              maxQueueLen={maxQueueLen}
            />
          )}

          {activeTab === 'simpy' && (
            <SimPyView initialData={simpyData} />
          )}

          {activeTab === 'benchmark' && (
            <BenchmarkLogsView benchmarkData={benchmarkData} />
          )}

          {activeTab === 'config' && (
            <ConfigModal
              serviceRates={serviceRates}
              modelEpisodes={modelEpisodes}
              onUpdateConfig={handleRetrain}
              isTraining={isTraining}
            />
          )}
        </main>
      </div>
    </div>
  );
}

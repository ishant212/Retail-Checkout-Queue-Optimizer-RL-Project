# OptiQueue — Reinforcement Learning Algorithm Audit Summary

**Audit Date:** October 14, 2026  
**Scope:** 9 RL Algorithms, Shared Environment (`RetailQueueEnv`), SimPy Simulation, Centralized Reward Function, and Test Suite.  
**Mode:** Strictly Read-Only (Zero Code Alterations).

---

## Algorithm Audit Verdict Matrix

| Algorithm | Standard Mathematical Structure | Main Audit Findings | Test Status | Verdict | Recommended Fix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q-Learning** | Tabular Off-Policy TD Control: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ | Missing terminal masking `(1 - done)` at episode horizon ($t=25$). Otherwise follows standard Bellman optimality equation. | PASS (Unit + API) | **PARTIAL** | Add `done` parameter to `update()`; zero out future value when `done=True`. |
| **SARSA** | Tabular On-Policy TD Control: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s', a') - Q(s,a)]$ | On-policy violation: `next_act` sampled for TD target is discarded; next loop step calls `choose_action()` again with stochastic exploration. Missing `(1 - done)`. | PASS (Unit + API) | **PARTIAL** | Cache and execute `next_act` in the subsequent time step; add `done` masking. |
| **Contextual Bandit (UCB1)** | Contextual Upper Confidence Bound: $A_t = \arg\max_a [Q(c,a) + c\sqrt{2\ln N(c)/N(c,a)}]$ | Correct 7-tuple context feature grouping and UCB1 formula. Uses sample mean ($1/N$) instead of exponential moving average; redundant $\epsilon$-greedy exploration added. | PASS (Unit + API) | **PASS** | Replace sample mean with recency-weighted update $\alpha$ for non-stationary arrival dynamics. |
| **REINFORCE (w/ Baseline)** | Episodic Monte Carlo Policy Gradient: $\theta \leftarrow \theta + \alpha \gamma^t (G_t - V(s_t)) \nabla \ln \pi$ | Textbook implementation with learned state-value baseline and $\gamma^t$ discounting. Softmax analytical policy gradient correct. | PASS (Unit + API) | **PASS** | Maintain current implementation. |
| **A2C** | 1-Step Advantage Actor-Critic: $\delta_t = r + \gamma V(s') - V(s)$, Actor updated via TD advantage | Core advantage calculation correct. Flawed entropy gradient calculation ($\nabla_\pi H$ rather than $\nabla_z H$). Missing `(1 - done)` in TD target. | PASS (Unit + API) | **PARTIAL** | Fix entropy Jacobian w.r.t logits: $\nabla_z H = -\pi (\ln \pi + H)$; add `done` masking to TD target. |
| **PPO** | Clipped Surrogate Objective with Generalized Advantage Estimation (GAE-$\lambda$) | GAE($\lambda$) recursion and analytical clipped surrogate ratio $r_t(\theta)$ implemented correctly. Buffer runs full-batch rather than mini-batch SGD. Missing `(1 - done)`. | PASS (Unit + API) | **PASS** | Add mini-batch shuffling during epoch iterations and add `(1 - done)` masking to GAE return calculation. |
| **DQN** | Deep Q-Network with Experience Replay and Polyak Target Network | Core Bellman target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ and soft target update ($\tau=0.1$) correct. Replay buffer stores 4-tuples, omitting `done`. | PASS (Unit + API) | **PASS** | Extend `ReplayBuffer` to 5-tuples `(s, a, r, s', done)` and mask terminal transitions. |
| **Double DQN (Proposed)** | Decoupled Action Selection and Evaluation: $r + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{online}}(s', a'))$ | Exact mathematical match to Hasselt et al. (2015). Successfully mitigates Q-value overestimation bias. Replay buffer missing `done`. | PASS (Unit + API) | **PASS** | Add `done` flag to transition tuple and zero out future Q-value at episode termination. |
| **DDPG** | Deterministic Policy Gradient: $\nabla_\theta J = \frac{1}{N}\sum \nabla_a Q(s,a)\nabla_\theta \mu(s)$ | **CRITICAL FAILURE:** Critic is never differentiated w.r.t actions; actor is trained with an arbitrary heuristic gradient array (`-1.0` / `-0.1`). Continuous DDPG is incompatible with 24 discrete actions without relaxation. | PASS (Unit + API) | **FAIL** | Implement Gumbel-Softmax continuous relaxation or replace with Wolpertinger / SAC-Discrete architecture. |

---

## Shared Infrastructure Audit Summary

| Component | Status | Finding | Actionable Fix |
| :--- | :--- | :--- | :--- |
| **Centralized Reward** | **VERIFIED** | 8-component formula with step-wise congestion duration and non-duplicative service reward is mathematically exact. | Keep as is. |
| **10-Lane Environment** | **VERIFIED** | State dimension (35 features) and Action space (24 discrete actions) properly registered and bounded. | Keep as is. |
| **SimPy Discrete-Event Engine** | **DEFECT** | `SimPyCheckoutSimulator` feeds truncated 10-int state to agents during live simulation instead of 35-dim continuous vector or 22-tuple key. | Pass full environment state from SimPy monitor into `agent.choose_action()`. |
| **Model Persistence** | **DEFECT** | No model checkpoint saving (`save`) or loading (`load`) implemented across any agent. | Add serialization methods to `BaseRLAgent` using NumPy or pickle. |
| **Automated Test Suite** | **VERIFIED** | 25/25 tests pass in 127s. Validates API contracts and execution pipelines, but does not verify gradient validity. | Add unit tests asserting non-zero loss reduction and gradient chain connectivity. |

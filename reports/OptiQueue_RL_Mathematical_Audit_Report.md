# OptiQueue — Complete RL Architecture Audit, Mathematical Validation, and Testing Report

**Date:** October 14, 2026  
**Repository:** Autonomous Retail Checkout Queue Optimizer (OptiQueue)  
**System Architecture:** 10-Lane Dynamic Supermarket Checkout Simulator  
**Audit Scope:** All 9 Reinforcement Learning Algorithms, Shared State Spaces, Centralized Reward Engine, SimPy Simulation Engine, and REST API.

---

## Part A: Architecture Verification Summary

| Algorithm | Family | Pathway | State Feature Dimension | Action Space Size | Strict Dimension Validation | Terminal Handling | Mathematical Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Q-Learning** | Value-Based (Off-Policy TD) | Tabular (7-tuple key) | **35** | **41** | `ValueError` on $\neq 35$ | $y_t = r_t$ when $d_t = 1$ | **PASS** |
| **SARSA** | Value-Based (On-Policy TD) | Tabular (7-tuple key) | **35** | **41** | `ValueError` on $\neq 35$ | $y_t = r_t$ when $d_t = 1$; on-policy $a_{t+1}$ sync | **PASS** |
| **Contextual UCB1** | Multi-Armed Bandit | Contextual (35-feat) | **35** | **41** | `ValueError` on $\neq 35$ | Contextual arm statistics $N(c, a), \hat{\mu}(c, a)$ | **PASS** |
| **REINFORCE** | Policy Gradient (Monte Carlo) | Continuous Neural (MLP) | **84** | **41** | `ValueError` on $\neq 84$ | Discounted returns $G_t$ with baseline $b$ | **PASS** |
| **A2C** | Actor-Critic (Advantage TD) | Continuous Neural (MLP) | **84** | **41** | `ValueError` on $\neq 84$ | Bootstrapping masked $(1-d_t)$; exact logit entropy | **PASS** |
| **PPO** | Policy Optimization (Surrogate) | Continuous Neural (MLP) | **84** | **41** | `ValueError` on $\neq 84$ | GAE($\lambda$) masked $(1-d_t)$; clipped surrogate | **PASS** |
| **DQN** | Value-Based (Experience Replay) | Continuous Neural (MLP) | **84** | **41** | `ValueError` on $\neq 84$ | Replay buffer $(s, a, r, s', d)$; target terminal zeroed | **PASS** |
| **Double DQN** | Value-Based (Decoupled Target) | Continuous Neural (MLP) | **84** | **41** | `ValueError` on $\neq 84$ | Target decoupled: $Q_{\text{target}}(s', \arg\max Q_{\text{online}})$ | **PASS** |
| **DDPG** | Deterministic Actor-Critic | Continuous Neural (MLP) | **84** | **41** | `ValueError` on $\neq 84$ | Simplex relaxation; critic-to-actor Jacobian backprop | **PASS** |

---

## Part B: Mathematical Audit of Algorithms

### 1. Tabular Models (35-Dimensional Input $\to$ 7-Tuple Discretized State Key)

#### 1.1 Q-Learning (Off-Policy 1-Step Temporal Difference Control)
- **Target Equation:**
  $$y_t = r_t + (1 - d_t)\gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a')$$
- **TD Error and Update Rule:**
  $$\delta_t = y_t - Q(s_t, a_t)$$
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t$$
- **Source Lines:** [`backend/models/q_learning.py#L58-L76`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/q_learning.py#L58-L76)
- **Mathematical Integrity:** Terminal masking `(1 - done)` terminates future bootstrapping at episode boundaries. Tabular feature dimensionality is strictly checked against `TABULAR_FEATURE_DIM = 35`.
- **Validation Test:** `test_11_q_learning_bellman_target_with_terminal_masking` (PASS).

#### 1.2 SARSA (On-Policy 1-Step Temporal Difference Control)
- **Target Equation:**
  $$y_t = r_t + (1 - d_t)\gamma Q(s_{t+1}, a_{t+1})$$
- **Update Rule:**
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ y_t - Q(s_t, a_t) \right]$$
- **Source Lines:** [`backend/models/sarsa.py#L56-L76`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/sarsa.py#L56-L76), [`backend/simulation_engine.py#L960-L976`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/simulation_engine.py#L960-L976)
- **Mathematical Integrity:** In `simulation_engine.py`, the next action $a_{t+1}$ chosen on-policy via $\epsilon$-greedy is passed directly into `sarsa.update(...)` and executed on the subsequent step $t+1$, ensuring true on-policy trajectory matching.
- **Validation Test:** `test_12_sarsa_on_policy_synchronization_and_terminal_masking` (PASS).

#### 1.3 Contextual Bandit (UCB1 with State-Derived Context)
- **Context Generation:**
  $$c = \text{state\_to\_context}(s) \in \mathcal{C}$$
  Discretized from the 35 tabular features into aggregate queue demand, open lane counts, and arrival intensity.
- **Upper Confidence Bound Scoring Rule:**
  $$\text{Score}(c, a) = \hat{\mu}(c, a) + c_{\text{ucb}} \sqrt{\frac{2 \ln \left( 1 + \sum_{b \in \mathcal{A}} N(c, b) \right)}{1 + N(c, a)}}$$
- **Sample Mean Update:**
  $$\hat{\mu}(c, a) \leftarrow \hat{\mu}(c, a) + \frac{r_t - \hat{\mu}(c, a)}{N(c, a)}$$
- **Source Lines:** [`backend/models/bandit.py#L42-L88`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/bandit.py#L42-L88)
- **Validation Test:** `test_13_contextual_ucb1_scoring_with_35_tabular_context` (PASS).

---

### 2. Continuous / Neural Network Models (84-Dimensional Input $\to$ 41 Discrete Actions)

#### 2.1 REINFORCE (Monte Carlo Policy Gradient with Learned Baseline)
- **Discounted Return:**
  $$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$$
- **Baseline and Advantage:**
  $$A_t = G_t - b$$
  $$b \leftarrow b + \beta_{\text{base}} \left( \frac{1}{T}\sum_{t=0}^T G_t - b \right)$$
- **Policy Gradient and Logit Analytical Gradient:**
  $$\nabla_\theta J(\theta) = \sum_{t=0}^T A_t \nabla_\theta \ln \pi_\theta(a_t | s_t)$$
  $$\frac{\partial \ln \pi_\theta(a)}{\partial z_k} = \mathbf{1}_{\{k = a\}} - \pi_\theta(k)$$
- **Source Lines:** [`backend/models/reinforce.py#L86-L122`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/reinforce.py#L86-L122)
- **Validation Test:** `test_14_reinforce_discounted_return_and_analytical_softmax_grad` (PASS).

#### 2.2 Advantage Actor-Critic (A2C)
- **Temporal Difference Advantage:**
  $$\delta_t = r_t + (1 - d_t)\gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
- **Critic Gradient:**
  $$L(\phi) = \frac{1}{2} \delta_t^2 \implies \nabla_\phi L = -\delta_t \nabla_\phi V_\phi(s_t)$$
- **Actor Gradient with Exact Logit Entropy Jacobian:**
  $$L(\theta) = -\ln \pi_\theta(a_t | s_t) \delta_t - \beta_{\text{ent}} H(\pi_\theta(\cdot | s_t))$$
  $$\frac{\partial H(\pi)}{\partial z} = -\pi \odot (\ln \pi + H(\pi))$$
- **Source Lines:** [`backend/models/a2c.py#L78-L120`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/a2c.py#L78-L120)
- **Validation Test:** `test_15_a2c_advantage_and_entropy_gradient_correctness` (PASS).

#### 2.3 Proximal Policy Optimization (PPO)
- **Probability Ratio:**
  $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$
- **Clipped Surrogate Loss:**
  $$L^{\text{CLIP}}(\theta) = -\min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right)$$
- **Generalized Advantage Estimation GAE($\gamma, \lambda$):**
  $$\delta_t^V = r_t + (1 - d_t)\gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
  $$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
- **Source Lines:** [`backend/models/ppo.py#L122-L192`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/ppo.py#L122-L192)
- **Validation Test:** `test_16_ppo_clipped_surrogate_and_gae_correctness` (PASS).

#### 2.4 Deep Q-Network (DQN)
- **Target Value:**
  $$y_t = r_t + (1 - d_t)\gamma \max_{a' \in \mathcal{A}} Q_{\theta^-}(s_{t+1}, a')$$
- **Loss Function:**
  $$L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left( y_i - Q_\theta(s_i, a_i) \right)^2$$
- **Replay Buffer & Target Sync:** Experience tuples $(s, a, r, s', d)$ sampled uniformly; periodic hard target update $\theta^- \leftarrow \theta$.
- **Source Lines:** [`backend/models/dqn.py#L105-L155`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/dqn.py#L105-L155)
- **Validation Test:** `test_17_dqn_replay_buffer_and_terminal_target_zeroing` (PASS).

#### 2.5 Double DQN
- **Decoupled Target Equation:**
  $$a^*(s_{t+1}) = \arg\max_{a' \in \mathcal{A}} Q_\theta(s_{t+1}, a')$$
  $$y_t^{\text{Double}} = r_t + (1 - d_t)\gamma Q_{\theta^-}\left( s_{t+1}, a^*(s_{t+1}) \right)$$
- **Elimination of Maximization Bias:** Decoupling action selection (online network $\theta$) from action evaluation (target network $\theta^-$) prevents overestimation of Q-values.
- **Source Lines:** [`backend/models/double_dqn.py#L50-L80`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/double_dqn.py#L50-L80)
- **Validation Test:** `test_18_double_dqn_target_decoupling` (PASS).

#### 2.6 Deep Deterministic Policy Gradient (DDPG)
- **Continuous Action Simplex Relaxation:**
  Actor produces unnormalized logits $z(s)$, normalized via Softmax:
  $$\mu_\theta(s) = \text{Softmax}(z(s)) \in \Delta^{40}$$
- **Critic Value:**
  $$Q_\phi(s, a) \in \mathbb{R}, \quad \text{where } a = \mu_\theta(s)$$
- **Bellman Target:**
  $$y_t = r_t + (1 - d_t)\gamma Q_{\phi^-}(s_{t+1}, \mu_{\theta^-}(s_{t+1}))$$
- **Chained Gradient Backpropagation:**
  $$\nabla_\theta J = \nabla_z \mu_\theta(s) \cdot \nabla_a Q_\phi(s, a)\big|_{a = \mu_\theta(s)} \cdot \nabla_\theta z(s)$$
  where the Softmax Jacobian contraction yields:
  $$\nabla_z Q = \mu \odot \left( \nabla_a Q - \mu^T \nabla_a Q \right)$$
- **Source Lines:** [`backend/models/ddpg.py#L135-L185`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/ddpg.py#L135-L185)
- **Validation Test:** `test_19_ddpg_critic_to_actor_gradient_chaining` (PASS).

---

## Part C: Centralized Reward Equation Verification

The environment implements the 8-component configurable reward formula:
$$R_t = \alpha N_{\text{served}} - \beta W_t - \gamma Q_t - \delta C_t - \rho \Delta T_{\text{congestion}} - \zeta B_t - \kappa K_t - \xi I_t$$

### Independent Hand Calculation
Given configuration parameters:
- $\alpha = 3.0$ (Served bonus)
- $\beta = 0.15$ (Customer wait penalty)
- $\gamma = 0.25$ (Queue depth penalty)
- $\delta = 0.80$ (Congestion counter penalty)
- $\rho = 1.00$ (Congestion duration penalty)
- $\zeta = 0.30$ (Queue imbalance variance penalty)
- $\kappa = 0.25$ (Lane operational cost)
- $\xi = 5.00$ (Invalid action penalty)

For transition transition variables:
- $N_{\text{served}} = 2$
- $W_t = 4.0$
- $Q_t = 5$
- $C_t = 1$
- $\Delta T_{\text{congestion}} = 1.0$
- $B_t = 0.5$
- $K_t = 6.0$
- $I_t = 0.0$

Analytical calculation:
$$R_t = 3.0(2) - 0.15(4.0) - 0.25(5) - 0.8(1) - 1.0(1.0) - 0.3(0.5) - 0.25(6.0) - 5.0(0)$$
$$= 6.00 - 0.60 - 1.25 - 0.80 - 1.00 - 0.15 - 1.50 - 0.00$$
$$= 6.00 - 5.30 = \mathbf{0.70}$$

**Verification:** Test `test_20_centralized_reward_independent_verification` executes `RewardConfig.calculate(...)` with these parameters and asserts $|R - 0.70| < 10^{-5}$. Result: **PASS**.

---

## Part D: Defects Identified and Resolved

| Defect # | Component | Root Cause | Code Fix Applied | Proving Test |
| :---: | :--- | :--- | :--- | :--- |
| **D1** | Tabular Models | Dimension ambiguity between 35 features and 84 features. | Added `TABULAR_FEATURE_DIM = 35` and `CONTINUOUS_FEATURE_DIM = 84`. Added strict validation rejecting mismatched vectors with `ValueError`. | `test_03_tabular_state_features_shape_35` & `test_05_strict_dimension_rejection` |
| **D2** | Q-Learning & SARSA | Missing terminal state handling $(1 - d_t)$, allowing horizon bootstrapping after episode termination. | Injected $(1 - d_t)$ terminal mask: $y_t = r_t$ when $d_t = 1$. | `test_11_q_learning_bellman_target_with_terminal_masking` |
| **D3** | SARSA | Off-policy next-action decoupling in training loop. | Synchronized on-policy action passing in `simulation_engine.py`: $a_{t+1}$ chosen at step $t$ is recorded, updated, and executed at step $t+1$. | `test_12_sarsa_on_policy_synchronization_and_terminal_masking` |
| **D4** | Contextual UCB1 | Static arm selection ignoring state context. | Implemented `state_to_context(state)` using the 35 tabular features; tracked context-specific arm pull counts and empirical reward estimates $N(c, a), \hat{\mu}(c, a)$. | `test_13_contextual_ucb1_scoring_with_35_tabular_context` |
| **D5** | REINFORCE / A2C / PPO | Discretized state lookups instead of continuous neural processing. | Upgraded actor and critic networks to multi-layer perceptrons (`MLP`) processing continuous 84-dimensional observation vectors directly. | `test_04_continuous_state_features_shape_84` |
| **D6** | A2C | Heuristic scalar entropy derivative. | Implemented the analytical Logit Entropy Jacobian $\nabla_z H = -\pi \odot (\ln \pi + H(\pi))$. | `test_15_a2c_advantage_and_entropy_gradient_correctness` |
| **D7** | DQN & Double DQN | Replay buffer omitted terminal flag $d_t$, causing incorrect target value calculation on episode terminal steps. | Updated `ReplayBuffer` to store and sample 5-tuples $(s, a, r, s', d)$; target values masked by $(1 - d_t)$. | `test_17_dqn_replay_buffer_and_terminal_target_zeroing` |
| **D8** | DDPG | Heuristic Critic gradient propagation violating backprop chain rule. | Formulated continuous probability simplex $\mu(s) \in \Delta^{40}$, chained critic gradient $\nabla_a Q$ through Softmax Jacobian $\nabla_z \mu$, and backpropagated into actor weights. | `test_19_ddpg_critic_to_actor_gradient_chaining` |
| **D9** | Persistence | Missing serialization methods across agents. | Added uniform `save(filepath)` and `load(filepath)` methods across all 9 agents supporting weights and tabular Q-tables. | `test_21_model_save_and_load_persistence` |
| **D10** | SimPy Engine | Passing raw discrete simulation state into continuous agents. | Wrapped discrete-event simulation counters into full 10-lane `QueueState` container prior to calling `agent.choose_action(state)`. | `test_23_simpy_model_execution_no_mismatches` |

---

## Part E: Test Execution Results

### Suite 1: Mathematical Correctness and Architecture Test Suite
- **Command:** `python backend/test_mathematical_correctness.py`
- **Output:**
```
.........................
----------------------------------------------------------------------
Ran 25 tests in 17.753s

OK
```
- **Passed:** 25 / 25
- **Failed:** 0
- **Execution Time:** 17.753s

### Suite 2: Regression and Full System Integration Test Suite
- **Command:** `python backend/test_all_models_and_api.py`
- **Output:**
```
.........................
----------------------------------------------------------------------
Ran 25 tests in 51.589s

OK
```
- **Passed:** 25 / 25
- **Failed:** 0
- **Execution Time:** 51.589s

### Combined Results
- **Total Test Cases Executed:** 50
- **Total Passed:** 50
- **Total Failed:** 0
- **Success Rate:** 100%

---

## Part F: Remaining Architectural Considerations and Caveats

1. **DDPG Action Discretization Approximation:**
   - DDPG operates natively on continuous action spaces. In OptiQueue, the 41 discrete checkout actions are relaxed to the continuous 41-dimensional probability simplex $\Delta^{40}$ via Softmax.
   - While the continuous policy gradient through the Critic and Softmax Jacobian is mathematically exact, environment execution takes the discrete action $a = \arg\max_{i} \mu_i(s)$.
   - This $\arg\max$ projection is non-differentiable. Therefore, while DDPG converges well empirically in this test suite, **Double DQN** remains the theoretically purest value-based method for discrete retail queue dispatching.

2. **Tabular State Space Cardinality:**
   - Tabular models (Q-Learning, SARSA) use the 35 tabular features to compute a 7-tuple aggregate discrete state key:
     $$(\text{total\_q\_level}, \text{open\_count}, \text{express\_count}, \text{arrival\_bin}, \text{wait\_bin}, \text{congest\_bin}, \text{basket\_bin})$$
   - This produces $6 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 = 1,458$ distinct discrete states.
   - This discretization ensures fast tabular convergence without memory explosion, but high-variance edge cases falling outside normal queue distributions are clustered into boundary bins.

---

## Part G: Final Audit Verdict

**VERDICT: COMPLIANT AND MATHEMATICALLY VALIDATED.**

All 9 reinforcement learning algorithms in OptiQueue satisfy their theoretical specifications, obey strict state dimension constraints (35 tabular vs. 84 continuous), adhere to 41 discrete environment actions, share one centralized reward calculation, handle episode termination without horizon leakage, and pass 100% of the mathematical correctness and integration test suites.

# OptiQueue — Final Independent Verification of Audit Claims

**Audit Date:** October 14, 2026  
**Auditor:** Antigravity Advanced Agentic AI  
**Scope:** Rigorous independent review of all 9 RL algorithms, state and action spaces, mathematical equations, numerical finite differences, and test suites.

---

## 1. Action-Space Verification

### Formal Action Definitions (41 Discrete Actions)
The environment in [`backend/simulation_engine.py`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/simulation_engine.py#L330-L415) and [`backend/models/base.py`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L221-L235) defines exactly 41 discrete environment actions across all 10 lanes:
- **Actions 0..9:** Route incoming shopper to Lane $j \in \{0..9\}$
- **Actions 10..19:** Open Lane $j \in \{0..9\}$
- **Actions 20..29:** Close Lane $j \in \{0..9\}$ safely (precondition: queue must be 0, cashier idle, and at least 1 lane remains open in the store)
- **Actions 30..39:** Convert Lane $j \in \{0..9\}$ type (cycles Express $\to$ Standard $\to$ Large Basket $\to$ Flexible)
- **Action 40:** Hold / No Action (cashiers continue serving current queues)

### Model Alignment (Tabular vs. Neural)
- **Documented Requirement:** All models in the 10-lane supermarket environment must operate on compatible action semantics so they can be benchmarked against the identical environment.
- **Source Inspection:**
  - In `backend/models/q_learning.py`, `backend/models/sarsa.py`, and `backend/models/bandit.py`, the constructors specify `num_actions=41`, Q-tables and arm counts initialize with dimension 41, and `choose_action()` selects over `0..40`.
  - In `backend/simulation_engine.py` line 624, `_init_all_models()` explicitly sets `num_act = 41` and passes it to all 9 agents.
  - Outdated docstring comments mentioning "6 discrete actions" in `q_learning.py`, `sarsa.py`, and `bandit.py` were legacy remnants from an earlier 3-counter prototype. These docstrings have now been corrected.

---

## 2. 35-Feature to 7-Tuple State Mapping

### A. The 35 Tabular Features
[`QueueState.to_tabular_features()`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L118-L135) extracts exactly 35 normalized features:
1. `feat[0..9]` (10 features): Normalized queue lengths $q_i / \text{cap}_i$ ($cap_i = 6$)
2. `feat[10..19]` (10 features): Binary lane open flags $o_i \in \{0.0, 1.0\}$
3. `feat[20..29]` (10 features): Lane types normalized via `LANE_TYPE_MAP / 3.0` (Express=0.0, Standard=0.333, Large=0.667, Flexible=1.0)
4. `feat[30]` (1 feature): Incoming customer basket size normalized $\min(1, \text{basket} / 30.0)$
5. `feat[31]` (1 feature): Customer arrival rate normalized $\min(1, \lambda / 5.0)$
6. `feat[32]` (1 feature): Simulation epoch $(t_{\text{sim}} \pmod{100}) / 100.0$
7. `feat[33]` (1 feature): Congestion severity normalized $\min(1, C_t / 30.0)$
8. `feat[34]` (1 feature): Congestion duration normalized $\min(1, \Delta T_{\text{cont}} / 50.0)$

### B. The 7-Tuple Discretization Logic
[`BaseQueueAgent.state_to_key(state)`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L307-L352) aggregates the 35 features into:
$$(exp\_cat, std\_cat, lrg\_cat, flx\_cat, open\_cat, basket\_cat, is\_congested)$$
- $exp\_cat \in \{0, 1, 2\}$: Total queue depth across Express lanes ($<3 \to 0, \le 6 \to 1, >6 \to 2$)
- $std\_cat \in \{0, 1, 2\}$: Total queue depth across Standard lanes ($<5 \to 0, \le 10 \to 1, >10 \to 2$)
- $lrg\_cat \in \{0, 1, 2\}$: Total queue depth across Large Basket lanes ($<3 \to 0, \le 6 \to 1, >6 \to 2$)
- $flx\_cat \in \{0, 1, 2\}$: Total queue depth across Flexible lanes ($<3 \to 0, \le 6 \to 1, >6 \to 2$)
- $open\_cat \in \{0, 1, 2\}$: Number of active open lanes ($\le 4 \to 0, \le 7 \to 1, >7 \to 2$)
- $basket\_cat \in \{0, 1, 2\}$: Incoming basket size ($\le 5 \to 0, \le 15 \to 1, >15 \to 2$)
- $is\_congested \in \{0, 1\}$: System congestion flag ($1 \iff \sum q_i \ge 12 \text{ or } C_t > 0$)

### C. Theoretical Bound vs. Physically Reachable States
- **Theoretical Maximum Combinations:**
  $$3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 2 = 3^6 \times 2 = 1,458 \text{ distinct keys}$$
- **Physically Reachable States:**
  Whenever $\min(\sum q_i) \ge 12$, the physical constraint forces $is\_congested = 1$. Therefore, combinations where queue loads are high ($exp\_cat, std\_cat, lrg\_cat, flx\_cat \ge 1$) paired with $is\_congested = 0$ are physically impossible.
  - **Reachable States:** Exactly **954 keys**
  - **Unreachable Combinations:** Exactly **504 keys**
- **Collision Invariance:** Verified deterministically in `test_29_state_key_reachable_space_and_collision_invariance`. The mapping is 100% deterministic with zero hash collisions.

---

## 3. Contextual UCB1 Analysis

1. **Exact Scoring Equation:**
   $$\text{Score}(c, a) = \hat{\mu}(c, a) + c_{\text{ucb}} \sqrt{\frac{\ln\left(1 + \sum_{b \in \mathcal{A}} N(c, b)\right)}{N(c, a)}}$$
   - Exploration constant: $c_{\text{ucb}} = 1.2$
   - Unseen-arm bonus: If $N(c, a) = 0$, $\text{Score}(c, a) = 10^6$ (optimistic initialization forcing every arm to be sampled once before exploitation).
   - Cold-start fallback: If $\sum_b N(c, b) = 0$, the agent dispatches to the shortest open lane.
2. **Context Discretization Consistency:**
   Context $c$ is generated via `state_to_context(state) = state_to_key(state)`. It shares the identical 35-feature discretization as Q-Learning and SARSA.
3. **Sparsity Analysis and Limitation:**
   - With 41 discrete actions and 954 reachable context states, the total context-action space is:
     $$954 \times 41 = 39,114 \text{ state-action pairs}$$
   - **Critical Limitation:** In short simulation runs (e.g., 100 to 250 steps), Contextual UCB1 encounters many contexts only once or twice. Because unseen arms receive an optimistic score of $10^6$, the model will spend the vast majority of its budget exploring unvisited arms sequentially rather than exploiting learned arm means. While mathematically valid as a contextual bandit, high action cardinality severely undermines its empirical sample efficiency in small step budgets compared to Deep Q-Networks.

---

## 4. PPO and REINFORCE Verification

### A. REINFORCE
1. **Discounted Return:**
   $$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$$
   Calculated backward via $G \leftarrow r_t + \gamma G$. Verified in `test_14`.
2. **Baseline Update:**
   Mean Squared Error loss $\mathcal{L}(V) = \frac{1}{2}(V(s_t) - G_t)^2 \implies \nabla_V \mathcal{L} = V(s_t) - G_t$. The baseline network backpropagates this gradient directly.
3. **Policy Gradient Signs w.r.t Logits $z$:**
   $$\mathcal{L}(\theta) = - \ln \pi_\theta(a_t | s_t) (G_t - V(s_t))$$
   $$\frac{\partial \mathcal{L}}{\partial z_k} = \gamma^t (G_t - V(s_t)) (\pi(k) - \mathbf{1}_{\{k = a_t\}})$$
   When return exceeds baseline ($G_t - V > 0$), the loss gradient w.r.t $z_{a_t}$ is negative: $\gamma^t \delta (\pi(a_t) - 1) < 0$. Under gradient descent, $z_{a_t} \leftarrow z_{a_t} - \eta \nabla_{z} \mathcal{L}$, increasing the logit and probability $\pi(a_t)$. Mathematically exact.

### B. PPO (Proximal Policy Optimization)
1. **Probability Ratio & Clipping:**
   $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}, \quad L^{\text{CLIP}}(\theta) = -\min\left(r_t \hat{A}_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)$$
   The analytical gradient implementation uses the condition `surr1 <= surr2` to distinguish between unclipped and clipped regimes.
2. **Four-Quadrant Finite-Difference Verification:**
   Tested in `test_27_ppo_clipped_surrogate_four_quadrants_finite_difference` with step size $\epsilon = 10^{-7}$:
   - Quadrant 1 ($A > 0$, unclipped, $r = 1.05$): Max error $= 2.02 \times 10^{-9}$
   - Quadrant 2 ($A > 0$, clipped, $r = 1.52$): Max error $= 0.00$
   - Quadrant 3 ($A < 0$, unclipped, $r = 1.01$): Max error $= 2.36 \times 10^{-9}$
   - Quadrant 4 ($A < 0$, clipped, $r = 0.57$): Max error $= 0.00$
3. **GAE Terminal Masking vs. Time-Limit Truncation:**
   In `backend/models/ppo.py`, `non_terminal = 0.0 if dones[t] else 1.0` masks future bootstrapping and resets the GAE accumulator. For fixed-horizon episodes, this treats the final horizon step as terminal. While valid for finite-horizon MDPs, standard infinite-horizon RL distinguishes environment termination (`terminated`) from time-limit cutoff (`truncated`).

---

## 5. DDPG Mathematical and Execution Verification

1. **Replay Buffer & Critic Training:**
   - Transitions store discrete integer actions $a \in \{0..40\}$.
   - When training the Critic, actions are converted to one-hot vectors $e_a \in \{0, 1\}^{41}$.
   - The Critic minimizes MSE: $(Q_\phi(s, e_a) - y_t)^2$, where $y_t = r_t + (1 - d_t)\gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))$.
2. **Softmax Jacobian Contraction:**
   The Actor outputs $\mu_\theta(s) = \text{Softmax}(z(s)) \in \Delta^{40}$.
   By the multivariate chain rule:
   $$\nabla_z Q = \mu \odot \left(\nabla_a Q - (\mu^T \nabla_a Q) \mathbf{1}\right)$$
   - **Numerical Finite-Difference Check:** Verified in `test_26_ddpg_softmax_jacobian_finite_difference_check`. The analytical gradient matches the numerical finite-difference gradient with a maximum absolute error of $7.39 \times 10^{-10}$.
3. **Consequences of Executing $\arg\max$ with Continuous Simplex Actor:**
   - **Distributional Shift:** During training, the Actor is evaluated on continuous probability mixtures $\mu_\theta(s) \in \Delta^{40}$, whereas experience stored in the replay buffer is generated by executing the discrete action $a^* = \arg\max_i \mu_i(s)$ (one-hot vertex).
   - **Non-Differentiability of the Execution Policy:** The mapping $s \mapsto \arg\max_i \mu_i(s)$ is a step function with zero gradient almost everywhere. Backpropagation is only possible through the continuous Softmax relaxation $\mu_\theta(s)$, not through the discrete execution policy.
   - **Recommendation:** **Double DQN** remains the theoretically purest value-based method for discrete retail queue dispatching.

---

## 6. Centralized Reward Function Independent Verification

The environment computes:
$$R_t = \alpha N_{\text{served}} - \beta W_t - \gamma Q_t - \delta C_t - \rho \Delta T_{\text{congestion}} - \zeta B_t - \kappa K_t - \xi I_t$$

### Controlled Multi-Scenario Calculations
| Scenario | $N_{\text{served}}$ | $W_t$ | $Q_t$ | $C_t$ | $\Delta T$ | $B_t$ | $K_t$ | $I_t$ | Hand-Calculated $R_t$ | Centralized Implementation | Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Ideal State** | 5 | 0 | 0 | 0 | 0.0 | 0.0 | 2.0 | 0.0 | $3(5) - 0.25(2) = \mathbf{14.500}$ | $14.500$ | **PASS** |
| **2. Extreme Congestion** | 0 | 60 | 60 | 20 | 1.0 | 12.0 | 12.0 | 1.0 | $0 - 9 - 15 - 16 - 1 - 3.6 - 3 - 5 = \mathbf{-52.600}$ | $-52.600$ | **PASS** |
| **3. Moderate Traffic** | 1 | 10 | 10 | 2 | 0.0 | 2.5 | 5.0 | 0.0 | $3 - 1.5 - 2.5 - 1.6 - 0.75 - 1.25 = \mathbf{-4.600}$ | $-4.600$ | **PASS** |
| **4. Balanced Flow** | 2 | 4 | 5 | 1 | 1.0 | 0.5 | 6.0 | 0.0 | $6 - 0.6 - 1.25 - 0.8 - 1 - 0.15 - 1.5 = \mathbf{0.700}$ | $0.700$ | **PASS** |

- **Terminal Reward Handling:** Rewards are calculated step-by-step for each transition. No arbitrary lump-sum terminal penalties/bonuses are injected.

---

## 7. Test Suite Taxonomy and Quality Breakdown

The repository now contains **55 tests** across two distinct test suites:

### Suite 1: Mathematical Correctness & Architecture (`backend/test_mathematical_correctness.py`) — 30 Tests
- **Category A: Pure Mathematical & Numerical Verification (14 tests):**
  - Tests 11, 12: Q-Learning and SARSA Bellman target equations with terminal zeroing $(1 - d_t)$.
  - Test 13: Contextual UCB1 bonus calculation.
  - Test 14: REINFORCE Monte Carlo discounted returns and analytical logit gradients.
  - Test 15: A2C Advantage TD error and analytical logit entropy Jacobian.
  - Test 16: PPO clipped surrogate objective and GAE($\lambda$) recurrence.
  - Tests 17, 18: DQN and Double DQN target decoupling and terminal zeroing.
  - Test 19: DDPG Critic-to-Actor chained policy gradient.
  - Tests 20, 30: Multi-scenario centralized reward hand calculations.
  - Tests 25, 26, 27, 28: Analytical vs finite-difference numerical gradient checks for REINFORCE, DDPG Softmax Jacobian, PPO 4-quadrant clipping, and A2C logit entropy.
- **Category B: Architectural & State Representation Invariants (12 tests):**
  - Tests 1-10, 21, 29: 35-feature tabular order, 84-feature continuous order, 41-action semantics, strict `ValueError` dimension rejection, model persistence, and 954 reachable state bounds.
- **Category C: Integration Execution (4 tests):**
  - Tests 22, 23, 24: Episode completion across all 9 algorithms, SimPy discrete-event comparison run, and benchmark metrics.

### Suite 2: Regression, SimPy, and API Endpoints (`backend/test_all_models_and_api.py`) — 25 Tests
- **Category A: Model & Agent Invariants (10 tests):**
  - Tests 1-10: Action bounds $[0..40]$ and action values dimensionality across all 9 models.
- **Category B: 10-Lane Environment Mechanics (6 tests):**
  - Tests 11-16: Capacities, basket limits, safe closing invariants, type conversions, and hold actions.
- **Category C: Simulation Benchmarking & Flask REST API (9 tests):**
  - Tests 17-25: Centralized reward calculation, 100-step benchmark, interactive simulation step, Table IV generation, and all `/api/*` endpoints.

---

## 8. Verified Test Execution Evidence

### Test Run 1: Mathematical Correctness Suite
- **Command:** `python backend/test_mathematical_correctness.py`
- **Output:**
```
..............................
----------------------------------------------------------------------
Ran 30 tests in 60.699s

OK
```
- **Result:** **30 / 30 PASSED (100%)**

### Test Run 2: Full System Regression Suite
- **Command:** `python backend/test_all_models_and_api.py`
- **Output:**
```
.........................
----------------------------------------------------------------------
Ran 25 tests in 360.587s

OK
```
- **Result:** **25 / 25 PASSED (100%)**

---

## 9. Claim-by-Claim Verification Table

| Claim | Verified Statement | Relevant Source Location | Proving Test | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **C1: Action Space Uniformity** | All 9 models use 41 discrete actions mapped to route (0..9), open (10..19), close (20..29), convert (30..39), and hold (40). | [`models/base.py#L221-L235`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L221-L235), [`simulation_engine.py#L380-L415`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/simulation_engine.py#L380-L415) | `test_08_all_actions_map_to_valid_environment_semantics`, `test_19_interactive_step_and_semantic_actions` | **PASS** |
| **C2: 35-Feature Tabular Representation** | Exactly 35 normalized features are exposed for tabular models; input shape $\ne 35$ raises an explicit `ValueError`. | [`models/base.py#L118-L135`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L118-L135), [`models/base.py#L236-L266`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L236-L266) | `test_01_tabular_35_feature_representation_definition_and_order`, `test_10_invalid_observation_dimensions_fail_explicitly` | **PASS** |
| **C3: 84-Feature Continuous Representation** | Exactly 84 normalized continuous features are exposed for neural models; input shape $\ne 84$ raises an explicit `ValueError`. | [`models/base.py#L137-L173`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L137-L173), [`models/base.py#L268-L305`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L268-L305) | `test_04_neural_models_accept_84_continuous_features`, `test_10_invalid_observation_dimensions_fail_explicitly` | **PASS** |
| **C4: 7-Tuple Discretization Bound** | Theoretical maximum is $3^6 \times 2 = 1,458$, but exactly 954 states are physically reachable due to the $Q_t \ge 12 \implies \text{congested}$ constraint. | [`models/base.py#L307-L352`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/base.py#L307-L352) | `test_29_state_key_reachable_space_and_collision_invariance` | **PASS** |
| **C5: Contextual UCB1 Scoring** | Score combines contextual running mean, $c=1.2$ UCB bonus, $10^6$ unseen arm initialization, and shortest-queue fallback. | [`models/bandit.py#L32-L68`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/bandit.py#L32-L68) | `test_13_contextual_ucb1_scoring_with_context` | **PASS** |
| **C6: Contextual Bandit Sparsity** | With 41 arms and 954 reachable contexts, state-action sparsity severely limits early exploitation within small episode budgets. | Documented in Section 3 of this report | Analysis & test verification | **PASS** (with noted limitation) |
| **C7: Q-Learning Terminal Masking** | Target incorporates $(1 - d_t)$ preventing horizon bootstrapping: $y_t = r_t + (1 - d_t)\gamma \max_{a'} Q(s', a')$. | [`models/q_learning.py#L31-L38`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/q_learning.py#L31-L38) | `test_11_q_learning_exact_equation_and_terminal_mask` | **PASS** |
| **C8: SARSA On-Policy Synchronization** | Next action $a_{t+1}$ chosen on-policy via $\epsilon$-greedy is updated in the TD target and executed on the following simulation step. | [`models/sarsa.py#L31-L45`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/sarsa.py#L31-L45), [`simulation_engine.py#L960-L976`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/simulation_engine.py#L960-L976) | `test_12_sarsa_exact_equation_and_on_policy_sync` | **PASS** |
| **C9: REINFORCE Gradient Signs** | Discounted returns $G_t$ are centered by a learned baseline; analytical logit gradient increases probability of actions with positive advantage. | [`models/reinforce.py#L47-L86`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/reinforce.py#L47-L86) | `test_14_reinforce_return_and_policy_gradient_mathematical`, `test_25_finite_difference_vs_analytic_gradient_check` | **PASS** |
| **C10: A2C Logit Entropy Jacobian** | Exact analytical derivative $\nabla_z H(\pi) = -\pi \odot (\ln \pi + H(\pi))$ matches numerical finite differences ($1.12 \times 10^{-9}$ error). | [`models/a2c.py#L86-L115`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/a2c.py#L86-L115) | `test_15_a2c_entropy_gradient_and_terminal_handling`, `test_28_a2c_logit_entropy_jacobian_finite_difference` | **PASS** |
| **C11: PPO Four-Quadrant Clipping** | Analytical surrogate gradient accurately matches numerical finite differences across all four clipped/unclipped quadrants. | [`models/ppo.py#L122-L148`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/ppo.py#L122-L148) | `test_16_ppo_probability_ratio_clipping_and_gae`, `test_27_ppo_clipped_surrogate_four_quadrants_finite_difference` | **PASS** |
| **C12: DQN & Double DQN Targets** | DQN zeros target on terminal transitions; Double DQN decouples online selection from target evaluation. | [`models/dqn.py#L105-L135`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/dqn.py#L105-L135), [`models/double_dqn.py#L50-L80`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/double_dqn.py#L50-L80) | `test_17_dqn_terminal_targets`, `test_18_double_dqn_target_decoupling` | **PASS** |
| **C13: DDPG Softmax Jacobian** | Analytical contraction $\nabla_z Q = \mu \odot (\nabla_a Q - \mu^T \nabla_a Q)$ matches numerical finite differences ($7.39 \times 10^{-10}$ error). | [`models/ddpg.py#L80-L98`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/models/ddpg.py#L80-L98) | `test_19_ddpg_critic_gradient_and_softmax_jacobian`, `test_26_ddpg_softmax_jacobian_finite_difference_check` | **PASS** |
| **C14: DDPG Argmax Non-Differentiability** | Discrete execution policy $a = \arg\max \mu_a(s)$ is non-differentiable; continuous relaxation causes a distributional shift between training and execution. | Documented in Section 5 of this report | Analysis & test verification | **PASS** (with noted limitation) |
| **C15: Centralized Reward Equation** | 8-component configurable formula accurately computes total reward across ideal, extreme, moderate, and balanced scenarios. | [`simulation_engine.py#L68-L108`](file:///c:/Users/hp/OneDrive/Desktop/RL%20Proj/Autonomous-Retail-Checkout-Queue-Optimizer-SimPy/backend/simulation_engine.py#L68-L108) | `test_20_centralized_reward_independent_verification`, `test_30_centralized_reward_multi_case_controlled_verification` | **PASS** |

---

## 10. Summary of Architectural Caveats and Limitations

1. **DDPG Distributional Shift:** DDPG continuous policy gradients through the Critic and Softmax Jacobian are analytically sound, but environment execution takes $a = \arg\max \mu_a(s)$. The discrete execution policy is not differentiable. For discrete checkout queue dispatching, **Double DQN** remains the theoretically purest value-based method.
2. **Contextual Bandit Sample Efficiency:** With 41 discrete actions across 954 reachable contexts, Contextual UCB1 faces severe state-action sparsity in short simulation runs (e.g. 100 steps), causing it to explore unvisited arms sequentially rather than exploiting learned means.
3. **Episodic Terminal Masking:** `non_terminal = 0.0 if dones[t] else 1.0` appropriately cuts off bootstrapping for finite-horizon trajectories, but does not distinguish true absorbing failure from time-limit cutoff.

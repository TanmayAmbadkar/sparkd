# VLL-HPS: Variational Latent-Linear Horizon-Predictive Shielding
**A Framework for High-Frequency Safe Reinforcement Learning**

---

## Abstract

Safe Reinforcement Learning (RL) in high-dimensional, contact-rich environments (e.g., humanoid robotics) faces a fundamental dilemma: **Expressivity vs. Tractability**. Deep neural networks can model complex dynamics but result in non-convex safety constraints that are computationally expensive to verify. Conversely, classical robust control offers fast verification but fails to model high-dimensional sensor data or complex contacts.

We propose **Variational Latent-Linear Horizon-Predictive Shielding (VLL-HPS)**, a framework that unifies Deep Learning with Spectral Operator Theory to achieve robust safety at microsecond latency. VLL-HPS learns a complex-valued latent space where non-linear dynamics are mapped to a global **Spectral (Diagonal) Linear** evolution. Simultaneously, it learns a **Latent Control Barrier Value Function (CBVF)** that distills a multi-step safety horizon into a single linear constraint via supervised regression on offline data. By combining a Variational Autoencoder (VAE) architecture with analytic uncertainty propagation, VLL-HPS provides an adaptive, probabilistic safety shield that solves a single-step Quadratic Program (QP) to guarantee multi-step safety. We demonstrate that this architecture overcomes the "high relative degree" problem inherent in robotic safety without the computational cost of online Model Predictive Control (MPC).

---

## 1. Introduction

Deploying Reinforcement Learning agents in the physical world requires strict safety guarantees. A bipedal robot cannot learn to walk if it destroys itself during the exploration phase. While "Soft" Safe RL methods (e.g., Lagrangian relaxation) optimize for asymptotic safety, they do not prevent catastrophic failures during training. "Hard" methods, specifically **Model-Predictive Shielding (MPS)**, interpose a safety filter between the agent and the environment to override unsafe actions.

However, existing shielding approaches struggle with the curse of dimensionality.
1.  **Explicit MPC Shields** require solving non-linear optimization problems over a horizon $H$ at every control step, introducing unacceptable latency ($10\text{-}50$ms) for high-frequency motor control.
2.  **Neural Control Barrier Functions (CBFs)** learn a safety boundary, but typically rely on iterative gradient-based solvers to find safe actions because the safety constraint is non-linear with respect to the control input.

**VLL-HPS** addresses these limitations by shifting the computational burden from **Inference Time** to **Training Time**.

We introduce three key contributions:
1.  **Spectral Variational Dynamics:** We model dynamics not as a dense matrix, but as a **Diagonal Complex Operator** within a variational framework. This reduces uncertainty propagation complexity from $O(d^2)$ to $O(d)$.
2.  **Horizon Distillation:** Instead of checking safety constraints for $H$ steps at runtime, we train a **Latent Linear Value Function** to regress the "worst-case safety" of the future horizon. This effectively reduces a multi-step MPC problem into a single-step constraint.
3.  **Analytic Projection:** By enforcing linearity in the latent space, the runtime shielding mechanism collapses into a closed-form analytic projection, allowing safety verification in microseconds ($< 10 \mu s$).

---

## 2. Related Work

**Safe Reinforcement Learning:** Constrained MDP approaches like CPO and PPO-Lagrangian treat safety as a long-term cost constraint. While effective asymptotically, they do not provide instance-wise guarantees during training.

**Model Predictive Shielding:** Methods like classical MPC or Recovery RL utilize a forward model to simulate trajectories and check for safety. These methods suffer from high computational cost and often require ground-truth state access, limiting their use in pixel-based or high-dimensional tasks.

**Koopman Operator Theory:** Koopman theory asserts that non-linear dynamics can be represented linearly in an infinite-dimensional Hilbert space. Recent Deep Koopman approaches approximate this using finite-dimensional autoencoders. However, most focus on reconstruction or control, not explicit safety. VLL-HPS extends this by integrating **Spectral Diagonalization** (Lusch et al.) with **Variational Inference** to robustify the model against aleatoric uncertainty (contacts/noise).

**Neural CBFs:** Neural CBFs approximate safe sets using deep networks. Existing works (e.g., Dawson et al.) often require solving convex relaxations or using SMT solvers. Our work simplifies this by enforcing a specific **Latent-Linear** topology, ensuring the safety constraint remains convex (linear) in the control space.

---

## 3. Methodology

We consider a dynamical system with state $s \in \mathcal{S} \subseteq \mathbb{R}^n$, action $u \in \mathcal{U} \subseteq \mathbb{R}^m$, and unknown dynamics $s_{t+1} = f(s_t, u_t)$. We assume access to an offline dataset of transitions $\mathcal{D} = \{(s_t, u_t, s_{t+1}, c_t)\}$, where $c_t$ is a scalar cost indicating proximity to failure.

### 3.1 Architecture Overview

VLL-HPS is composed of four learned components trained jointly:
1.  **Probabilistic Encoder** $\phi_\theta(s)$: Maps observations to a distribution over Complex Latent States.
2.  **Spectral Dynamics** $\Lambda, B$: A globally linear, diagonal transition model.
3.  **Generative Decoder** $\psi_\theta(z)$: Reconstructs observations (regularizer).
4.  **Latent Linear CBF** $V_\omega(z)$: A linear head predicting horizon safety.

### 3.2 Component 1: The Complex Variational Encoder
To capture oscillatory physics (gaits) and sensor noise, we map the state $s_t$ to a complex-valued Gaussian distribution $z_t \in \mathbb{C}^d$.

$$ q_\phi(z_t | s_t) = \mathcal{N}(\mu_t, \text{diag}(\sigma^2_t)) $$

The encoder outputs two real vectors for the mean ($\mu_{re}, \mu_{im}$) and two for the variance ($\sigma^2_{re}, \sigma^2_{im}$). This "lifting" process unravels the non-linear manifold of the high-dimensional state into a space where dynamics are linear.

### 3.3 Component 2: Spectral Koopman Dynamics
We assume the latent state evolves according to a **Diagonal** linear operator $\Lambda \in \mathbb{C}^d$.

$$ z_{t+1} = \Lambda \odot z_t + B u_t $$

*   **$\Lambda$ (Eigenvalues):** Learnable complex vector. $|\lambda_j| \le 1$ ensures stability. The phase $\angle \lambda_j$ captures periodic frequencies.
*   **$B$ (Control Matrix):** Learnable complex matrix mapping real actions $u_t$ to the complex latent space.

**Uncertainty Propagation:** A critical advantage of the diagonal structure is efficient variance propagation. If $z_t$ has variance $\Sigma_t$ (diagonal), the predicted variance at $t+1$ is simply:
$$ \Sigma_{t+1} = |\Lambda|^2 \odot \Sigma_t + \Sigma_{process\_noise} $$
This operation is $O(d)$, unlike the $O(d^3)$ matrix multiplication required for dense dynamics.

### 3.4 Component 3: Horizon Distillation (The CBF)
Standard shielding checks $H$ constraints: $h(s_{t+1}) \ge 0, \dots, h(s_{t+H}) \ge 0$.
VLL-HPS distills this into a single value function $V(z)$.

**Data Labeling (The Oracle):**
We process the offline dataset to generate **Horizon Safety Targets** $y_t^H$. Let $S(s_t) = 1.0 - \alpha \cdot c_t$ be the immediate safety score.
$$ y_t^H = \min_{k=0 \dots H} (\gamma^k S(s_{t+k})) $$
This target represents the "bottleneck" safety value over the future horizon.

**The Linear Head:**
We learn a linear projection in the latent space to regress this target:
$$ V(z) = \mathbf{w}^\top \begin{bmatrix} \text{Re}(z) \\ \text{Im}(z) \end{bmatrix} + \beta $$

By training $V(z_t)$ to predict $y_t^H$, we effectively bake the "lookahead" into the value of the current state.

### 3.5 Training Objective
We train in two phases to ensure the latent topology stabilizes before safety boundaries are learned.

**Phase 1: Dynamics Learning (VDK)**
Autoregressive rollout for $T=5$ steps.
$$ \mathcal{L}_{dyn} = \sum_{k=1}^T \left( \|s_k - \text{Dec}(\hat{z}_k)\|^2 + \|\text{Enc}(s_k) - \hat{z}_k\|^2 \right) + \text{KL-Divergence} $$

**Phase 2: Safety Value Iteration**
Robust regression against the Horizon Oracle.
$$ \mathcal{L}_{safety} = \| V(\hat{z}_{t+1}) - y_{t+1}^H \|^2 + \lambda_{robust} \cdot \text{ReLU}(V(\hat{z}_{t+1}) - y_{t+1}^H) $$
*Note: The asymmetric ReLU term penalizes "Optimism." If the linear model thinks a state is safer than the ground truth, it incurs a heavy penalty. Pessimism is permitted.*

---

## 4. Runtime Shielding Algorithm

At inference time, we solve a constrained optimization problem to find the closest safe action $u^*$ to the RL agent's policy $u_{RL}$.

### 4.1 The Constraint
We require the future horizon to be safe with high probability ($1-\delta$).
$$ \mathbb{P}(V(z_{t+1}) \ge 0) \ge 1 - \delta $$

Using the propagated variance from the VDK, we calculate an **Adaptive Margin** $M_t$:
$$ M_t = \Phi^{-1}(1-\delta) \cdot \sqrt{ \mathbf{w}^\top \Sigma_{V, t+1} \mathbf{w} } $$

The constraint becomes a deterministic linear inequality:
$$ \mathbf{w}^\top (\Lambda z_t + B u) + \beta \ge M_t $$

### 4.2 Analytic Solution
This optimization problem is a Quadratic Program with a single linear constraint:
$$ \min_u \frac{1}{2} \| u - u_{RL} \|^2 \quad \text{s.t.} \quad (\mathbf{w}^\top B) u \ge M_t - \beta - \mathbf{w}^\top (\Lambda z_t) $$

Let $\mathbf{a} = \mathbf{w}^\top B$ and $b = \text{RHS}$.
If $\mathbf{a}^\top u_{RL} < b$, the solution is the **Orthogonal Projection**:
$$ u^* = u_{RL} + \frac{b - \mathbf{a}^\top u_{RL}}{\| \mathbf{a} \|^2} \mathbf{a} $$

This involves only vector dot products and addition. No iterative solver is required.

---

## 5. Theoretical Analysis

### 5.1 Complexity Analysis
*   **Dense MPC Shield:** Requires $O((mH)^3)$ to solve. Scaling is cubic with horizon $H$.
*   **VLL-HPS:** Requires $O(d)$ for dynamics propagation and $O(m)$ for projection. Scaling is **constant** with respect to horizon $H$ (due to distillation).

### 5.2 Addressing High Relative Degree
In systems with inertia (like a moving car), action $u$ (acceleration) does not affect position in a single step ($Relative Degree \ge 2$). Standard single-step CBFs fail here because $\frac{\partial h}{\partial u} = 0$.
VLL-HPS solves this via **Horizon Distillation**.
*   The target $y^H$ incorporates the inevitable future crash.
*   The learned weights $\mathbf{w}$ adjust such that the "danger" is visible in the latent value $V(z)$ immediately.
*   The gradient $\nabla_u V(z_{t+1}) = \mathbf{w}^\top B$ becomes non-zero, restoring control authority.

### 5.3 Robustness via Variational Inference
The term $M_t$ acts as a dynamic safety buffer. In regions where the linear model has high aleatoric uncertainty (e.g., unpredictable contacts), $\Sigma_{t+1}$ spikes, increasing $M_t$. This forces the shield to be conservative, effectively telling the agent: *"We don't know what will happen here, so back off."* This provides robustness against model mismatch without manual tuning.

---

## 6. Conclusion

VLL-HPS presents a paradigm shift for safe robotic control. By leveraging the theoretical properties of the Koopman Operator, we transform the non-convex, multi-step problem of safety verification into a convex, single-step projection in a learned latent space. The result is a shield that retains the expressivity of deep learning and the foresight of MPC, but operates at the speed of a linear reactive controller.
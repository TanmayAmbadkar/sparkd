# VLL-HPS: Variational Latent-Linear Horizon-Predictive Shielding

This repository implements **VLL-HPS**, a framework for safe reinforcement learning that combines deep variational autoencoders with **Spectral Koopman Operator Theory** to learn a latent safety shield.

## Key Features

- **Spectral Diagonal Dynamics**: Learns a low-dimensional latent space ($z$) where dynamics are diagonalized in the complex domain ($Z_{t+1} = \Lambda \odot Z_t + BU_t$), reducing complexity from $O(d^2)$ to $O(d)$.
- **Guaranteed Stability**: Explicitly constrains the real part of eigenvalues ($\text{Re}(\Lambda) \le 0$) to ensure bounded latent trajectories.
- **Probabilistic Safety**: Uses a variational approach to model uncertainty and enforce safety with high probability.
- **Horizon-Predictive Shielding**: Uses a learned Control Barrier Function (CBF) head to predict long-term safety horizons.
- **Differentiable Optimization**: Integrates a QP solver (OSQP) at runtime to minimally alter RL actions to ensure safety.

---

## Neural Architecture

The `VDK_Shield` model consists of four main components:

1.  **Probabilistic Encoder** ($q_\phi(z|x)$):
    *   Maps high-dimensional observations $x$ to a distribution over latent states $z \sim \mathcal{N}(\mu, \sigma^2)$.
    *   **TabularEncoder**: MLP for state vectors.
    *   **VisualEncoder**: CNN for pixel observations.

2.  **Generative Decoder** ($p_\theta(x|z)$):
    *   Reconstructs observations from latent states.

3.  **Spectral Koopman Dynamics**:
    *   **Diagonal Operator** $\Lambda = \mu + i\omega$ (learnable parameters).
    *   **Control Matrix** $B = B_{re} + i B_{im}$.
    *   Dynamics: $z_{t+1} = (\mu + i\omega) \odot z_t + (B_{re} + i B_{im}) u_t$.

4.  **Latent CBF Head** ($w, \beta$):
    *   A linear head on top of the latent features (concatenated Re/Im parts) to predict the safety value.
    *   $V(z) = w^T [\text{Re}(z); \text{Im}(z)] + \beta$.

---

## Training Process

Training is performed in two distinct stages.

### Phase 1: Variational Koopman Dynamics (VDK)
**Objective**: Learn a latent space that is generative, linear, and stable.

**Loss Functions**:
*   $\mathcal{L}_{rec}$: Reconstruction loss (observation space).
*   $\mathcal{L}_{lin}$: Latent linearity loss $\|\mu_{next} - \text{Re}(z_{next}^{pred})\|^2$.
*   $\mathcal{L}_{KL}$: KL divergence between posterior and predicted prior.
*   $\mathcal{L}_{spec}$: Spectral regularization on $|\Lambda|$ to discourage vanishing/exploding gradients.

### Phase 2: Robust Safety Value Iteration (CBF)
**Objective**: Learn a safety value function $V(z)$ that predicts the "Horizon Safety" of a state.

**Loss Function**:
*   **Robust Asymmetric MSE**: Penalizes over-optimism (predicting safe when unsafe) more heavily than over-conservatism.

---

## Runtime Shielding

At inference time, the VLL-HPS shield intercepts actions $u_{RL}$ from the agent:

1.  **Encode**: State $x_t \to z_t$.
2.  **Safety Margin**: Calculate adaptive margin $M_t$ by propagating uncertainty variance through the spectral dynamics $|\Lambda|^2$.
3.  **QP Optimization**: Solve the following Quadratic Program to find safe action $u^*$:
    $$u^* = \arg\min_u \|u - u_{RL}\|^2$$
    Subject to:
    $$(w^T B_{eff}) u \ge M_t - \text{Term}_{state} - \beta$$
    where the constraints are precomputed from latent parameters effectively.

---

## Usage

### Unified Entry Point
All experiments are now run through a single `main.py` script using [Hydra](https://hydra.cc/) for configuration.

### Basic Training (Unsafe / Baseline)
Train a standard agent (SAC or PPO) without shielding:

```bash
# Run SAC on Ant
python main.py agent=sac env=ant

# Run PPO on Ant
python main.py agent=ppo env=ant
```

### Training VLL-HPS (End-to-End Pipeline)
To run the full VLL-HPS data collection, training, and execution pipeline:

```bash
python main.py agent=sac env=ant train_vll=true
```

**Common Hydra Overrides:**
*   `train_vll=true`: Enables the VLL-HPS pipeline.
*   `vll_steps=20000`: Random steps for VLL pre-training.
*   `vll_epochs_dyn=100`: Epochs for training dynamics.
*   `num_steps=1000000`: Total training steps.
*   `seed=123`: Random seed.

Example:
```bash
python main.py agent=ppo env=ant train_vll=true vll_steps=10000 seed=42
```

### Rendering and Evaluation
To verify and visualize the agent:

```bash
python main.py agent=ppo env=ant render=true eval_steps=5000
```
Videos are saved to `runs_{agent}/<experiment_name>/videos/`.

## Directory Structure
*   `src/`: Core source code.
    *   `src/algorithms`: PPO and SAC implementations.
    *   `src/envs`: Environment wrappers (Humanoid, Ant, etc.).
    *   `src/policies`: Agent policies and Shield logic.
*   `conf/`: Hydra configuration files (`agent/`, `env/`, `config.yaml`).
*   `runs_{agent}/`: Unified output directory for logs, checkpoints, and videos.
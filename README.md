# RAMPS: Robust Adaptive Multi-Step Predictive Shielding

This repository contains the official implementation for the paper: **"Safer Policies via Affine Representations using Koopman Dynamics"**. RAMPS is a scalable safety shielding framework for deep reinforcement learning that operates in high-dimensional, continuous state and action spaces.

It combines a learned, globally linear dynamics model based on **Koopman operator theory** with a **multi-step robust Control Barrier Function (CBF)** to provide strong safety assurances during exploration.

---

## Key Features

- **Scalable Dynamics Learning:** Uses a Deep Koopman Operator to learn a single, globally linear model of the environment's dynamics, avoiding the curse of dimensionality that affects methods based on state-space partitioning.
- **Robust Multi-Step Shielding:** Implements a novel multi-step robust CBF that can handle systems with high relative degrees ("trap states") and provides guarantees even with an imperfect, learned model.
- **Adaptive Horizon Selection:** Employs a binary search to find the largest feasible prediction horizon at each timestep, maximizing foresight without relying on inaccurate long-term predictions.
- **High Performance:** The shielding logic is highly optimized, with state-independent components pre-computed to allow for lightweight, real-time execution. It is compatible with high-speed QP solvers like OSQP and qpOASES.
- **Modular and Flexible:** The code is structured to be modular, allowing for easy integration with different RL agents (PPO and SAC is included) and environments.

---

## Core Components

The implementation is centered around two main Python classes:

1.  `KoopmanLightning`: A [PyTorch Lightning](https://www.pytorchlightning.ai/) module for training the Deep Koopman Operator. It implements a multi-component loss function to ensure the learned model is accurate, stable, and reconstructs the original state.

2.  `CBFPolicy`: The RAMPS shield itself. This class takes a learned Koopman model and a set of safety constraints (defined as polyhedra) and performs the real-time safety verification.
    - `update_model()`: The pre-computation method that builds the QP structure.
    - `solve()`: The lightweight, real-time method that finds a safe action.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Gymnasium
- OSQP (or qpOASES)
- NumPy, SciPy

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/sparkd.git](https://github.com/your-username/sparkd.git)
    cd sparkd
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Training an Agent

The main training script is `main_ppo.py`. You can run it with various command-line arguments to configure the environment, hyperparameters, and shielding settings.

To train a PPO agent on the `SafeCheetah` environment with the RAMPS shield enabled:

```bash
python main_ppo.py \
    --env_name cheetah \
    --seed 42 \
    --num_steps 1000000 \
    --lr 3e-4 \
    --horizon 5 \
    --red_dim 34 \
    --cuda
```

To run an ablation study without the safety shield:
```bash
python main_ppo.py --env_name cheetah --no_safety --num_steps 1000000 --lr 3e-4 --cuda
```

---

## How It Works

RAMPS operates in an iterative loop:

1.  **Collect Data:** An RL agent (e.g., PPO) interacts with the environment to collect a dataset of state-action-next_state transitions.
2.  **Learn Dynamics:** The `KoopmanLightning` model is trained on this data to learn the `A`, `B`, and `c` matrices of a global linear system and a worst-case error bound `epsilon`.
3.  **Update Shield:** The `CBFPolicy` shield is initialized or updated with the newly learned dynamics. It performs an expensive, one-time pre-computation of the QP constraints.
4.  **Safe Exploration:** The RL agent continues to explore, but every action it proposes is first sent to the `CBFPolicy.solve()` method. The shield efficiently solves a multi-step QP to find the closest safe action, which is then executed in the environment.
5.  **Repeat:** The new, safe data is added to the dataset, and the process repeats, creating a virtuous cycle where a more accurate model leads to a less conservative and higher-performing shield.

---
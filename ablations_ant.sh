#!/bin/bash

# ======================================================================================
# Parallel Hyperparameter Sweep Script for Safe PPO
#
# Description:
# This script launches multiple training runs in parallel to efficiently sweep
# through hyperparameters. It manages the number of concurrent jobs to avoid
# overloading the system.
#
# Instructions:
# 1. Save this file as `run_sweep_parallel.sh`.
# 2. Make it executable: `chmod +x run_sweep_parallel.sh`
# 3. Run the script: `./run_sweep_parallel.sh`
#
# ======================================================================================

# --- Configuration ---
# Set the maximum number of jobs to run in parallel.
# A good starting point is the number of CPU cores on your machine.
MAX_JOBS=4

# --- Activate Conda Environment ---
echo "Activating conda environment: spiceenv"
eval "$(conda shell.bash hook)" || true
conda activate spiceenv

if [[ $? -ne 0 ]] || [[ -z "$CONDA_PREFIX" ]] || [[ "$(basename "$CONDA_PREFIX")" != "spiceenv" ]]; then
    echo "Failed to activate conda environment 'spiceenv'."
    echo "Please make sure conda is initialized and the environment exists."
    exit 1
fi
echo "Conda environment activated successfully: $CONDA_PREFIX"


# --- Setup ---
# Create a directory to store logs for each run
LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"
echo "Log files will be saved in the '$LOG_DIR' directory."

# Using an array for the command is more robust than a single string
PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"

# --- WARNING ---
# The --render flag is included below as you requested.
# Running multiple GUI applications in parallel can be very slow, unstable,
# and may open an overwhelming number of windows.
# For efficient sweeps, it is HIGHLY recommended to REMOVE the "--render" flag.
BASE_ARGS=("$PYTHON_EXECUTABLE" main_ppo.py --env_name ant --num_steps 1000000)
SEEDS=(0 1 2)

# --- Function to manage and launch jobs ---
run_job() {
    # This loop is a robust way to manage concurrent jobs. It will pause
    # the script if the maximum number of jobs is already running.
    while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
        # Wait for 1 second before checking again
        sleep 1
    done

    # Execute the command in the background
    # Redirect stdout and stderr to a dedicated log file
    echo "-----------------------------------------"
    echo "LOGGING TO: $2"
    echo "EXECUTING: $1"
    eval "$1" > "$2" 2>&1 &
    sleep 0.2 # Small delay to allow job to register
}


# --- Sweep 1: Varying CBF Gamma (cbf_gamma) ---
echo "========================================="
echo "Starting Sweep 1: CBF Gamma (cbf_gamma)"
echo "========================================="
CBF_GAMMA_VALUES=(0.05 0.4 0.7 0.99)
for val in "${CBF_GAMMA_VALUES[@]}"; do
  for seed_val in "${SEEDS[@]}"; do
    FIXED_ARGS="--horizon 5 --percentile 99 --red_dim 100"
    LOG_FILE="$LOG_DIR/ant_gamma_${val}_seed_${seed_val}.log"
    CMD="${BASE_ARGS[*]} --cbf_gamma $val --seed $seed_val $FIXED_ARGS"
    run_job "$CMD" "$LOG_FILE"
  done
done


# --- Sweep 2: Varying Prediction Horizon (horizon) ---
echo "========================================="
echo "Starting Sweep 2: Prediction Horizon (horizon)"
echo "========================================="
HORIZON_VALUES=(2 5 10 20)
for val in "${HORIZON_VALUES[@]}"; do
  for seed_val in "${SEEDS[@]}"; do
    FIXED_ARGS="--cbf_gamma 0.4 --percentile 99 --red_dim 100"
    LOG_FILE="$LOG_DIR/ant_horizon_${val}_seed_${seed_val}.log"
    CMD="${BASE_ARGS[*]} --horizon $val --seed $seed_val $FIXED_ARGS"
    run_job "$CMD" "$LOG_FILE"
  done
done


# --- Sweep 3: Varying Percentile ---
echo "========================================="
echo "Starting Sweep 3: Percentile"
echo "========================================="
PERCENTILE_VALUES=(25 50 75 99)
for val in "${PERCENTILE_VALUES[@]}"; do
  for seed_val in "${SEEDS[@]}"; do
    FIXED_ARGS="--cbf_gamma 0.4 --horizon 5 --red_dim 100"
    LOG_FILE="$LOG_DIR/ant_percentile_${val}_seed_${seed_val}.log"
    CMD="${BASE_ARGS[*]} --percentile $val --seed $seed_val $FIXED_ARGS"
    run_job "$CMD" "$LOG_FILE"
  done
done

# --- Final Wait ---
# Wait for all remaining background jobs to complete before exiting the script
echo "========================================="
echo "All jobs launched. Waiting for remaining runs to complete..."
wait
echo "All hyperparameter sweeps are complete."
echo "========================================="


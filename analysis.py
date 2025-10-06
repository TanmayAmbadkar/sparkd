import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_deviation_comparison(file_linear, file_koopman, smoothing_window=10):
    """
    Reads two CSV files from experiment runs and plots a smoothed comparison
    of the 'Value' column (Average Deviation).

    Args:
        file_linear (str): Filepath for the linear model (D0) run.
        file_koopman (str): Filepath for the Koopman model (D34) run.
        smoothing_window (int): The window size for the rolling average smoothing.
    """
    try:
        # --- Load Data ---
        print(f"Reading data from {file_linear}...")
        df_linear = pd.read_csv(file_linear)
        
        print(f"Reading data from {file_koopman}...")
        df_koopman = pd.read_csv(file_koopman)

        # --- Smooth Data ---
        # Apply a rolling average to smooth out noise in the data
        df_linear['Value_Smoothed'] = df_linear['Value'].rolling(window=smoothing_window, min_periods=1).mean()
        df_koopman['Value_Smoothed'] = df_koopman['Value'].rolling(window=smoothing_window, min_periods=1).mean()

        # --- Plotting ---
        # plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot Linear Model (D0)
        ax.plot(df_linear['Step'], df_linear['Value_Smoothed'], label='RAMPS + L', color='tab:blue', linewidth=2)
        
        # Plot Koopman Model (D34)
        ax.plot(df_koopman['Step'], df_koopman['Value_Smoothed'], label='RAMPS + K', color='tab:orange', linewidth=2)
        
        # --- Formatting ---
        # ax.set_title('Shield Intervention Analysis: Koopman vs. Linear Model', fontsize=16, weight='bold')
        ax.set_xlabel('Training Timestep', fontsize=12)
        ax.set_ylabel('Average Action Deviation', fontsize=12)
        
        ax.legend(fontsize=11)
        # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        plt.tight_layout()
        
        # Save the figure
        output_filename = 'shield_deviation_comparison.png'
        plt.savefig(output_filename, dpi=300)
        print(f"\nPlot saved successfully as '{output_filename}'")
        
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find the file - {e.filename}")
        print("Please ensure the CSV files are in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # Define the file paths for your two runs
    # Make sure these filenames match the files in your directory
    linear_model_file = 'run-2025-09-25_01-47-16_PPO_cheetah_H5_D0_G0.4_S0_P99_safe-tag-agent_avg_deviation.csv'
    koopman_model_file = 'run-2025-09-25_01-50-59_PPO_cheetah_H5_D34_G0.4_S0_P99_safe-tag-agent_avg_deviation.csv'
    
    # Generate the plot
    plot_deviation_comparison(linear_model_file, koopman_model_file, smoothing_window=20)

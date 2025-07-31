import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_individual_timeseries(timeseries_df, save_dir="results/individual_plots/"):
    """
    Plot and save a timeseries for each run individually.
    
    Args:
        timeseries_df: DataFrame with timeseries data.
        save_dir: Directory to save individual plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for run_id in timeseries_df['run_id'].unique():
        run_data = timeseries_df[timeseries_df['run_id'] == run_id]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Count-based plot
        ax1.plot(run_data['frame'], run_data['count_fraction'] * 100, 'b-', label=f'Run {run_id}')
        ax1.set_ylabel('% Native Contacts (Count)')
        ax1.set_title(f'Native Contacts Over Time - Run {run_id}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Energy-based plot
        ax2.plot(run_data['frame'], run_data['energy_fraction'] * 100, 'r-', label=f'Run {run_id}')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('% Native Contacts (Energy)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        save_path = save_dir / f"run_{run_id:03d}_timeseries.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Close figure to save memory
        
    print(f"Saved individual plots to {save_dir}")

def plot_all_timeseries_subplots(timeseries_df, save_path="results/all_runs_subplots.png"):
    """
    Plot all timeseries in a grid of subplots.
    
    Args:
        timeseries_df: DataFrame with timeseries data.
        save_path: Path to save the combined figure.
    """
    run_ids = sorted(timeseries_df['run_id'].unique())
    n_runs = len(run_ids)
    
    # Create a grid of subplots
    # We'll have 2 rows of plots per run (count and energy)
    fig, axes = plt.subplots(n_runs * 2, 1, figsize=(12, 5 * n_runs), sharex=True)
    
    for i, run_id in enumerate(run_ids):
        run_data = timeseries_df[timeseries_df['run_id'] == run_id]
        
        # Count-based plot
        ax1 = axes[i*2]
        ax1.plot(run_data['frame'], run_data['count_fraction'] * 100, 'b-', label=f'Run {run_id}')
        ax1.set_ylabel(f'Run {run_id}\\n% Count')
        ax1.grid(True, alpha=0.3)
        
        # Energy-based plot
        ax2 = axes[i*2 + 1]
        ax2.plot(run_data['frame'], run_data['energy_fraction'] * 100, 'r-', label=f'Run {run_id}')
        ax2.set_ylabel(f'Run {run_id}\\n% Energy')
        ax2.grid(True, alpha=0.3)

    axes[0].set_title('All Native Contacts Timeseries', fontsize=16)
    axes[-1].set_xlabel('Frame')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved all-run subplot to {save_path}")
    plt.show()

def plot_overlaid_timeseries(timeseries_df, save_path="results/overlaid_timeseries.png"):
    """
    Plot overlaid timeseries for all runs.
    
    Args:
        timeseries_df: DataFrame with timeseries data.
        save_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Overlaid Count-based native contacts
    for run_id in timeseries_df['run_id'].unique():
        run_data = timeseries_df[timeseries_df['run_id'] == run_id]
        ax1.plot(run_data['frame'], run_data['count_fraction'] * 100, alpha=0.5, label=f'Run {run_id}')
    
    ax1.set_ylabel('% Native Contacts (Count)')
    ax1.set_title('Overlaid Native Contacts Timeseries')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Overlaid Energy-based native contacts
    for run_id in timeseries_df['run_id'].unique():
        run_data = timeseries_df[timeseries_df['run_id'] == run_id]
        ax2.plot(run_data['frame'], run_data['energy_fraction'] * 100, alpha=0.5)
        
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('% Native Contacts (Energy)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved overlaid plot to {save_path}")
    plt.show()

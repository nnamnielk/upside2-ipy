#!/usr/bin/env python3
"""
Script to handle missing .pkl files by creating a hybrid loading approach.
This script modifies the loading functions to work with missing metadata files.
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

def load_reference_metadata(results_dir="results/"):
    """
    Load reference metadata from the first available .pkl file.
    This provides consistent donor/acceptor lists across all runs.
    """
    results_path = Path(results_dir)
    pkl_files = list(results_path.glob("*_hbond_results.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found to use as reference")
    
    # Use the first available .pkl file as reference
    reference_file = pkl_files[0]
    print(f"Using {reference_file} as reference metadata")
    
    with open(reference_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata

def create_synthetic_metadata(energy_maps_shape, reference_metadata):
    """
    Create synthetic metadata for runs missing .pkl files.
    
    Args:
        energy_maps_shape: tuple of (n_frames, n_residues, n_residues)
        reference_metadata: dict with reference donor/acceptor info
    
    Returns:
        dict with synthetic metadata
    """
    n_frames, n_residues, _ = energy_maps_shape
    
    synthetic_metadata = {
        'donors': reference_metadata['donors'],
        'acceptors': reference_metadata['acceptors'], 
        'n_residues': n_residues,
        'n_frames': n_frames
    }
    
    return synthetic_metadata

def load_hbond_data_to_dataframe_hybrid(file_prefix, results_dir="results/", 
                                       reference_metadata=None, show_timing=False):
    """
    Hybrid loading function that handles missing .pkl files and corrupted .npy files.
    
    Args:
        file_prefix: e.g. "Kmarx_Pab1.run.0"
        results_dir: directory containing the results files
        reference_metadata: reference metadata dict (loaded once and reused)
        show_timing: whether to print timing information
    
    Returns:
        pandas.DataFrame with hydrogen bond data
    """
    start_time = time.time()
    
    # Construct file paths
    npy_file = Path(results_dir) / f"{file_prefix}_hbond_energy_maps.npy"
    pkl_file = Path(results_dir) / f"{file_prefix}_hbond_results.pkl"
    
    # Check if .npy file exists (required)
    if not npy_file.exists():
        raise FileNotFoundError(f"Energy maps file not found: {npy_file}")
    
    # Check if .npy file is corrupted (too small)
    file_size = npy_file.stat().st_size
    if file_size < 1000000:  # Less than 1MB indicates corruption
        raise ValueError(f"Energy maps file appears corrupted (size: {file_size} bytes): {npy_file}")
    
    # Load energy maps
    load_start = time.time()
    try:
        energy_maps = np.load(npy_file)
    except (ValueError, OSError) as e:
        raise ValueError(f"Failed to load energy maps from {npy_file}: {e}")
    
    # Try to load .pkl file, use synthetic metadata if missing
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            metadata = pickle.load(f)
        metadata_source = "original"
    else:
        if reference_metadata is None:
            reference_metadata = load_reference_metadata(results_dir)
        metadata = create_synthetic_metadata(energy_maps.shape, reference_metadata)
        metadata_source = "synthetic"
        print(f"Warning: Using synthetic metadata for {file_prefix}")
    
    load_time = time.time() - load_start
    
    n_frames, n_residues, _ = energy_maps.shape
    
    # Extract run_id from file_prefix
    run_id = file_prefix.split('.')[-1] if '.' in file_prefix else file_prefix
    
    # Convert to long format DataFrame
    process_start = time.time()
    data_rows = []
    
    for frame in range(n_frames):
        for i in range(n_residues):
            for j in range(i+1, n_residues):  # Only upper triangle
                energy = energy_maps[frame, i, j]
                if energy > 1e-6:  # Only meaningful interactions
                    data_rows.append({
                        'frame': frame,
                        'residue_i': i,
                        'residue_j': j,
                        'hbond_energy': energy,
                        'run_id': run_id
                    })
    
    df = pd.DataFrame(data_rows)
    process_time = time.time() - process_start
    
    # Add metadata as attributes
    df.attrs = {
        'n_donors': len(metadata['donors']),
        'n_acceptors': len(metadata['acceptors']),
        'n_residues': metadata['n_residues'],
        'n_frames': metadata['n_frames'],
        'file_prefix': file_prefix,
        'metadata_source': metadata_source
    }
    
    total_time = time.time() - start_time
    
    if show_timing:
        print(f"\nTiming for {file_prefix} ({metadata_source} metadata):")
        print(f"  File loading: {load_time:.3f}s")
        print(f"  Data processing: {process_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Interactions found: {len(df)}")
    
    return df

class NativeContactsAnalyzerHybrid:
    """
    Enhanced analyzer that handles missing .pkl files gracefully.
    """
    
    def __init__(self, results_dir="results/", csv_dir="results/csv_data/"):
        self.results_dir = Path(results_dir)
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_data = []
        self.native_contacts = None
        self.native_contact_energies = None
        self.reference_metadata = None
    
    def _ensure_reference_metadata(self):
        """Load reference metadata if not already loaded."""
        if self.reference_metadata is None:
            self.reference_metadata = load_reference_metadata(self.results_dir)
    
    def load_single_run(self, run_number: int, file_pattern="Kmarx_Pab1.run", 
                       save_csv=True, show_timing=False):
        """Load a single run with hybrid metadata handling."""
        self._ensure_reference_metadata()
        file_prefix = f"{file_pattern}.{run_number}"
        
        try:
            df = load_hbond_data_to_dataframe_hybrid(
                file_prefix, self.results_dir, self.reference_metadata, show_timing
            )
            
            # Ensure runs_data list is large enough
            while len(self.runs_data) <= run_number:
                self.runs_data.append(None)
            
            self.runs_data[run_number] = df
            
            # Save as CSV if requested
            if save_csv:
                csv_path = self.csv_dir / f"run_{run_number:03d}.csv"
                self.save_dataframe_to_csv(df, csv_path)
            
            metadata_source = df.attrs.get('metadata_source', 'unknown')
            print(f"Successfully loaded run {run_number} with {len(df)} interactions ({metadata_source} metadata)")
            return df
            
        except FileNotFoundError as e:
            print(f"Could not load run {run_number}: {e}")
            return None
    
    def load_all_runs(self, file_pattern="Kmarx_Pab1.run", max_runs=None, 
                     save_csv=True, show_timing=False):
        """Load all available runs with hybrid handling."""
        self._ensure_reference_metadata()
        
        # Find all available runs
        npy_files = list(self.results_dir.glob(f"{file_pattern}*_hbond_energy_maps.npy"))
        
        run_numbers = []
        for f in npy_files:
            prefix = f.stem.replace("_hbond_energy_maps", "")
            try:
                run_num = int(prefix.split('.')[-1])
                run_numbers.append(run_num)
            except ValueError:
                continue
        
        run_numbers.sort()
        if max_runs:
            run_numbers = run_numbers[:max_runs]
        
        print(f"Found {len(run_numbers)} runs to load: {run_numbers}")
        
        successful_loads = 0
        synthetic_count = 0
        
        for run_num in run_numbers:
            df = self.load_single_run(run_num, file_pattern, save_csv, show_timing)
            if df is not None:
                successful_loads += 1
                if df.attrs.get('metadata_source') == 'synthetic':
                    synthetic_count += 1
        
        print(f"Successfully loaded {successful_loads} out of {len(run_numbers)} runs")
        print(f"Used synthetic metadata for {synthetic_count} runs")
        return successful_loads
    
    def save_dataframe_to_csv(self, df, filepath, include_metadata=True):
        """Save DataFrame to CSV with metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        
        if include_metadata and hasattr(df, 'attrs') and df.attrs:
            metadata_file = filepath.with_suffix('.meta.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(df.attrs, f, indent=2)
        
        print(f"Saved DataFrame to {filepath}")
    
    def identify_native_contacts(self, run_number=0, energy_threshold=1e-6):
        """Identify native contacts from frame 0."""
        if run_number >= len(self.runs_data) or self.runs_data[run_number] is None:
            raise ValueError(f"Run {run_number} not loaded")
        
        df = self.runs_data[run_number]
        frame_0 = df[df['frame'] == 0].copy()
        native_pairs = frame_0[frame_0['hbond_energy'] > energy_threshold]
        
        self.native_contacts = set()
        for _, row in native_pairs.iterrows():
            pair = (min(row['residue_i'], row['residue_j']), 
                   max(row['residue_i'], row['residue_j']))
            self.native_contacts.add(pair)
        
        self.native_contact_energies = {}
        for _, row in native_pairs.iterrows():
            pair = (min(row['residue_i'], row['residue_j']), 
                   max(row['residue_i'], row['residue_j']))
            self.native_contact_energies[pair] = row['hbond_energy']
        
        print(f"Identified {len(self.native_contacts)} native contacts from run {run_number}, frame 0")
        return self.native_contacts, self.native_contact_energies
    
    def calculate_native_contacts_timeseries(self, run_number: int, energy_threshold=1e-6):
        """Calculate native contacts preservation over time."""
        if self.native_contacts is None:
            raise ValueError("Native contacts not identified. Run identify_native_contacts() first.")
        
        if run_number >= len(self.runs_data) or self.runs_data[run_number] is None:
            raise ValueError(f"Run {run_number} not loaded")
        
        df = self.runs_data[run_number]
        frames = sorted(df['frame'].unique())
        results = []
        
        total_native_contacts = len(self.native_contacts)
        total_native_energy = sum(self.native_contact_energies.values())
        
        for frame in frames:
            frame_data = df[df['frame'] == frame]
            
            present_native_contacts = 0
            current_native_energy = 0.0
            
            for pair in self.native_contacts:
                i, j = pair
                contact_rows = frame_data[
                    ((frame_data['residue_i'] == i) & (frame_data['residue_j'] == j)) |
                    ((frame_data['residue_i'] == j) & (frame_data['residue_j'] == i))
                ]
                
                if not contact_rows.empty:
                    max_energy = contact_rows['hbond_energy'].max()
                    if max_energy > energy_threshold:
                        present_native_contacts += 1
                        current_native_energy += max_energy
            
            count_fraction = present_native_contacts / total_native_contacts if total_native_contacts > 0 else 0
            energy_fraction = current_native_energy / total_native_energy if total_native_energy > 0 else 0
            
            results.append({
                'frame': frame,
                'count_fraction': count_fraction,
                'energy_fraction': energy_fraction,
                'run_id': run_number
            })
        
        return pd.DataFrame(results)
    
    def calculate_all_native_contacts_timeseries(self, energy_threshold=1e-6):
        """Calculate native contacts preservation for all loaded runs."""
        if self.native_contacts is None:
            raise ValueError("Native contacts not identified. Run identify_native_contacts() first.")
        
        all_results = []
        for run_num, df in enumerate(self.runs_data):
            if df is not None:
                print(f"Processing run {run_num}...")
                run_results = self.calculate_native_contacts_timeseries(run_num, energy_threshold)
                all_results.append(run_results)
        
        if not all_results:
            raise ValueError("No runs loaded")
        
        return pd.concat(all_results, ignore_index=True)

def main():
    """Test the hybrid approach."""
    print("Testing hybrid approach for missing .pkl files and corrupted .npy files")
    
    # Create analyzer
    analyzer = NativeContactsAnalyzerHybrid()
    
    # Test loading a few runs
    print("\n=== Testing individual run loading ===")
    analyzer.load_single_run(0, show_timing=True)  # Has .pkl file
    analyzer.load_single_run(1, show_timing=True)  # Missing .pkl file
    analyzer.load_single_run(2, show_timing=True)  # Missing .pkl file
    
    print("\n=== Loading all runs ===")
    analyzer.load_all_runs(max_runs=5)  # Test with first 5 runs
    
    print("\n=== Identifying native contacts ===")
    analyzer.identify_native_contacts(run_number=0)
    
    print("\n=== Calculating time series ===")
    timeseries_data = analyzer.calculate_all_native_contacts_timeseries()
    print(f"Generated time series data with {len(timeseries_data)} points")
    
    print("\nHybrid approach working successfully!")

if __name__ == "__main__":
    main()

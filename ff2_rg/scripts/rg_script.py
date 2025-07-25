import os

import glob

import pandas as pd

import numpy as np

import mdtraj as md

from collections import defaultdict

import sys



# Add the current directory to path to import mdtraj_upside

sys.path.append('.')

import mdtraj_upside
def get_trajectory_files():

    """Get all trajectory files from run1 and run2 directories."""

    run1_files = glob.glob('../run1/*-out_*.h5')

    run2_files = glob.glob('../run2/*-out_*.h5')

    

    all_files = run1_files + run2_files

    print(f"Found {len(run1_files)} files in run1 and {len(run2_files)} files in run2")

    print(f"Total files: {len(all_files)}")

    

    return all_files
def extract_pdb_id(filename):

    """Extract PDB ID from filename by splitting on '-' and taking first element."""

    basename = os.path.basename(filename)

    return basename.split('-')[0]
def get_unique_pdb_files(all_files):

    """Get one representative file for each unique PDB ID."""

    pdb_to_file = {}

    

    for filepath in all_files:

        pdb_id = extract_pdb_id(filepath)

        if pdb_id not in pdb_to_file:

            pdb_to_file[pdb_id] = filepath

    

    print(f"Found {len(pdb_to_file)} unique PDB IDs:")

    for pdb_id in sorted(pdb_to_file.keys()):

        print(f"  {pdb_id}: {os.path.basename(pdb_to_file[pdb_id])}")

    

    return pdb_to_file
def analyze_trajectory(filepath):

    """Load trajectory and extract length and amino acid count."""

    try:

        print(f"Loading {os.path.basename(filepath)}...")

        

        # Load trajectory using the custom upside loader

        traj = md.load(filepath)

        

        trajectory_length = traj.n_frames

        num_amino_acids = traj.n_residues

        

        print(f"  Frames: {trajectory_length}, Residues: {num_amino_acids}")

        

        return trajectory_length, num_amino_acids

        

    except Exception as e:

        print(f"  ERROR loading {filepath}: {str(e)}")

        return None, None
# Main analysis

print("Starting trajectory analysis...")

print("=" * 50)



# Get all trajectory files

all_files = get_trajectory_files()

print()



# Get one file per unique PDB ID

pdb_to_file = get_unique_pdb_files(all_files)

print()



# Analyze each unique trajectory

results = []

print("Analyzing trajectories...")

print("-" * 30)



for pdb_id in sorted(pdb_to_file.keys()):

    filepath = pdb_to_file[pdb_id]

    trajectory_length, num_amino_acids = analyze_trajectory(filepath)

    

    if trajectory_length is not None and num_amino_acids is not None:

        results.append({

            'pdb_id': pdb_id,

            'trajectory_length': trajectory_length,

            'number_of_amino_acids': num_amino_acids

        })

    print()
# Create and display results DataFrame

df = pd.DataFrame(results)

df = df.sort_values('pdb_id').reset_index(drop=True)



print("\nFinal Results:")

print("=" * 50)

print(df.to_string(index=False))



print(f"\nSummary Statistics:")

print(f"Number of proteins analyzed: {len(df)}")

print(f"Average trajectory length: {df['trajectory_length'].mean():.0f} frames")

print(f"Average protein size: {df['number_of_amino_acids'].mean():.1f} amino acids")

print(f"Trajectory length range: {df['trajectory_length'].min()} - {df['trajectory_length'].max()} frames")

print(f"Protein size range: {df['number_of_amino_acids'].min()} - {df['number_of_amino_acids'].max()} amino acids")
# Save results to CSV

output_file = 'trajectory_analysis_results.csv'

df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
# Display the final DataFrame for easy copying

print("\nFinal DataFrame:")

df
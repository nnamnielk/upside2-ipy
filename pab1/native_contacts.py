#!/usr/bin/env python3
"""
Native Hydrogen Bond Energy Analysis Script

This script calculates a residue-residue hydrogen bond energy map for each frame
of a molecular dynamics trajectory. The energy calculation is designed to replicate
the smooth potential functions used in the upside2 force field's hbond.cpp.

Usage:
    python native_contacts.py <trajectory_file> <output_prefix> [--stride STRIDE]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
import pickle

warnings.filterwarnings('ignore')

# Add utils to path for mdtraj_upside
sys.path.append('../ff2_rg/utils')
import mdtraj_upside as mu

# --- Energy Function Parameters (approximating hbond.cpp) ---

# Radial potential (distance in nm)
RADIAL_OUTER_BARRIER = 0.35  # 3.5 Å
RADIAL_INNER_BARRIER = 0.20  # 2.0 Å
INV_RADIAL_WIDTH = 200     # Sharpness of the distance potential walls (1/nm)

# Angular potential (dot product)
ANGULAR_WALL_DP = 0.0      # Corresponds to a 90-degree angle
INV_ANGULAR_WIDTH = 50     # Sharpness of the angular potential wall


def sigmoid(x):
    """Computes the logistic sigmoid function and its derivative."""
    exp_x = np.exp(x)
    one_plus_exp_x = 1 + exp_x
    val = exp_x / one_plus_exp_x
    deriv = val * (1 - val)
    return val, deriv


def hbond_radial_potential(distance):
    """
    Calculates a smooth radial potential based on H-O distance.
    Returns a value between 0 (no interaction) and 1 (ideal).
    """
    outer_sigmoid_val, _ = sigmoid((RADIAL_OUTER_BARRIER - distance) * INV_RADIAL_WIDTH)
    inner_sigmoid_val, _ = sigmoid((distance - RADIAL_INNER_BARRIER) * INV_RADIAL_WIDTH)
    return outer_sigmoid_val * inner_sigmoid_val


def hbond_angular_potential(dot_product):
    """
    Calculates a smooth angular potential based on a dot product.
    Returns a value between 0 (disallowed angle) and 1 (allowed).
    """
    val, _ = sigmoid((dot_product - ANGULAR_WALL_DP) * INV_ANGULAR_WIDTH)
    return val


def identify_hbond_atoms(traj):
    """
    Identify hydrogen bond donors and acceptors and map them to residues.
    
    Returns:
        donors: list of (H_idx, N_idx, residue_idx) tuples
        acceptors: list of (O_idx, C_idx, residue_idx) tuples
    """
    donors = []
    acceptors = []
    topology = traj.topology
    
    # Create atom-to-residue mapping
    atom_to_res = {atom.index: atom.residue.index for atom in topology.atoms}

    # Find donors (H bonded to N)
    for bond in topology.bonds:
        a1, a2 = bond
        if a1.element.symbol == 'H' and a2.element.symbol == 'N':
            donors.append((a1.index, a2.index, atom_to_res[a1.index]))
        elif a1.element.symbol == 'N' and a2.element.symbol == 'H':
            donors.append((a2.index, a1.index, atom_to_res[a2.index]))
    
    # Find acceptors (O bonded to C)
    for bond in topology.bonds:
        a1, a2 = bond
        if a1.element.symbol == 'O' and a2.element.symbol == 'C':
            acceptors.append((a1.index, a2.index, atom_to_res[a1.index]))
        elif a1.element.symbol == 'C' and a2.element.symbol == 'O':
            acceptors.append((a2.index, a1.index, atom_to_res[a2.index]))
    
    print(f"Found {len(donors)} potential donors")
    print(f"Found {len(acceptors)} potential acceptors")
    
    return donors, acceptors


def calculate_hbond_energy_map(xyz, donors, acceptors, n_residues):
    """
    Calculate the residue-residue hydrogen bond energy map for a single frame.
    
    Returns:
        energy_map: (n_residues, n_residues) numpy array with interaction energies.
    """
    energy_map = np.zeros((n_residues, n_residues))
    
    for h_idx, n_idx, donor_res_idx in donors:
        for o_idx, c_idx, acceptor_res_idx in acceptors:
            # Exclude intra-residue and adjacent-residue interactions
            if abs(donor_res_idx - acceptor_res_idx) < 2:
                continue
                
            h_pos, n_pos = xyz[h_idx], xyz[n_idx]
            o_pos, c_pos = xyz[o_idx], xyz[c_idx]
            
            # Calculate H-O distance (in nanometers)
            ho_vec = o_pos - h_pos
            ho_dist = np.linalg.norm(ho_vec)
            
            # Quick check to skip distant pairs
            if ho_dist > RADIAL_OUTER_BARRIER + 0.1: # Add buffer
                continue

            # Calculate radial energy component
            radial_energy = hbond_radial_potential(ho_dist)
            if radial_energy < 1e-4:
                continue

            # Calculate angular energy components
            rHO = ho_vec / ho_dist
            
            hn_vec = h_pos - n_pos
            rHN = hn_vec / np.linalg.norm(hn_vec)
            
            oc_vec = o_pos - c_pos
            rOC = oc_vec / np.linalg.norm(oc_vec)
            
            dotHOC = np.dot(rHO, rOC)
            dotOHN = -np.dot(rHO, rHN)
            
            angular_energy1 = hbond_angular_potential(dotHOC)
            angular_energy2 = hbond_angular_potential(dotOHN)

            # Total energy is the product of the components
            total_energy = radial_energy * angular_energy1 * angular_energy2
            
            if total_energy > 1e-4:
                energy_map[donor_res_idx, acceptor_res_idx] += total_energy
                energy_map[acceptor_res_idx, donor_res_idx] += total_energy
    
    return energy_map


def run_hbond_analysis(traj_file, stride=1):
    """
    Run the full hydrogen bond energy analysis on a trajectory.
    """
    print(f"Loading trajectory: {traj_file}")
    traj = mu.load_upside_traj(traj_file, stride=stride)
    n_frames = traj.n_frames
    n_residues = traj.n_residues
    print(f"Loaded {n_frames} frames, {traj.n_atoms} atoms, {n_residues} residues")
    
    donors, acceptors = identify_hbond_atoms(traj)
    
    energy_maps = np.zeros((n_frames, n_residues, n_residues))
    
    print("Analyzing trajectory frames...")
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"Processing frame {i}/{n_frames}")
        energy_maps[i] = calculate_hbond_energy_map(traj.xyz[i], donors, acceptors, n_residues)
    
    print("Analysis complete!")
    
    return {
        'energy_maps': energy_maps,
        'donors': donors,
        'acceptors': acceptors,
        'n_residues': n_residues,
        'n_frames': n_frames
    }


def save_results(results, output_prefix):
    """Save analysis results to files."""
    # Save the energy maps array
    np.save(f"{output_prefix}_hbond_energy_maps.npy", results['energy_maps'])
    
    # Save other relevant info
    with open(f"{output_prefix}_hbond_results.pkl", 'wb') as f:
        pickle.dump({
            'donors': results['donors'],
            'acceptors': results['acceptors'],
            'n_residues': results['n_residues'],
            'n_frames': results['n_frames']
        }, f)
        
    print(f"Results saved with prefix: {output_prefix}")
    print(f"  - {output_prefix}_hbond_energy_maps.npy")
    print(f"  - {output_prefix}_hbond_results.pkl")


def main():
    parser = argparse.ArgumentParser(description='Analyze hydrogen bond energy maps in a trajectory.')
    parser.add_argument('trajectory_file', help='Path to trajectory file (.up)')
    parser.add_argument('output_prefix', help='Prefix for output files')
    parser.add_argument('--stride', type=int, default=10, help='Stride for trajectory reading (default: 10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file {args.trajectory_file} not found")
        sys.exit(1)
    
    results = run_hbond_analysis(args.trajectory_file, stride=args.stride)
    save_results(results, args.output_prefix)
    
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()

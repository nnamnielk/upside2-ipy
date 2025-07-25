import os
import glob
import subprocess

def process_pdb_files():
    """
    For every file with -top in the name in the folder 'native/',
    this script will run the command 'utils/PDB_to_initial_structure.py'.
    This script PDB_to_initial_structure is supposed to take the .pdb file
    and convert it into .npy files and so forth, as input for a program called upside.
    Save these input files in 'inputs/'. The name of the file should follow the
    PDB id (which is the part of the file name that occurs before "-top").
    """
    # Ensure the output directory exists
    output_dir = 'inputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all files with -top in the name in the native/ directory
    pdb_files = glob.glob('native/*-top.pdb')

    for pdb_file in pdb_files:
        # Extract the PDB ID from the filename
        pdb_id = os.path.basename(pdb_file).split('-top.pdb')[0]

        # Construct the output basename
        output_basename = os.path.join(output_dir, pdb_id)

        # Construct the command to run
        python_executable = os.path.join('.venv', 'Scripts', 'python.exe')
        command = [
            python_executable,
            'utils/PDB_to_initial_structure.py',
            pdb_file,
            output_basename
        ]

        # Run the command
        print(f"Running command: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {pdb_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pdb_file}: {e}")
        print("-" * 20)

if __name__ == "__main__":
    process_pdb_files()

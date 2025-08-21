import h5py
import numpy as np
import sys
import os

def read_fasta_sequence(filepath):
    """Reads a sequence from a FASTA file."""
    aa_conv_dict = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU",
                    "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
                    "Y": "TYR", "V": "VAL",}
    with open(filepath, 'r') as f:
        sequence_1_letter = ""
        for line in f:
            if not line.startswith('>'):
                sequence_1_letter += line.strip()
    sequence_3_letter = [aa_conv_dict[c] for c in sequence_1_letter]
    return np.array(sequence_3_letter, dtype='S3')

def convert_h5_to_up(h5_path, protein_name, up_path):
    """
    Converts an .h5 file to a .up file by combining the output from the .h5 file
    with input data from other sources.
    """
    template_path = 'chig.run.up'
    fasta_path = os.path.join('inputs', f'{protein_name}.fasta')
    npy_path = os.path.join('inputs', f'{protein_name}.initial.npy')

    try:
        with h5py.File(h5_path, 'r') as h5_file, \
             h5py.File(template_path, 'r') as template_file, \
             h5py.File(up_path, 'w') as up_file:

            # 1. Copy the 'output' group from the source .h5 file
            print(f"Copying 'output' group from {h5_path}...")
            if 'output' in h5_file:
                h5_file.copy('output', up_file)
            else:
                print(f"'output' group not found in {h5_path}. Skipping.")


            # 2. Copy the 'input' group from the template .up file
            print(f"Copying 'input' group from {template_path}...")
            if 'input' in template_file:
                template_file.copy('input', up_file)
            else:
                print(f"'input' group not found in {template_path}. Cannot proceed.")
                return

            # 3. Overwrite specific datasets in the 'input' group
            print("Overwriting 'input/pos' and 'input/sequence'...")
            
            # Overwrite 'input/pos'
            if os.path.exists(npy_path):
                initial_pos = np.load(npy_path)
                # The shape required by the loader is (n_atoms, 3, 1)
                reshaped_pos = np.expand_dims(initial_pos, axis=2)
                
                if 'pos' in up_file['input']:
                    del up_file['input/pos']
                up_file.create_dataset('input/pos', data=reshaped_pos, dtype='f4')
            else:
                print(f"Warning: {npy_path} not found. 'input/pos' will be from template.")

            # Overwrite 'input/sequence'
            if os.path.exists(fasta_path):
                sequence_data = read_fasta_sequence(fasta_path)
                if 'sequence' in up_file['input']:
                    del up_file['input/sequence']
                up_file.create_dataset('input/sequence', data=sequence_data, dtype='S3')
            else:
                print(f"Warning: {fasta_path} not found. 'input/sequence' will be from template.")

        print(f"Successfully created {up_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_h5_to_up.py <source_h5_path> <protein_name> <output_up_path>")
        sys.exit(1)
    
    source_h5 = sys.argv[1]
    protein = sys.argv[2]
    output_up = sys.argv[3]
    
    convert_h5_to_up(source_h5, protein, output_up)

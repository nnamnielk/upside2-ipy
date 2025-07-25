import h5py
import sys
import numpy as np

def print_hdf5_structure(name, obj):
    """Prints the name and type of groups and datasets."""
    print(name, end="")
    if isinstance(obj, h5py.Group):
        print(" (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f" (Dataset: shape={obj.shape}, dtype={obj.dtype})")
        # Optionally, print some data for small datasets
        if np.prod(obj.shape) < 10:
            print(f"   Data: {obj[...]}")
    else:
        print(" (Unknown object)")

def inspect_h5_file(filepath):
    """Opens an HDF5 file and prints its structure."""
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Inspecting file: {filepath}")
            f.visititems(print_hdf5_structure)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python h5_inspector.py <path_to_h5_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    inspect_h5_file(file_path)

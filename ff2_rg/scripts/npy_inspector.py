import numpy as np
import sys

def inspect_npy_file(filepath):
    """Opens a .npy file and prints its shape and dtype."""
    try:
        data = np.load(filepath)
        print(f"Inspecting file: {filepath}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        # Optionally, print a small sample of the data
        if np.prod(data.shape) < 20:
            print(f"  Data: {data}")
        else:
            print(f"  Data sample: {data.flatten()[:10]}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python npy_inspector.py <path_to_npy_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    inspect_npy_file(file_path)

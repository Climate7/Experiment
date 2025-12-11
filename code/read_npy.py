import numpy as np
import argparse
import os

def read_npy_info(file_path, output_file=None):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    output_lines = []
    
    def log(message):
        print(message)
        output_lines.append(str(message))

    try:
        data = np.load(file_path, allow_pickle=True)
        
        log(f"--- Information for: {os.path.basename(file_path)} ---")
        log(f"File Path: {os.path.abspath(file_path)}")
        log(f"Type: {type(data)}")
        
        if isinstance(data, np.ndarray):
            log(f"Shape: {data.shape}")
            log(f"Data Type: {data.dtype}")
            log(f"Size: {data.size}")
            log(f"Number of Dimensions: {data.ndim}")
            
            log("\n--- Data Content (Preview) ---")
            log(data)
        else:
            log("\n--- Data Content ---")
            log(data)
        
        if output_file:
            # Ensure directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\nOutput saved to: {output_file}")
            
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and display information about a .npy file.")
    parser.add_argument("file_path", help="Path to the .npy file")
    parser.add_argument("-o", "--output", help="Path to save the output text file", default=None)
    
    args = parser.parse_args()
    
    read_npy_info(args.file_path, args.output)

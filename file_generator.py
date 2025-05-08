# generate_files_variable_size.py
import os
import json
import time
import argparse
import random
import string
import math
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def generate_random_string(length):
    """Generates a random string of fixed length."""
    # Ensure length is non-negative
    length = max(0, int(length))
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def generate_single_file_variable_size(args_tuple):
    """
    Worker function to generate a single JSON file with variable size.
    Accepts a tuple of arguments for use with pool.imap.
    """
    index, output_dir, min_mib, max_mib, file_prefix = args_tuple
    file_path = os.path.join(output_dir, f"{file_prefix}_{index:09d}.json") # Pad index

    # --- Determine random target size for this file ---
    target_mib = random.uniform(min_mib, max_mib)
    target_bytes_total = int(target_mib * 1024**2) # Convert MiB to bytes

    # --- Calculate payload size ---
    # Estimate JSON overhead (keys, quotes, commas, braces) - very approximate
    # For larger files, the payload dominates, so overhead is less critical percentage-wise
    base_overhead = 200 # Slightly increased estimate for potentially more complex structure/keys
    target_payload_bytes = max(10, target_bytes_total - base_overhead)
    payload_len = target_payload_bytes # Assuming mostly 1-byte chars in payload

    # --- Create JSON content ---
    data = {
        "id": index,
        "target_mib_size": round(target_mib, 3), # Store the intended size
        "timestamp": time.time(),
        "payload": generate_random_string(payload_len), # The large part
        "numeric_feature": random.random() * 1000,
        "category": random.choice(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]),
        "metadata": {
            "source": f"generator_varsize_{random.randint(1, 5)}",
            "quality_score": random.uniform(0.1, 1.0),
            "checksum": hex(random.randint(0, 2**32-1)) # Dummy checksum
        },
        "nested_list": [random.random() for _ in range(random.randint(5, 20))]
    }

    # --- Write file ---
    try:
        with open(file_path, 'w') as f:
            # Use compact separators for slightly smaller file size
            json.dump(data, f, indent=None, separators=(',', ':'))
        return True # Indicate success
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")
        return False # Indicate failure
    except Exception as e:
        print(f"Unexpected error for file {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate a large number of small JSON files with variable sizes using multiprocessing.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated JSON files.")
    parser.add_argument("--total_size_gb", type=float, default=10.0, help="Target total size of all files in GiB (approximate due to variable file sizes).")
    # New arguments for size range
    parser.add_argument("--min_mib_per_file", type=float, default=0.5, help="Minimum size of each individual file in MiB.")
    parser.add_argument("--max_mib_per_file", type=float, default=5.0, help="Maximum size of each individual file in MiB.")

    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (defaults to CPU count).")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size for multiprocessing.imap.")
    parser.add_argument("--file_prefix", type=str, default="vardata", help="Prefix for generated filenames.")

    args = parser.parse_args()

    # --- Input Validation ---
    if args.min_mib_per_file <= 0 or args.max_mib_per_file <= 0:
        print("Error: min/max MiB per file must be positive.")
        return
    if args.min_mib_per_file > args.max_mib_per_file:
        print("Error: min_mib_per_file cannot be greater than max_mib_per_file.")
        return

    # --- Calculations ---
    target_total_bytes = args.total_size_gb * 1024**3
    # Estimate average file size for calculating the number of files needed
    avg_mib_per_file = (args.min_mib_per_file + args.max_mib_per_file) / 2.0
    avg_bytes_per_file_est = avg_mib_per_file * 1024**2

    num_files_to_generate = math.ceil(target_total_bytes / avg_bytes_per_file_est)

    print(f"--- Configuration ---")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Total Size: {args.total_size_gb:.2f} GiB (approximate)")
    print(f"File Size Range: {args.min_mib_per_file:.2f} MiB - {args.max_mib_per_file:.2f} MiB")
    print(f"Estimated Avg MiB/File: {avg_mib_per_file:.2f} MiB")
    print(f"Estimated Files to Generate: {num_files_to_generate:,}")
    print(f"File Prefix: {args.file_prefix}")
    print(f"Chunk Size: {args.chunk_size}")

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} worker processes.")

    # Prepare arguments for worker processes
    tasks = [(i, args.output_dir, args.min_mib_per_file, args.max_mib_per_file, args.file_prefix)
             for i in range(num_files_to_generate)]

    print(f"\nStarting file generation...")
    start_time = time.time()
    files_generated = 0
    errors = 0

    # --- Multiprocessing ---
    try:
        with Pool(processes=num_workers) as pool:
            results_iterator = pool.imap(generate_single_file_variable_size, tasks, chunksize=args.chunk_size)

            for success in tqdm(results_iterator, total=num_files_to_generate, desc="Generating Files"):
                if success:
                    files_generated += 1
                else:
                    errors += 1

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during multiprocessing: {e}")
    finally:
        pass # Pool is closed automatically

    # --- Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Generation Summary ---")
    print(f"Successfully generated: {files_generated:,} files")
    if errors > 0:
        print(f"Errors encountered: {errors:,}")
    print(f"Total duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")

    # Calculate actual total size (optional, can be slow for many files)
    # total_bytes_generated = sum(os.path.getsize(os.path.join(args.output_dir, f)) for f in os.listdir(args.output_dir) if f.startswith(args.file_prefix) and f.endswith('.json'))
    # print(f"Actual total size generated: {total_bytes_generated / 1024**3:.2f} GiB")

    if files_generated > 0 and duration > 0:
        files_per_sec = files_generated / duration
        print(f"Files generated per second: {files_per_sec:.2f}")

    print("File generation process finished.")

if __name__ == "__main__":
    main()
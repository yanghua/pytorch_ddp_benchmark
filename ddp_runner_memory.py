# train.py
# WARNING: This script intentionally removes most error handling (try...except blocks)
# as requested. This makes it EXTREMELY brittle and prone to crashing if ANY
# file is missing, corrupted, unreadable, or if any other unexpected issue
# occurs during data loading, processing, or training.
# This is generally NOT recommended for robust training or debugging.
# Use with caution, primarily for specific low-level debugging scenarios where
# you want the process to crash immediately upon error.

import os
import argparse
import time
import random
import datetime
import json
import math
import sys # For exiting on error if needed

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm # Optional: for epoch progress on rank 0

# --- Configuration ---
MAX_SEED_VAL = 2**32 - 1
# Example: Assume we extract/create features of this dimension from JSON
TARGET_FEATURE_DIM = 512

# --- Utility Functions (Standard DDP Setup - without try/except) ---

def setup_ddp(rank, world_size):
    """Initializes the distributed environment (NO ERROR HANDLING)."""
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    timeout = datetime.timedelta(minutes=30) # Increased timeout

    # Assume MASTER_ADDR and MASTER_PORT are set by torchrun
    # Direct call without try/except - will crash on failure
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)

    print(f"Rank {rank}/{world_size}: Process group initialized (Backend: {backend}, Timeout: {timeout}).")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Set device to CUDA:{rank}")

def cleanup_ddp():
    """Cleans up the distributed environment."""
    # Check if initialized before destroying
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed environment cleaned up.")

def set_seeds(seed, rank):
    """Sets random seeds for reproducibility across processes."""
    process_seed = (seed + rank) % MAX_SEED_VAL
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

def main_print(content, rank):
    """Prints content only from the main process (rank 0)."""
    if rank == 0:
        print(content)

# --- Dataset Class for Variable-Sized JSON Files (NO ERROR HANDLING in __getitem__) ---

class JsonFileDatasetVariableUnsafe(Dataset):
    """
    Dataset that reads data from individual variable-sized JSON files.
    WARNING: __getitem__ has NO error handling and will crash on bad files/data.
    """
    def __init__(self, data_dir, rank=0, world_size=1, file_prefix="vardata", max_files=None):
        self.data_dir = data_dir
        self.file_paths = []
        self.rank = rank # Store rank for potential worker-specific logging

        # Basic check before slow scan
        if not os.path.isdir(data_dir):
             # Use main_print for consistency, then raise to stop execution
             main_print(f"FATAL ERROR (Rank {rank}): Data directory not found: {data_dir}", rank)
             # All ranks should see this and ideally stop
             raise FileNotFoundError(f"Rank {rank}: Data directory not found: {data_dir}")

        # --- File Listing (Still has basic error handling for the scan itself) ---
        main_print(f"Rank {rank}: Scanning directory '{data_dir}' for '{file_prefix}_*.json' files...", rank)
        start_scan = time.time()

        all_files_in_dir = os.listdir(data_dir)
        matching_files = [f for f in all_files_in_dir if f.startswith(file_prefix) and f.endswith('.json')]
        # CRITICAL: Sort filenames to ensure consistent order across all ranks
        matching_files.sort()
        self.file_paths = [os.path.join(data_dir, f) for f in matching_files]

        if max_files is not None and max_files > 0:
            main_print(f"Rank {rank}: Limiting dataset to first {max_files:,} files found.", rank)
            self.file_paths = self.file_paths[:max_files]

        end_scan = time.time()
        num_found = len(self.file_paths)
        main_print(f"Rank {rank}: Found {num_found:,} JSON files matching prefix '{file_prefix}' in {end_scan - start_scan:.2f} seconds.", rank)

        if not self.file_paths:
            # Use main_print, then raise
            main_print(f"FATAL ERROR (Rank {rank}): No '{file_prefix}_*.json' files found in directory: {data_dir}. Check --data_dir and --file_prefix.", rank)
            raise RuntimeError(f"Rank {rank}: No matching files found in {data_dir}")

    def __len__(self):
        print(f"+++++++++++++{len(self.file_paths)}")
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Reads, parses a single JSON file, and returns a fixed-size tensor.
        WARNING: NO ERROR HANDLING INSIDE. Will crash on IOErrors, JSONDecodeErrors,
                 KeyErrors, TypeErrors, IndexError, etc.
        """
        file_path = self.file_paths[idx]

        # --- Stage 1: File Reading and Parsing (NO try/except) ---
        # This block will crash if the file cannot be opened or read,
        # or if json.load fails on corrupted data.
        with open(file_path, 'r') as f:
            data = json.load(f)

        # --- Stage 2: Feature Extraction / Tensor Creation (NO try/except) ---
        # These lines will crash if keys are missing (KeyError),
        # if data cannot be cast to float (ValueError/TypeError), etc.
        feature1 = float(data["numeric_feature"]) # Assumes key exists
        feature2 = float(data["metadata"]["quality_score"]) # Assumes nested keys exist
        feature3 = float(len(data["payload"])) # Assumes key exists

        # Create the output tensor (fixed size for easy batching)
        tensor_data = torch.zeros(TARGET_FEATURE_DIM, dtype=torch.float32)

        # This will crash if TARGET_FEATURE_DIM < 3 (IndexError)
        tensor_data[0] = feature1
        tensor_data[1] = feature2
        tensor_data[2] = feature3

        # Optional: Simulate more complex processing
        # tensor_data += torch.randn(TARGET_FEATURE_DIM) * 0.01

        print(f"***************{len(tensor_data)}")
        return tensor_data


class MemoryCachedJsonDataset(Dataset):
    """
    Dataset that preloads all JSON data into memory during initialization.
    WARNING: __getitem__ has NO error handling and will crash on bad data.
    """
    def __init__(self, data_dir, rank=0, world_size=1, file_prefix="vardata", max_files=None):
        self.data_dir = data_dir
        self.file_paths = []
        self.rank = rank
        self.cached_data = []  # 内存缓存存储解析后的JSON数据

        # 文件扫描逻辑与原实现保持一致
        if not os.path.isdir(data_dir):
            main_print(f"FATAL ERROR (Rank {rank}): Data directory not found: {data_dir}", rank)
            raise FileNotFoundError(f"Rank {rank}: Data directory not found: {data_dir}")

        main_print(f"Rank {rank}: Scanning directory '{data_dir}' for '{file_prefix}_*.json' files...", rank)
        start_scan = time.time()

        all_files_in_dir = os.listdir(data_dir)
        matching_files = [f for f in all_files_in_dir if f.startswith(file_prefix) and f.endswith('.json')]
        matching_files.sort()
        self.file_paths = [os.path.join(data_dir, f) for f in matching_files]

        if max_files is not None and max_files > 0:
            main_print(f"Rank {rank}: Limiting dataset to first {max_files:,} files found.", rank)
            self.file_paths = self.file_paths[:max_files]

        end_scan = time.time()
        num_found = len(self.file_paths)
        main_print(f"Rank {rank}: Found {num_found:,} JSON files in {end_scan - start_scan:.2f} seconds.", rank)

        if not self.file_paths:
            main_print(f"FATAL ERROR (Rank {rank}): No matching files found.", rank)
            raise RuntimeError(f"Rank {rank}: No files found")

        # 新增：预加载所有JSON数据到内存
        main_print(f"Rank {rank}: Loading {len(self.file_paths)} JSON files into memory...", rank)
        load_start = time.time()
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                self.cached_data.append(json.load(f))  # 直接存储解析后的字典
        load_end = time.time()
        main_print(f"Rank {rank}: Successfully cached {len(self.cached_data)} files in {load_end - load_start:.2f}s", rank)

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        """
        直接从内存获取预加载的JSON数据，不再访问磁盘
        """
        data = self.cached_data[idx]  # 从内存获取数据

        # 以下特征提取逻辑保持与原实现完全一致
        feature1 = float(data["numeric_feature"])
        feature2 = float(data["metadata"]["quality_score"])
        feature3 = float(len(data["payload"]))

        tensor_data = torch.zeros(TARGET_FEATURE_DIM, dtype=torch.float32)
        tensor_data[0] = feature1
        tensor_data[1] = feature2
        tensor_data[2] = feature3

        return tensor_data




# --- Minimal Model for Testing ---

class SimpleModel(nn.Module):
    """A very basic model that just performs a few operations."""
    def __init__(self, input_dim=TARGET_FEATURE_DIM, hidden_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1) # Output a single value

    def forward(self, x):
        # Perform minimal computation to ensure gradients flow
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

# --- Training Function (Minimal Error Handling) ---

def run_load_test(rank, world_size, args):
    """Main DDP load testing loop (minimal error handling)."""
    main_print(f"--- Starting Load Test Rank {rank}/{world_size} ---", rank)

    # Direct call - will crash on failure
    setup_ddp(rank, world_size)
    set_seeds(args.seed, rank)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        main_print(f"Rank {rank}: Using Device: CUDA:{rank}", rank)
    else:
        device = torch.device("cpu")
        main_print(f"Rank {rank}: Using Device: CPU", rank)

    # 1. Create Dataset (Initialization has minimal error checks inside)
    # If the constructor raises an error, the script will stop here.
    dataset = MemoryCachedJsonDataset(
        args.data_dir,
        rank,
        world_size,
        args.file_prefix,
        args.max_files # Limit files for quicker testing runs if needed
    )

    # --- Barrier after dataset init ---
    if dist.is_initialized() and world_size > 1:
        main_print(f"Rank {rank}: Waiting at barrier after dataset init...", rank)
        dist.barrier()
        main_print(f"Rank {rank}: Passed dataset init barrier.", rank)

    # 2. Create Distributed Sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True
    )
    main_print(f"Rank {rank}: DistributedSampler created (Shuffle=True, DropLast=True).", rank)

    # 3. Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
        timeout=300 if args.num_workers > 0 else 0
    )
    main_print(f"Rank {rank}: DataLoader created (Workers={args.num_workers}, Batch={args.batch_size}, Prefetch={args.prefetch_factor}, Persistent={args.num_workers > 0}).", rank)

    # 4. Create Minimal Model and move to device
    model = SimpleModel(input_dim=TARGET_FEATURE_DIM).to(device)

    # 5. Wrap Model with DDP
    model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None, find_unused_parameters=False)
    main_print(f"Rank {rank}: Model wrapped with DDP.", rank)

    # 6. Create Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --- Load Testing Loop ---
    main_print(f"Rank {rank}: Starting load test loop for {args.epochs} epochs.", rank)
    model.train()
    total_steps = 0
    global_start_time = time.time()

    interval_load_times = []
    interval_train_times = []
    interval_batch_times = []
    interval_start_time = time.time()

    # Outer loop for epochs
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch) # Essential for DDP shuffling
        main_print(f"Rank {rank}: Starting Epoch {epoch}", rank)
        #if rank == 0:
             #epoch_progress = tqdm(total=len(dataloader), desc=f"Epoch {epoch}", unit="batch", file=sys.stdout) # Ensure tqdm prints to stdout

        batch_iter_start_time = time.time() # Start timing for first batch load

        # Inner loop for batches
        # NO try/except around the dataloader iteration. If a worker crashes
        # due to an error in __getitem__, the behavior here can be unpredictable
        # (hang, crash, etc.)
        for i, batch_data in enumerate(dataloader):
            # --- Measure Load Time ---
            #print(f"------------------ {len(batch_data)}, i is {i}")
            batch_load_end_time = time.time()
            current_load_time = batch_load_end_time - batch_iter_start_time
            if rank == 0: interval_load_times.append(current_load_time)

            # --- Move Data to Device (NO try/except) ---
            # This will crash if there's a CUDA error, OOM, etc.
            data_tensor = batch_data.to(device, non_blocking=True)
            batch_move_end_time = time.time()

            # --- Minimal Train Step (NO try/except) ---
            # These steps will crash on OOM, CUDA errors, numerical issues, etc.
            outputs = model(data_tensor)
            loss = outputs.mean() * 0.001 # Dummy loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # --- End Train Step ---

            batch_train_end_time = time.time()
            current_train_time = batch_train_end_time - batch_move_end_time
            current_batch_time = batch_train_end_time - batch_iter_start_time
            if rank == 0:
                interval_train_times.append(current_train_time)
                interval_batch_times.append(current_batch_time)

            total_steps += 1

            # --- Logging (Rank 0 Only) ---
            if rank == 0:
                #epoch_progress.update(1)
                if (i + 1) % args.log_interval == 0 and len(interval_load_times) > 0:
                    avg_load = sum(interval_load_times) / len(interval_load_times)
                    avg_train = sum(interval_train_times) / len(interval_train_times)
                    avg_batch = sum(interval_batch_times) / len(interval_batch_times)

                    interval_duration = time.time() - interval_start_time
                    samples_processed = args.log_interval * args.batch_size * world_size
                    throughput = samples_processed / interval_duration if interval_duration > 0 else 0

                    #epoch_progress.set_postfix({
                    #    "Load(s)": f"{avg_load:.4f}",
                    #    "Train(s)": f"{avg_train:.4f}",
                    #    "Batch(s)": f"{avg_batch:.4f}",
                    #    "Samples/s": f"{throughput:.1f}"
                    #}, refresh=True) # Refresh postfix

                    # Reset interval stats
                    interval_load_times = []
                    interval_train_times = []
                    interval_batch_times = []
                    interval_start_time = time.time()

            # --- Prepare for next iteration ---
            batch_iter_start_time = time.time()

            # Optional: Add step limit for quick tests
            if args.max_steps is not None and total_steps >= args.max_steps:
                main_print(f"Rank {rank}: Reached max_steps ({args.max_steps}), stopping.", rank)
                # Use break to exit inner loop, outer loop check will handle epoch exit
                break

        # --- End of Epoch ---
        #if rank == 0:
            #epoch_progress.close()

        # Check step limit after epoch finishes as well
        if args.max_steps is not None and total_steps >= args.max_steps:
            break # Exit outer loop (epochs)


    # --- End Training ---
    global_end_time = time.time()
    total_duration = global_end_time - global_start_time
    main_print(f"\n--- Load Test Finished (Rank {rank}) ---", rank)
    main_print(f"Total Steps: {total_steps}", rank)
    main_print(f"Total Duration: {total_duration:.2f} seconds", rank)
    if total_steps > 0 and total_duration > 0 and world_size > 0 and args.batch_size > 0:
         # Ensure no division by zero
         global_samples_processed = total_steps * args.batch_size * world_size
         avg_throughput = global_samples_processed / total_duration
         main_print(f"Average Global Throughput: {avg_throughput:.2f} samples/sec", rank)

    cleanup_ddp()

# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP DataLoader Performance Test (NO ERROR HANDLING)")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the generated variable-size JSON files.")
    parser.add_argument("--file_prefix", type=str, default="vardata", help="Prefix used for the generated filenames (e.g., 'vardata').")
    parser.add_argument("--max_files", type=int, default=None, help="Optional: Limit total number of files used from the dataset for quicker tests.")

    # DDP/Training arguments
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs for the test.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size PER GPU. Adjust based on GPU memory.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (minimal impact on load test).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log_interval", type=int, default=50, help="Log performance metrics every N batches (on rank 0).")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional: Maximum total training steps to run.")


    # DataLoader arguments (CRITICAL for performance testing)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers per process (GPU). Tune this heavily!")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches loaded in advance by each worker (requires num_workers > 0).")

    args = parser.parse_args()

    # --- Get DDP Rank and World Size (from torchrun environment) ---
    # Assume environment variables are set, otherwise this will fail later
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"DDP Environment: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    # Set base seed consistently across ranks before adjusting per rank
    args.seed = args.seed if args.seed is not None else random.randint(0, MAX_SEED_VAL)
    main_print(f"Using base seed: {args.seed}", rank)

    # Start the load test process
    # No try/except around the main function call
    run_load_test(rank, world_size, args)

    main_print("Script finished normally.", rank)

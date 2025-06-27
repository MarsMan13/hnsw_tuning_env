import os

# IMPORTANT: Set these environment variables BEFORE importing numpy, torch, etc.
# This prevents each worker process from spawning its own threads, which would
# lead to thread over-subscription and performance degradation.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import itertools
import multiprocessing

from src.utils import load_search_results, plot_searched_points_3d
from src.solutions import print_optimal_hyperparameters
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze results of hyperparameter tuning in parallel.")
    parser.add_argument("-o", "--optima", action="store_true", default=False, help="Print optimal hyperparameters.")
    parser.add_argument("-g", "--graph", action="store_true", default=False, help="Generate 3D plots.")
    parser.add_argument("--surface", action="store_true", default=False, help="Use surface plot instead of scatter for 3D graphs.")
    parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores to use for parallel processing.")
    return parser.parse_args()

def worker(params):
    """
    A single worker function that processes one combination of parameters.
    This function is executed by each process in the multiprocessing pool.
    """
    # Unpack parameters for the current task
    solution, impl, dataset, recall_min, qps_min, args = params

    # Construct filename and load results
    try:
        if recall_min:
            metric_str = f"{recall_min}r"
            filename = f"{solution}_{impl}_{dataset}_{recall_min}r_{None}q.csv"
        else:
            metric_str = f"{qps_min}q"
            filename = f"{solution}_{impl}_{dataset}_{None}r_{qps_min}q.csv"

        print(f"[Worker {os.getpid()}] Processing: {solution}, {impl}, {dataset}, {metric_str}")
        results = load_search_results(solution, filename)
        if not results:
            print(f"Warning: No results found for {filename}. Skipping.")
            return f"No results for {filename}"
    except Exception as e:
        return f"Error loading {filename}: {e}"

    # --- Perform analysis based on arguments ---
    try:
        # Print optimal hyperparameters if requested
        if args.optima:
            print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)

        # Generate 3D graph if requested
        if args.graph:
            plot_searched_points_3d(
                results,
                solution=solution,
                filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{metric_str}.png",
                recall_min=recall_min,
                qps_min=qps_min,
                sampling_count=10, # Or pass this as a parameter if it varies
                surface=args.surface,
            )
        return f"Successfully processed {filename}"
    except Exception as e:
        return f"Error processing {filename}: {e}"

def main():
    """
    Main function to set up and run the parallel analysis.
    """
    args = parse_args()

    # --- Configuration for the experiments ---
    SOLUTIONS = ["brute_force"]
    IMPLS = ["faiss", "hnswlib"]
    DATASETS = [
        "glove-100-angular",
        "nytimes-256-angular",
        "sift-128-euclidean",
        "youtube-1024-angular"
    ]
    RECALL_MINS = [0.90, 0.95, 0.975, 0.99]

    # --- Prepare a list of all tasks to be executed ---
    tasks = []

    # 1. Prepare tasks for recall_min constraint
    recall_tasks_params = itertools.product(SOLUTIONS, IMPLS, DATASETS, RECALL_MINS)
    for solution, impl, dataset, recall_min in recall_tasks_params:
        # Each task is a tuple of arguments for the worker function
        tasks.append((solution, impl, dataset, recall_min, None, args))

    # 2. Prepare tasks for qps_min constraint
    qps_tasks_params = itertools.product(SOLUTIONS, IMPLS, DATASETS)
    for solution, impl, dataset in qps_tasks_params:
        for qps_min in get_qps_metrics_dataset(impl=impl, dataset=dataset):
            tasks.append((solution, impl, dataset, None, qps_min, args))

    print(f"--- Starting parallel analysis for {len(tasks)} tasks using {args.cores} cores ---")

    # --- Run tasks in parallel using a multiprocessing Pool ---
    # The 'with' statement ensures the pool is properly closed
    try:
        with multiprocessing.Pool(processes=args.cores) as pool:
            # imap_unordered processes tasks in parallel and returns results as they complete
            for i, result in enumerate(pool.imap_unordered(worker, tasks), 1):
                print(f"({i}/{len(tasks)}) Task finished: {result}")
    except Exception as e:
        print(f"A critical error occurred in the main pool: {e}")

    print("--- All analysis tasks completed. ---")

if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly on all platforms.
    main()
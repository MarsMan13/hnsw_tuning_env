import itertools
import multiprocessing
from functools import partial
import numpy as np
from scipy.stats import spearmanr

from src.constants import TUNING_BUDGET, MAX_SAMPLING_COUNT, SEED
from src.utils import (get_local_optimal_hyperparameter, get_optimal_hyperparameter, 
                       plot_efS_3d, plot_multi_accumulated_timestamp, 
                       plot_searched_points_3d, plot_timestamp, 
                       save_search_results, load_search_results)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset



def postprocess_results(solution, impl, dataset,
                        recall_min=None, qps_min=None, sampling_count=MAX_SAMPLING_COUNT, seed=SEED, tuning_budget=TUNING_BUDGET):
    results = load_search_results(solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv", 
                                  seed=seed, sampling_count=sampling_count)
    plot_timestamp(
        results, 
        solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    plot_multi_accumulated_timestamp(
        {solution: results},
        dirname=solution,
        filename=f"{solution}_multi_accumulated_timestamp_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    plot_searched_points_3d(
        results, 
        solution=solution, 
        filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    plot_efS_3d(
        results, 
        solution=solution, 
        filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )

def main():
    SOLUTIONS = [
        # "brute_force",
        "our_solution",
        # "grid_search",
        # "random_search",
        # "vd_tuner",
        # "optuna",
    ]
    IMPLS = [
        # "faiss",
        "hnswlib",
        # "milvus",
    ]
    DATASETS = [
        # "nytimes-256-angular",
        # "glove-100-angular",
        # "sift-128-euclidean",
        # "youtube-1024-angular",
        # "deep1M-256-angular",
        "nytimes-256-angular-100p-hnswlib-random",
    ]
    SAMPLING_COUNT = [
        # 1,
        # 3,
        # 5,
        10,
    ]
    RECALL_MINS = [
        # 0.90, 
        # 0.925, 
        # 0.95, 
        # 0.975, 
        # 0.99
    ]
    MOCK_SEED = "42"
    # --- Start of multiprocessing modification ---

    #* 1. Create a list to hold all the tasks to be executed.
    # A task is a tuple of arguments for the _process_single_metric function.
    tasks = []

    all_iters = itertools.product(SOLUTIONS, IMPLS, DATASETS, SAMPLING_COUNT)
    print("--- Preparing tasks for parallel execution ---")
    for solution, impl, dataset, sampling_count in all_iters:
        # Prepare tasks for each recall_min value
        for recall_min in RECALL_MINS:
            # Add a tuple of arguments for the worker function
            task_args = (solution, impl, dataset, recall_min, None, sampling_count, MOCK_SEED)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, recall_min={recall_min}, sampling={sampling_count}")

        # Prepare tasks for each qps_min value
        for qps_min in get_qps_metrics_dataset(impl, dataset):
            # Add a tuple of arguments for the worker function
            task_args = (solution, impl, dataset, None, qps_min, sampling_count, MOCK_SEED)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, qps_min={qps_min}, sampling={sampling_count}")

    print("\n--- All tasks prepared. Starting parallel processing. ---")

    #* 2. Use a multiprocessing Pool to execute tasks in parallel.
    # It's recommended to use a number of processes equal to the number of CPU cores.
    try:
        # Use all available CPU cores, or specify a number.
        num_processes = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        print(f"Creating a pool of {num_processes} worker processes for {len(tasks)} tasks.")
        # 'with' statement ensures the pool is properly closed after use.
        with multiprocessing.Pool(processes=num_processes) as pool:
            # pool.starmap takes a function and an iterable of argument tuples.
            # It unpacks each tuple and calls the function with those arguments.
            # e.g., for a task (a, b, c), it calls _process_single_metric(a, b, c)
            results = pool.starmap(postprocess_results, tasks)
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")

    #* 3. All tasks are completed.
    print("\n--- Parallel processing finished. ---")


if __name__ == "__main__":
    main()
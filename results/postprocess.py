import itertools
import multiprocessing
from functools import partial

from src.constants import TUNING_BUDGET
from src.utils import filename_builder, get_optimal_hyperparameter, load_search_results, \
    plot_multi_accumulated_timestamp, plot_searched_points_3d, plot_timestamp, save_optimal_hyperparameters
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

# def process_file():
#     SOLUTION = "vd_tuner"
#     FILENAME = "vd_tuner_hnswlib_nytimes-256-angular_8h_False_3.csv"
#     TUNING_BUDGET = 3600 * 8
#     RECALL_MIN = 0.95
#     results = load_search_results(SOLUTION, FILENAME)
#     _FILENAME = FILENAME.split(".csv")[0]
#     plot_timestamp(results, SOLUTION, f"{_FILENAME}_timestamp_plot.png", recall_min=RECALL_MIN)
#     plot_searched_points_3d(results, SOLUTION, f"{_FILENAME}_searched_points_3d.png", recall_min=RECALL_MIN)

MOCK_SEED = 0

def _process_single_metric(
    impl: str, 
    dataset: str, 
    solutions: list, 
    recall_min: float = None, 
    qps_min: int = None,
    sampling_count: int = None
):
    """
    Helper function to process results for a single metric (either recall_min or qps_min).
    It loads data, plots timestamp, and saves optimal hyperparameters.
    """
    # Announce which process is handling which task
    metric_type = "recall" if recall_min is not None else "qps"
    metric_value = recall_min if recall_min is not None else qps_min
    print(f"[Process {multiprocessing.current_process().pid}] Processing: {impl}, {dataset}, {metric_type}={metric_value}, sampling={sampling_count}")

    results_combi = {}
    optimal_combi = {}

    #* 1. Load results for all solutions under the given condition
    for solution in solutions:
        filename = filename_builder(
            solution, impl, dataset, recall_min, qps_min
        )
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)
        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(
                results, recall_min=recall_min, qps_min=qps_min
            )
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1] 
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            optimal_hp = (hp, perf)
            results = [optimal_hp]  # For brute_force, we only keep the optimal hyperparameter
        results_combi[solution] = results

    #* 2. Determine metric type and value for file naming and plotting
    metric_type = "recall" if recall_min is not None else "qps"
    metric_value = recall_min if recall_min is not None else qps_min

    #* 3. Plotting accumulated_timestamp
    plot_multi_accumulated_timestamp(
        results=results_combi,
        dirname="all",
        filename=f"{impl}_{dataset}_{metric_type}_{metric_value}_accumulated.png",
        recall_min=recall_min,
        qps_min=qps_min,
        tuning_budget=TUNING_BUDGET,
        seed=MOCK_SEED,
        sampling_count=sampling_count,
    )

    #* 4. Save Optimal Hyperparameters of each solution
    for solution, results in results_combi.items():
        optimal_combi[solution] = get_optimal_hyperparameter(
            results, recall_min=recall_min, qps_min=qps_min
        )
    
    save_optimal_hyperparameters(
        impl=impl,
        dataset=dataset,
        optimal_combi=optimal_combi,
        recall_min=recall_min,
        qps_min=qps_min,
        seed=MOCK_SEED,
        sampling_count=sampling_count,
    )
    
    # ! 5) TODO for the combined logic can be placed here


def main():
    SOLUTIONS = [
        "brute_force",
        "our_solution",
        "grid_search",
        "random_search",
        "vd_tuner",
        # "grid_search_heuristic",
        # "random_search_heuristic",
    ]
    IMPLS = [
        "faiss",
        "hnswlib"
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
    ]
    SAMPLING_COUNT = [
        10,
        1, 
        3, 
        5, 
    ]
    RECALL_MINS = [0.90, 0.95, 0.975, 0.99]

    # --- Start of multiprocessing modification ---

    #* 1. Create a list to hold all the tasks to be executed.
    # A task is a tuple of arguments for the _process_single_metric function.
    tasks = []
    
    all_iters = itertools.product(IMPLS, DATASETS, SAMPLING_COUNT)

    print("--- Preparing tasks for parallel execution ---")
    for impl, dataset, sampling_count in all_iters:
        # Prepare tasks for each recall_min value
        for recall_min in RECALL_MINS:
            # Add a tuple of arguments for the worker function
            task_args = (impl, dataset, SOLUTIONS, recall_min, None, sampling_count)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, recall_min={recall_min}, sampling={sampling_count}")

        # Prepare tasks for each qps_min value
        for qps_min in get_qps_metrics_dataset(dataset):
            # Add a tuple of arguments for the worker function
            task_args = (impl, dataset, SOLUTIONS, None, qps_min, sampling_count)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, qps_min={qps_min}, sampling={sampling_count}")
    
    print("\n--- All tasks prepared. Starting parallel processing. ---")

    #* 2. Use a multiprocessing Pool to execute tasks in parallel.
    # It's recommended to use a number of processes equal to the number of CPU cores.
    try:
        # Use all available CPU cores, or specify a number.

        num_processes = 12 if multiprocessing.cpu_count() <= 16 else multiprocessing.cpu_count() - 12
        print(f"Creating a pool of {num_processes} worker processes for {len(tasks)} tasks.")
        
        # 'with' statement ensures the pool is properly closed after use.
        with multiprocessing.Pool(processes=num_processes) as pool:
            # pool.starmap takes a function and an iterable of argument tuples.
            # It unpacks each tuple and calls the function with those arguments.
            # e.g., for a task (a, b, c), it calls _process_single_metric(a, b, c)
            pool.starmap(_process_single_metric, tasks)

    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")

    #* 3. All tasks are completed.
    print("\n--- Parallel processing finished. ---")
    
if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly,
    # especially on Windows and macOS. It prevents child processes from
    # re-importing and re-executing the main script's code.
    main()

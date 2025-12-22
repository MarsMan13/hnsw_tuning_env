import itertools
import multiprocessing
from functools import partial
import numpy as np
from scipy.stats import spearmanr

from src.constants import TUNING_BUDGET
from src.utils import (filename_builder, get_optimal_hyperparameter, load_search_results,
    plot_multi_accumulated_timestamp, save_optimal_hyperparameters, optimal_hyperparameters_for_times,
    # plot_multi_accumulated_timestamp_for_main_figure
)
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
    sampling_count: int = None,
    tuning_time: int = TUNING_BUDGET,
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
        else:
            results = [result for result in results if result[1][0] <= tuning_time]  # Filter results by tuning time
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
        tuning_budget=tuning_time,
        seed=MOCK_SEED,
        sampling_count=sampling_count,
    )
    #* 3.5 Plotting accumulated_timestamp for main figure
    # plot_multi_accumulated_timestamp_for_main_figure(
    #     results=results_combi,
    #     dirname="all",
    #     filename=f"{impl}_{dataset}_{metric_type}_{metric_value}_accumulated.png",
    #     recall_min=recall_min,
    #     qps_min=qps_min,
    #     tuning_budget=tuning_time,
    #     seed=MOCK_SEED,
    #     sampling_count=sampling_count,
    # )

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

    #* 5. Optimal hyperparameters for times
    optimal_hp = optimal_hp
    times_hp_combi = {}
    for solution, results in results_combi.items():
        optimal_perf = optimal_hp[1][2] if recall_min else optimal_hp[1][1]
        times_hp_combi[solution] = []
        if results and optimal_perf > 0:
            times_perf = optimal_hyperparameters_for_times(
                results, recall_min=recall_min, qps_min=qps_min
            )
            for perf in times_perf:
                times_hp_combi[solution].append(perf / optimal_perf)
        else:
            times_hp_combi[solution] = [0.0] * 4

    #* 6. Value of perfs
    spearmnr_hp_combi = {}
    for solution, results in results_combi.items():
        if recall_min:
            # filtered_results = [(perf_hp[1][0], perf_hp[1][2]) for perf_hp in results if perf_hp[1][1] >= recall_min]
            filtered_results = [(perf_hp[1][0], perf_hp[1][2]) for perf_hp in results]
        else:
            # filtered_results = [(perf_hp[1][0], perf_hp[1][1]) for perf_hp in results if perf_hp[1][2] >= qps_min]
            filtered_results = [(perf_hp[1][0], perf_hp[1][1]) for perf_hp in results]
        x1 = [perf[0] for perf in filtered_results]
        y1 = [perf[1] for perf in filtered_results]
        if len(x1) > 1:
            spearmnr_hp_combi[solution] = spearmanr(x1, y1).correlation
        else:
            spearmnr_hp_combi[solution] = 0.0
    return times_hp_combi, spearmnr_hp_combi

def main():
    SOLUTIONS = [
        "brute_force",
        "our_solution",
        "grid_search",
        "random_search",
        "vd_tuner",
        "optuna",
        # "test_solution",
        # "grid_search_heuristic",
        # "random_search_heuristic",
        # "1_tests",
        # "3_tests",
        # "5_tests",
        # "10_tests",
    ]
    IMPLS = [
        "faiss",
        "hnswlib",
        # "milvus",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
        # "deep1M-256-angular",
    ]
    SAMPLING_COUNT = [
        10,
        1,
        3,
        5,
    ]
    RECALL_MINS = [0.90, 0.925, 0.95, 0.975, 0.99]
    # --- Start of multiprocessing modification ---

    #* 1. Create a list to hold all the tasks to be executed.
    # A task is a tuple of arguments for the _process_single_metric function.
    tasks = []

    all_iters = itertools.product(IMPLS, DATASETS, SAMPLING_COUNT)
    solution_perfs_per_times = {
        solution : [] for solution in SOLUTIONS
    }
    print("--- Preparing tasks for parallel execution ---")
    for impl, dataset, sampling_count in all_iters:
        # Prepare tasks for each recall_min value
        for recall_min in RECALL_MINS:
            # Add a tuple of arguments for the worker function
            task_args = (impl, dataset, SOLUTIONS, recall_min, None, sampling_count)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, recall_min={recall_min}, sampling={sampling_count}")

        # Prepare tasks for each qps_min value
        for qps_min in get_qps_metrics_dataset(impl, dataset):
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
            results = pool.starmap(_process_single_metric, tasks)
        ####
        available_tasks = len(tasks)
        solution_to_perf = {
            solution : [0.0] * 4 for solution in SOLUTIONS
        }
        solution_to_spearmnr = {
            solution : [] for solution in SOLUTIONS
        }
        for result in results:
            times_hp_combi, stdev_hp_combi = result
            if times_hp_combi["brute_force"] == [0.0, 0.0, 0.0, 0.0]:
                available_tasks -= 1
                continue
            for solution, perf in times_hp_combi.items():
                solution_to_perf[solution] = [x + y for x, y in zip(solution_to_perf[solution], perf)]
            for solution, stdev in stdev_hp_combi.items():
                if solution in solution_to_spearmnr:
                    solution_to_spearmnr[solution].append(stdev)
        for solution in solution_to_perf:
            solution_to_perf[solution] = [x / available_tasks for x in solution_to_perf[solution]]
        print("\n--- Parallel processing completed successfully. ---")
        print("Average performance per solution:")
        for solution, perf in solution_to_perf.items():
            print(solution, end=",")
            print(','.join(f"{x:.4f}" for x in perf), end="\n")
        print("\nStandard deviation of performance per solution:")
        for solution, spm in solution_to_spearmnr.items():
            if spm:
                avg_spm = sum(spm) / len(spm)
                print(f"{solution}: {avg_spm:.4f}")
            else:
                print(f"{solution}: No valid results found.")
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")

    #* 3. All tasks are completed.
    print("\n--- Parallel processing finished. ---")

def test():
    for sampling_count in [10]:
        _process_single_metric(
            impl="faiss",
            dataset="nytimes-256-angular",
            solutions=[
                "brute_force",
                "our_solution",
                "grid_search",
                "random_search",
                "vd_tuner",
            ],
            recall_min=0.975,
            qps_min=None,
            sampling_count=sampling_count,
            tuning_time=TUNING_BUDGET,
        )

if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly,
    # especially on Windows and macOS. It prevents child processes from
    # re-importing and re-executing the main script's code.
    main()
    # test()
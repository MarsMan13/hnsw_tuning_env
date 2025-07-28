import itertools
import multiprocessing
import numpy as np
from scipy.stats import spearmanr
import traceback

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder, get_optimal_hyperparameter, load_search_results,
    plot_multi_accumulated_timestamp, save_optimal_hyperparameters,
    optimal_hyperparameters_for_times
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

MOCK_SEED = 0

def load_results_for_solutions(solutions, impl, dataset, recall_min, qps_min, sampling_count, tuning_time):
    """
    Loads search results for each solution and filters them by tuning time.
    """
    results_combi = {}
    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)
        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            perf = (0.0, *optimal_hp[1][1:])  # dummy time=0.0 + performance tuple
            results = [(hp, perf)]
        else:
            results = [result for result in results if result[1][0] <= tuning_time]
        results_combi[solution] = results
    return results_combi

def calculate_spearman_for_results(results_combi, recall_min, qps_min):
    """
    Computes Spearman correlation between tuning time and performance metric for each solution.
    """
    spearman_combi = {}
    for solution, results in results_combi.items():
        if recall_min:
            filtered = [(r[1][0], r[1][2]) for r in results if r[1][2] >= recall_min]
        else:
            filtered = [(r[1][0], r[1][1]) for r in results if r[1][1] >= qps_min]
        x, y = zip(*filtered) if filtered else ([], [])
        if len(x) > 1:
            rho, _ = spearmanr(x, y)
            spearman_combi[solution] = rho
        else:
            spearman_combi[solution] = None
    return spearman_combi

def process_single_metric(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None, tuning_time=TUNING_BUDGET):
    """
    Processes one metric: loads results, plots graphs, computes Spearman, and saves optimal hyperparameters.
    """
    metric_type = "recall" if recall_min is not None else "qps"
    metric_value = recall_min if recall_min is not None else qps_min
    print(f"[Process {multiprocessing.current_process().pid}] Processing: {impl}, {dataset}, {metric_type}={metric_value}, sampling={sampling_count}")

    results_combi = load_results_for_solutions(solutions, impl, dataset, recall_min, qps_min, sampling_count, tuning_time)

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

    optimal_combi = {
        solution: get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
        for solution, results in results_combi.items()
    }

    save_optimal_hyperparameters(
        impl=impl,
        dataset=dataset,
        optimal_combi=optimal_combi,
        recall_min=recall_min,
        qps_min=qps_min,
        seed=MOCK_SEED,
        sampling_count=sampling_count,
    )

    times_hp_combi = {}
    for solution, results in results_combi.items():
        optimal_perf = optimal_combi["brute_force"]
        best_perf_value = optimal_perf[1][2] if recall_min else optimal_perf[1][1]
        times_hp_combi[solution] = []
        if results and best_perf_value > 0:
            times_perf = optimal_hyperparameters_for_times(results, recall_min=recall_min, qps_min=qps_min)
            times_hp_combi[solution] = [perf/best_perf_value for perf in times_perf]
        else:
            times_hp_combi[solution] = [0.0] * 4

    spearman_combi = calculate_spearman_for_results(results_combi, recall_min, qps_min)
    return times_hp_combi, spearman_combi

def main():
    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner"]
    IMPLS = ["faiss", "hnswlib"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular"]
    SAMPLING_COUNT = [10]
    RECALL_MINS = [0.90, 0.925, 0.95, 0.975, 0.99]

    tasks = []
    for impl, dataset, sampling_count in itertools.product(IMPLS, DATASETS, SAMPLING_COUNT):
        for recall_min in RECALL_MINS:
            tasks.append((impl, dataset, SOLUTIONS, recall_min, None, sampling_count))
        for qps_min in get_qps_metrics_dataset(impl, dataset):
            tasks.append((impl, dataset, SOLUTIONS, None, qps_min, sampling_count))

    try:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        print(f"Starting parallel processing with {num_processes} processes for {len(tasks)} tasks.")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_single_metric, tasks)

        solution_perf = {s: [0.0]*4 for s in SOLUTIONS}
        solution_spearman = {s: [] for s in SOLUTIONS}
        valid_tasks = len(tasks)

        for times_combi, spearman_combi in results:
            if times_combi["brute_force"] == [0.0]*4:
                valid_tasks -= 1
                continue
            for solution, perf in times_combi.items():
                solution_perf[solution] = [x + y for x, y in zip(solution_perf[solution], perf)]
            for solution, rho in spearman_combi.items():
                if rho is not None:
                    solution_spearman[solution].append(rho)

        for solution in solution_perf:
            solution_perf[solution] = [x/valid_tasks for x in solution_perf[solution]]

        print("\nAverage performance per solution:")
        for solution, perf in solution_perf.items():
            print(solution, ",".join(f"{x:.4f}" for x in perf))

        print("\nAverage Spearman correlation per solution:")
        for solution, rhos in solution_spearman.items():
            if rhos:
                print(f"{solution}: {np.mean(rhos):.4f}")
            else:
                print(f"{solution}: No valid results.")

    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        traceback.print_exc()

    print("\n--- All tasks finished successfully. ---")

if __name__ == "__main__":
    main()

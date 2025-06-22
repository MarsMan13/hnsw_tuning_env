import itertools
import multiprocessing
import os

from src.constants import TUNING_BUDGET, SEED
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.brute_force.run import run as brute_force
from src.solutions.random_search.run import run as random_search
from src.solutions.vd_tuner.run import run as vd_tuner
from src.solutions.our_solution.run import run as our_solution
from src.solutions.grid_search.run import run as grid_search

## Configuration lists (these can remain global or passed as arguments to run_experiments)
IMPLS = [
    # "hnswlib",
    "faiss",
]
DATASETS = [
    "nytimes-256-angular",
    "glove-100-angular",
    "sift-128-euclidean",
    # "youtube-1024-angular",
    # "msmarco-384-angular",
    # "dbpediaentity-768-angular",
]
SOLUTIONS = [
    # (brute_force, "brute_force"),
    (grid_search, "grid_search"),
    (random_search, "random_search"),
    # (vd_tuner, "vd_tuner"),
    (our_solution, "our_solution"),
]
RECALL_MINS = [
    0.90,
    0.95,
    # 0.975,
]
QPS_MINS = [
    # 5000,
    # 20000,
    # 30000,
]
SAMPLING_COUNT = [
    10,
]
####

def worker_function(params):
    impl, dataset, solution_func, solution_name, recall_min, qps_min, sampling_count = params
    try:
        results = solution_func(
            impl=impl, dataset=dataset, recall_min=recall_min, qps_min=qps_min, 
            sampling_count=sampling_count, env=(TUNING_BUDGET, SEED)
        )
        return {
            "solution": solution_name,
            "impl": impl,
            "dataset": dataset,
            "recall_min": recall_min,
            "qps_min": qps_min,
            "sampling_count": sampling_count,
            "results": results,
        }
    except Exception as e:
        print(f"Error in {solution_name} for {impl} on {dataset}: {e}")
        return None 

def run_experiments(
    implements: list, 
    datasets: list, 
    solutions: list, 
    recall_mins: list, 
    qps_mins: list, 
    sampling_counts: list,
    num_cores: int = 6 # Default to 6 cores, can be os.cpu_count()
):
    multiprocessing.set_start_method('spawn', force=True) 

    # Generate all combinations of parameters for recall-based and QPS-based tuning
    all_combinations = list(itertools.product(implements, datasets, solutions, recall_mins, [None], sampling_counts))
    all_combinations += list(itertools.product(implements, datasets, solutions, [None], qps_mins, sampling_counts))

    # Prepare tasks for the worker function
    tasks = [
        (impl, dataset, solution_func, solution_name, recall_min, qps_min, sampling_count)
        for impl, dataset, (solution_func, solution_name), recall_min, qps_min, sampling_count in all_combinations
    ]

    print(f"Using {num_cores} cores for parallel processing.")
    
    # Use multiprocessing.Pool to run tasks in parallel
    with multiprocessing.Pool(processes=num_cores, maxtasksperchild=1) as pool:
        # maxtasksperchild=1 ensures each child process is fresh after one task,
        # which can help with memory leaks or resource cleanup, though it adds overhead.
        for result in pool.imap_unordered(worker_function, tasks):
            if result is not None:
                print(f"Completed: {result['solution']} for {result['impl']} on {result['dataset']}")
                # Process and print optimal hyperparameters
                print_optimal_hyperparameters(
                    result['results'], 
                    recall_min=result['recall_min'], 
                    qps_min=result['qps_min']
                )
                # Post-process results (e.g., saving plots/data)
                postprocess_results(
                    result["results"],
                    solution=result["solution"],
                    impl=result["impl"],
                    dataset=result["dataset"],
                    recall_min=result["recall_min"],
                    qps_min=result["qps_min"],
                    tuning_budget=TUNING_BUDGET, # TUNING_BUDGET is from src.constants
                    sampling_count=result["sampling_count"],
                )
            else:
                print("Error in processing a task, skipping...")
    
    print("All tasks completed.")

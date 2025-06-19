import itertools
import multiprocessing

from src.constants import TUNING_BUDGET
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.brute_force.run import run as brute_force
from src.solutions.random_search.run import run as random_search
from src.solutions.vd_tuner.run import run as vd_tuner
from src.solutions.our_solution.run import run as our_solution
from src.solutions.grid_search.run import run as grid_search

IMPLS = [
    # "hnswlib",
    "faiss",
]
DATASETS = [
    "nytimes-256-angular",
    # "glove-100-angular",
    # "sift-128-euclidean",
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
    # 0.90,
    # 0.90,
    # 0.975,
]
QPS_MINS = [
    # 5000,
    # 20000,
    30000,
]

def worker_function(params):
    impl, dataset, solution_func, solution_name, recall_min, qps_min = params
    # try:
    results = solution_func(impl=impl, dataset=dataset, recall_min=recall_min, qps_min=qps_min)
    return {
        "solution": solution_name,
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results,
    }
    # except Exception as e:
    #     print(f"Error in {solution_name} for {impl} on {dataset}: {e}")
    #     raise e
    #     return None

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    all_combinations = list(itertools.product(IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, [None]))
    all_combinations += list(itertools.product(IMPLS, DATASETS, SOLUTIONS, [None], QPS_MINS))
    tasks = [
        (impl, dataset, solution_func, solution_name, recall_min, qps_min)
        for impl, dataset, (solution_func, solution_name), recall_min, qps_min in all_combinations
    ]
    # num_cores_to_use = os.cpu_count() or 4
    num_cores_to_use = 6
    print(f"Using {num_cores_to_use} cores for parallel processing.")
    
    with multiprocessing.Pool(processes=num_cores_to_use, maxtasksperchild=1) as pool:
        process_results = []
        for result in pool.imap_unordered(
            worker_function, tasks
        ):
            if result is not None:
                print(f"Completed: {result['solution']} for {result['impl']} on {result['dataset']}")
                print_optimal_hyperparameters(
                    result['results'], 
                    recall_min=result['recall_min'], 
                    qps_min=result['qps_min']
                )
                postprocess_results(
                    result["results"],
                    solution=result["solution"],
                    impl=result["impl"],
                    dataset=result["dataset"],
                    recall_min=result["recall_min"],
                    qps_min=result["qps_min"],
                    tuning_budget=TUNING_BUDGET,
                )
            else:
                print("Error in processing a task, skipping...")
    print("All tasks completed.")

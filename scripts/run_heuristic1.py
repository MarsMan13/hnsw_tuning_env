import itertools

from scripts import run_experiments
from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, QPS_MINS

from src.constants import MAX_SAMPLING_COUNT
from src.solutions.our_solution.run import run as our_solution
from src.solutions.random_search.run_heuristic1 import run as random_search
from src.solutions.grid_search.run_heuristic1 import run as grid_search
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

if __name__ == "__main__":
    IMPLS = [
        # "hnswlib",
        "faiss",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
        # "msmarco-384-angular",
        # "dbpediaentity-768-angular",
    ]
    SOLUTIONS = [
        (our_solution, "our_solution"),
        (grid_search, "grid_search_heuristic"),
        (random_search, "random_search_heuristic"),
    ]
    RECALL_MINS = [
        0.90,
        0.95,
        0.975,
    ]
    SAMPLING_COUNT = [
        # 1,
        # 3,
        # 5,
        10,
    ]

    tasks = []
    # Case 1: when recall_min is active
    for impl, dataset, (solution_func, solution_name), sampling_count, recall_min in itertools.product(
        IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT, RECALL_MINS
    ):
        tasks.append((impl, dataset, solution_func, solution_name, recall_min, None, sampling_count))
    # Case 2: when qps_min is active
    # for impl, dataset, (solution_func, solution_name), sampling_count in itertools.product(
    #     IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT
    # ):
    #     for qps_min in get_qps_metrics_dataset(dataset):
    #         tasks.append((impl, dataset, solution_func, solution_name, None, qps_min, sampling_count))
    run_experiments(tasks=tasks)

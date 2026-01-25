import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import itertools

from scripts import run_experiments
from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, QPS_MINS

from src.solutions.brute_force.run import run as brute_force
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

if __name__ == "__main__":
    IMPLS = [
        "faiss",
        # "hnswlib",
    ]
    DATASETS = [
        # "nytimes-256-angular-100p",
        # "nytimes-256-angular-50p",
        # "nytimes-256-angular-10p",
        # "nytimes-256-angular-5p",
        # "nytimes-256-angular-1p",
        # "nytimes-256-angular-1p-hnswlib-random",
        # "nytimes-256-angular-10p-hnswlib-random",
        # "nytimes-256-angular-100p-hnswlib-random",
        "synthetic-128-angular-100p",
        "synthetic-128-angular-10p",
        "synthetic-128-angular-1p",
    ]
    SOLUTIONS = [
        (brute_force, "brute_force"),
    ]
    RECALL_MINS = [
        # 0.85,
        # 0.875,
        # 0.90,
        # 0.925,
        0.95,
        # 0.975,
        # 0.99,
    ]
    SAMPLING_COUNT = [
        10,
    ]

    tasks = []
    # Case 1: when recall_min is active
    for impl, dataset, (solution_func, solution_name), sampling_count, recall_min in itertools.product(
        IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT, RECALL_MINS
    ):
        tasks.append((impl, dataset, solution_func, solution_name, recall_min, None, sampling_count))
    # Case 2: when qps_min is active
    for impl, dataset, (solution_func, solution_name), sampling_count in itertools.product(
        IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT
    ):
        for qps_min in (30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000):
        # for qps_min in get_qps_metrics_dataset(impl, dataset):
            tasks.append((impl, dataset, solution_func, solution_name, None, qps_min, sampling_count))
    run_experiments(tasks=tasks)

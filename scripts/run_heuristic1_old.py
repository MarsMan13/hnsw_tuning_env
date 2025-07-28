import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import itertools

from scripts import run_experiments
from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, TUNING_BUDGET, SEED

from src.solutions.our_solution.stats import Stats
from src.solutions.our_solution.run import run as run 
from src.solutions.our_solution.run_heuristic1 import run as run_heuristic1
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset
from src.solutions import postprocess_results
from src.utils import is_already_saved
    
SOLUTIONS = [
    (run, "our_solution"),
    (run_heuristic1, "test_solution2"),
]

def worker_function(params):
    impl, dataset, recall_min, qps_min, sampling_count = params

    # run.py
    _, stats1, __ = run(
        impl=impl, dataset=dataset, recall_min=recall_min, qps_min=qps_min,
        sampling_count=sampling_count, stats=True
    )
    stats_dict = stats1.stats
    # run_heuristic1.py
    _, stats2 = run_heuristic1(
        impl=impl, dataset=dataset, recall_min=recall_min, qps_min=qps_min,
        sampling_count=sampling_count, stats=True
    )
    stats_dict2 = stats2.stats
    # postprocess results
    Stats.compare_stats(stats_dict, stats_dict2, heuristic_type="heuristic1", impl=impl, dataset=dataset, 
                        recall_min=recall_min, qps_min=qps_min)

if __name__ == "__main__":
    IMPLS = [
        "hnswlib",
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
    RECALL_MINS = [
        0.90,
        0.925,
        0.95,
        0.975,
        0.99
    ]
    SAMPLING_COUNT = [
        10,
    ]

    # Case 1: when recall_min is active
    tasks = []
    for impl, dataset, sampling_count, recall_min in itertools.product(
        IMPLS, DATASETS, SAMPLING_COUNT, RECALL_MINS
    ):
        tasks.append((impl, dataset, recall_min, None, sampling_count))
    # Case 2: when qps_min is active
    for impl, dataset, (solution_func, solution_name), sampling_count in itertools.product(
        IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT
    ):
        qps_mins= get_qps_metrics_dataset(impl=impl, dataset=dataset)
        for qps_min in qps_mins:    
            tasks.append((impl, dataset, None, qps_min, sampling_count))
    for task in tasks:
        worker_function(task)
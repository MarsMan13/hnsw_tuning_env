import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import itertools

from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, TUNING_BUDGET, SEED

from src.solutions.our_solution.stats import Stats
from src.solutions.our_solution.run_all import run as run_all
from src.solutions.our_solution.run_heuristic0 import run as run_heuristic0
from src.solutions.our_solution.run_heuristic1 import run as run_heuristic1
from src.solutions.our_solution.run_heuristic2 import run as run_heuristic2
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset
from src.solutions import postprocess_results, print_optimal_hyperparameters

DEBUG_OUTPUT = ""

def worker_function(
        impl, dataset, recall_min=None, qps_min=None    
    ):
    global DEBUG_OUTPUT
    get_perf = lambda x: x[1][2] if recall_min is not None else x[1][1]
    # run.py
    time_stats = []
    perf_stats = []
    for run in [run_heuristic0, run_heuristic1, run_heuristic2, run_all]:
        results = run(impl=impl, dataset=dataset, recall_min=recall_min, qps_min=qps_min,)
        opt, _ = print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)
        if opt is None:
            opt_time, opt_perf = TUNING_BUDGET, 0.0
            DEBUG_OUTPUT += "No optimal hyperparameter found for " 
        else:
            opt_time, opt_perf = opt[1][0], get_perf(opt)
        opt_time = results[-1][1][0]
        DEBUG_OUTPUT += f"{impl},{dataset},{run.__name__},{opt_time},{opt_perf}\n"
        print(opt_time, opt_perf)
        time_stats.append(opt_time)
        perf_stats.append(opt_perf)
    # normalize each stats by the first one
    if perf_stats[0] == 0:
        return [(0.0, i) for i in range(len(time_stats))], [(0.0, i) for i in range(len(perf_stats))]
    time_stats = [x / time_stats[0] for i, x in enumerate(time_stats)]
    perf_stats = [x / perf_stats[0] for i, x in enumerate(perf_stats)]
    return time_stats, perf_stats
        

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
        "deep1M-256-angular",
        # "msmarco-384-angular",
        # "dbpediaentity-768-angular",
    ]
    RECALL_MIN = 0.975
    QPS_MIN = "q75"
    output = ""
    for dataset in DATASETS:
        sum_time_stats = [0.0] * 4
        sum_perfs_stats = [0.0] * 4
        time_stats, perf_stats = worker_function(
            "hnswlib", dataset, RECALL_MIN, None
        )
        for i in range(len(time_stats)):
            sum_time_stats[i] += time_stats[i]
            sum_perfs_stats[i] += perf_stats[i]
        time_stats, perf_stats = worker_function(
            "faiss", dataset, RECALL_MIN, None
        )
        for i in range(len(time_stats)):    
            sum_time_stats[i] += time_stats[i]
            sum_perfs_stats[i] += perf_stats[i]
        time_stats, perf_stats = worker_function(
            "hnswlib", dataset, None, 
            get_qps_metrics_dataset(impl="hnswlib", dataset=dataset, ret_dict=True)[QPS_MIN], 
        )
        for i in range(len(time_stats)):    
            sum_time_stats[i] += time_stats[i]
            sum_perfs_stats[i] += perf_stats[i]
        time_stats, perf_stats = worker_function(
            "faiss", dataset, None, 
            get_qps_metrics_dataset(impl="faiss", dataset=dataset, ret_dict=True)[QPS_MIN], 
        )
        for i in range(len(time_stats)):    
            sum_time_stats[i] += time_stats[i]
            sum_perfs_stats[i] += perf_stats[i]
        ###
        DIV_FACTOR = 4
        for i in range(len(time_stats)):
            sum_time_stats[i] /= DIV_FACTOR
            sum_perfs_stats[i] /= DIV_FACTOR
        output += f"{dataset},{','.join([f'{x:.4f}' for x in sum_time_stats])},{','.join([f'{x:.4f}' for x in sum_perfs_stats])}\n"
    print(output)

    print("Debug Output:")
    print(DEBUG_OUTPUT)
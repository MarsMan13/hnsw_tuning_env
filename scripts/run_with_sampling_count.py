import itertools
import multiprocessing

from sympy import im

from src.constants import TUNING_BUDGET
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.brute_force.run import run as brute_force
from src.solutions.random_search.run import run as random_search
from src.solutions.vd_tuner.run import run as vd_tuner
from src.solutions.our_solution.run import run as our_solution
from src.solutions.grid_search.run import run as grid_search


SAMPLING_COUNT = [
    None,
    1,
    3,
    5,
    10
]

def worker_function(params):
    impl, dataset, solution_func, solution_name, recall_min, qps_min, test_sampling_count = params
    

if __name__ == "__main__":
    all_combinations = list(itertools.product(
        IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, QPS_MINS, TEST_SAMPING_COUNT
    ))
    tasks = [
        (impl, dataset, solution_func, solution_name, recall_min, qps_min, test_sampling_count)
        for impl, dataset, (solution_func, solution_name), recall_min, qps_min, test_sampling_count in all_combinations
    ]
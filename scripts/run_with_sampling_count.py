from scripts import run_experiments_from_list
from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, QPS_MINS

from src.constants import MAX_SAMPLING_COUNT
from src.solutions.brute_force.run import run as brute_force
from src.solutions.random_search.run import run as random_search
from src.solutions.vd_tuner.run import run as vd_tuner
from src.solutions.our_solution.run import run as our_solution
from src.solutions.grid_search.run import run as grid_search

SAMPLING_COUNT = [
    10,
    1,
    3, 
    5,
]

if __name__ == "__main__":
    run_experiments_from_list(
        implements=IMPLS,
        datasets=DATASETS,
        solutions=SOLUTIONS,
        recall_mins=RECALL_MINS,
        qps_mins=QPS_MINS,
        sampling_counts=SAMPLING_COUNT,
        num_cores=6
    )

from main.solutions import print_optimal_hyperparameters
from static.ground_truths import GroundTruth
from main.constants import IMPL, DATASET, SEED, TUNING_BUDGET
from main.utils import plot_timestamp, save_search_results
import random

RECALL_MIN = 0.90

def run(recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl=IMPL, dataset=DATASET)
    random.seed(SEED)
    results = []
    while True:
        M = random.randint(4, 64)
        efC = random.randint(8, 512)
        efS = random.randint(10, 1024)
        recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS)
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out")
            break
        results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    print_optimal_hyperparameters(results, recall_min=recall_min)
    return results

if __name__ == "__main__":
    results = run()
    save_search_results(results, solution="brute_force", filename=f"random_search_{IMPL}_{DATASET}_{SEED}.png")
    plot_timestamp(results, solution="brute_force", filename=f"random_search_{IMPL}_{DATASET}_{SEED}.png", min_recall=0.95)
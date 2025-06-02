from main.solutions import postprocess_results, print_optimal_hyperparameters
from static.ground_truths import GroundTruth
from main.constants import IMPL, DATASET, SEED, TUNING_BUDGET, RECALL_MIN
import random

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl=impl, dataset=dataset)
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
    return results

if __name__ == "__main__":
    for RECALL_MIN in [0.90, 0.95, 0.99]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["hnswlib"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(results, solution="random_search", impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)